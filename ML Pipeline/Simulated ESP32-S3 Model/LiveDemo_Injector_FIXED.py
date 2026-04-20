"""
LiveDemo_Injector.py
────────────────────────────────────────────────────────────
Real-time GRU anomaly detection + interactive anomaly injection.
All controls live in a single matplotlib window — no tkinter needed.

Feature order (matches LiveDemo3.py and your scaler):
  [0] temp   [1] hr    [2] spo2
  [3] ax     [4] ay    [5] az
  [6] gx     [7] gy    [8] gz
"""

import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import serial
import time

# ─────────────────────────────────────────────────────────────
# 1. Feature indices  (must match your scaler's column order)
# ─────────────────────────────────────────────────────────────
TEMP   = 0
HR     = 1
SPO2   = 2
ACCEL_X, ACCEL_Y, ACCEL_Z = 3, 4, 5
GYRO_X,  GYRO_Y,  GYRO_Z  = 6, 7, 8

# ─────────────────────────────────────────────────────────────
# 2. Anomaly helper functions
# ─────────────────────────────────────────────────────────────

def _blend(row, idx, target, b):
    row = row.copy()
    row[idx] = row[idx] * (1 - b) + target * b
    return row

def _fall_anomaly(row, blend, t):
    row = row.copy()
    if t < 0.5:
        spike = blend * 35.0
        row[ACCEL_X] += spike * np.random.choice([-1, 1])
        row[ACCEL_Y] += spike
        row[ACCEL_Z] += spike * np.random.choice([-1, 1])
        row[GYRO_X]  += blend * 200.0 * np.random.randn()
        row[GYRO_Y]  += blend * 200.0 * np.random.randn()
    else:
        sb = min(1.0, (t - 0.5) * 4.0)
        row[ACCEL_X] = row[ACCEL_X] * (1 - sb)
        row[ACCEL_Y] = row[ACCEL_Y] * (1 - sb) - 9.8 * sb
        row[ACCEL_Z] = row[ACCEL_Z] * (1 - sb)
        row[GYRO_X]  = row[GYRO_X]  * (1 - sb)
        row[GYRO_Y]  = row[GYRO_Y]  * (1 - sb)
        row[GYRO_Z]  = row[GYRO_Z]  * (1 - sb)
    return row

def _tremor_anomaly(row, blend, t):
    row = row.copy()
    freq = 5.0
    phase = 2 * np.pi * freq * t
    row[ACCEL_X] += blend * 1.5  * np.sin(phase)
    row[ACCEL_Y] += blend * 1.5  * np.sin(phase + np.pi / 3)
    row[ACCEL_Z] += blend * 1.5  * np.cos(phase)
    row[GYRO_X]  += blend * 25.0 * np.sin(phase + np.pi / 6)
    row[GYRO_Y]  += blend * 25.0 * np.cos(phase + np.pi / 4)
    return row

def _cardiac_arrest(row, blend, t):
    row = _blend(row, HR,   0.0,  blend)
    row = _blend(row, SPO2, 60.0, blend)
    row = _blend(row, TEMP, 34.5, blend)
    return row

# ─────────────────────────────────────────────────────────────
# 3. Anomaly catalogue
# ─────────────────────────────────────────────────────────────
ANOMALIES = {
    "Tachycardia":    {"color": "#E24B4A", "ramp": 5.0,
                       "fn": lambda r, b, t: _blend(r, HR,   170.0, b)},
    "Bradycardia":    {"color": "#E24B4A", "ramp": 4.0,
                       "fn": lambda r, b, t: _blend(r, HR,    35.0, b)},
    "Hypoxia":        {"color": "#185FA5", "ramp": 6.0,
                       "fn": lambda r, b, t: _blend(r, SPO2,  82.0, b)},
    "Fever":          {"color": "#BA7517", "ramp": 8.0,
                       "fn": lambda r, b, t: _blend(r, TEMP,  39.2, b)},
    "Hypothermia":    {"color": "#185FA5", "ramp": 8.0,
                       "fn": lambda r, b, t: _blend(r, TEMP,  33.5, b)},
    "Fall":           {"color": "#534AB7", "ramp": 0.1,  "fn": _fall_anomaly},
    "Tremor":         {"color": "#534AB7", "ramp": 2.0,  "fn": _tremor_anomaly},
    "Cardiac Arrest": {"color": "#791F1F", "ramp": 3.0,  "fn": _cardiac_arrest},
}

# ─────────────────────────────────────────────────────────────
# 4. Injection state  (mutated by button callbacks)
# ─────────────────────────────────────────────────────────────
injection = {
    "active": None,   # name of active anomaly or None
    "start":  None,   # time.monotonic() when activated
    "ramp":   1.0,
}

def inject(raw_row):
    """Apply active anomaly to a raw (unscaled) row. Returns modified copy."""
    if injection["active"] is None:
        return raw_row
    t     = time.monotonic() - injection["start"]
    blend = min(1.0, t / max(injection["ramp"], 0.001))
    return ANOMALIES[injection["active"]]["fn"](raw_row, blend, t)

def activate(name):
    injection["active"] = name
    injection["start"]  = time.monotonic()
    injection["ramp"]   = ANOMALIES[name]["ramp"]
    status_text.set_text(f"Injecting: {name}")
    status_text.set_color(ANOMALIES[name]["color"])
    fig.canvas.draw_idle()

def deactivate():
    injection["active"] = None
    status_text.set_text("Normal")
    status_text.set_color("#3B6D11")
    fig.canvas.draw_idle()

# ─────────────────────────────────────────────────────────────
# 5. Load assets
# ─────────────────────────────────────────────────────────────
print("Loading model and scaler...")
autoencoder = tf.keras.models.load_model('best_gru_autoencoder.keras')
scaler      = joblib.load('sensor_scaler.gz')

WINDOW_SIZE  = 60
NUM_FEATURES = 9
THRESHOLD    = 0.10

# ─────────────────────────────────────────────────────────────
# 6. Physiological signal generator  (HR + SpO2)
# ─────────────────────────────────────────────────────────────

# Set True  → HR and SpO2 always come from the generator
#             (real IMU + temp still used when serial is live)
# Set False → use whatever the BioHub sends (original behaviour)
USE_SYNTHETIC_BIO = True

class PhysiologicalSignalGenerator:
    """
    Layered oscillator model for realistic resting HR and SpO2 at 20 Hz.

    HR layers
    ─────────
    1. Slow baseline drift — random walk ±8 bpm over minutes
       (posture shifts, thermoregulation, autonomic tone)
    2. Mayer waves — ~0.1 Hz sympathetic oscillation, ±3 bpm
    3. Respiratory Sinus Arrhythmia (RSA) — HR rises on exhale,
       falls on inhale at the breathing frequency, ±6 bpm
    4. Measurement noise + integer quantization  (BioHub output)

    SpO2 layers
    ───────────
    1. Slow perfusion drift — random walk, biased toward normal
    2. Respiratory coupling — same breath phase as RSA but
       anticorrelated (slight dip on inhale, peak on exhale), ±0.4 %
    3. Sensor noise + 0.5 % step quantization  (BioHub resolution)
    """

    SAMPLE_RATE = 20  # Hz

    def __init__(self,
                 hr_baseline  = 72.0,
                 spo2_baseline= 98.0,
                 resp_rate_hz = 0.25):   # ~15 breaths/min
        self._dt           = 1.0 / self.SAMPLE_RATE

        self.hr_baseline   = hr_baseline
        self.spo2_baseline = spo2_baseline

        # ── Shared respiratory oscillator
        self._resp_rate    = resp_rate_hz
        self._resp_phase   = np.random.uniform(0, 2 * np.pi)

        # ── Mayer wave oscillator (~0.1 Hz)
        self._mayer_phase  = np.random.uniform(0, 2 * np.pi)
        self._mayer_rate   = 0.1

        # ── Slow drift state (random walk, mean-reverting)
        self._hr_drift     = 0.0
        self._spo2_drift   = 0.0

    def next(self):
        """Return (hr, spo2) for the current timestep and advance state."""

        # ── Advance oscillators
        self._resp_phase  += 2 * np.pi * self._resp_rate  * self._dt
        self._mayer_phase += 2 * np.pi * self._mayer_rate * self._dt

        resp  = np.sin(self._resp_phase)
        mayer = np.sin(self._mayer_phase)

        # ── Drift: small random walk with soft mean-reversion
        self._hr_drift   += 0.015 * np.random.randn() - 0.002 * self._hr_drift
        self._hr_drift    = np.clip(self._hr_drift, -8.0,  8.0)

        self._spo2_drift += 0.003 * np.random.randn() - 0.005 * self._spo2_drift
        self._spo2_drift  = np.clip(self._spo2_drift, -1.5,  0.5)

        # ── Heart Rate
        hr = (self.hr_baseline
              + self._hr_drift
              + 3.0 * mayer           # Mayer waves:  ±3 bpm
              + 6.0 * resp            # RSA:          ±6 bpm
              + 0.4 * np.random.randn())  # measurement noise
        hr = float(np.clip(round(hr), 40, 180))   # integer BPM

        # ── SpO2  (respiratory coupling anticorrelated with HR RSA)
        spo2 = (self.spo2_baseline
                + self._spo2_drift
                - 0.4 * resp                      # inhale dip, exhale peak
                + 0.15 * np.random.randn())        # sensor noise
        spo2 = np.clip(spo2, 90.0, 100.0)
        spo2 = float(round(spo2 * 2) / 2)         # 0.5 % steps (BioHub res.)

        return hr, spo2


# Instantiate once — maintains oscillator state across all frames
_bio_gen = PhysiologicalSignalGenerator()

# ─────────────────────────────────────────────────────────────
# 7. Serial connection
# ─────────────────────────────────────────────────────────────
print("Connecting to ESP32...")
try:
    ser = serial.Serial('COM5', 115200, timeout=0.1)
    ser.reset_input_buffer()
    # Startup handshake — unconditionally push NORMAL so the display
    # matches Python's initial state regardless of what was on screen before.
    # Small delay lets the ESP32 finish its own setup() before we write.
    time.sleep(2.5)
    ser.write(b'NORMAL\n')
    print("Connected. Sent startup NORMAL.")
except Exception as e:
    print(f"Serial failed: {e}  →  running in demo mode")
    ser = None

def read_serial():
    """
    Reads one line from ESP32.
    Expected CSV: imu_x, imu_y, imu_z, temp, status, hr, spo2  (7 values)
    Returns 9-element array in order: [temp, hr, spo2, ax, ay, az, gx, gy, gz]

    HR and SpO2 are always replaced by _bio_gen when USE_SYNTHETIC_BIO=True.
    Real IMU and temp channels are preserved when serial is connected.
    """
    syn_hr, syn_spo2 = _bio_gen.next()

    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()

            # Skip the CSV header line and separator lines the sketch emits on startup.
            # Note: startswith('-') alone would incorrectly discard valid data lines
            # where the first field (ax) is negative (e.g. "-1.77,0.06,...").
            # Use startswith('---') to match only separator lines like "----------".
            if line.startswith('accel') or line.startswith('---'):
                return None
            d = [float(v) for v in line.split(',')]
            # Sketch output order: ax, ay, az, gx, gy, gz, hr, spo2, temp
            if len(d) == 9:
                ax, ay, az, gx, gy, gz, hw_hr, hw_spo2, temp = d
                hr   = syn_hr   if USE_SYNTHETIC_BIO else hw_hr
                spo2 = syn_spo2 if USE_SYNTHETIC_BIO else hw_spo2
                # Return in pipeline order: [temp, hr, spo2, ax, ay, az, gx, gy, gz]
                return np.array([temp, hr, spo2,
                                 ax, ay, az,
                                 gx, gy, gz], dtype=np.float32)
        except (ValueError, Exception):
            pass

    # Full demo mode — all channels synthetic
    t = time.monotonic()
    return np.array([
        36.6 + 0.1  * np.random.randn(),       # temp
        syn_hr,                                  # hr   ← generator
        syn_spo2,                                # spo2 ← generator
        0.05 * np.random.randn(),               # ax
        0.05 * np.random.randn(),               # ay
        9.80 + 0.05 * np.random.randn(),        # az
        0.30 * np.random.randn(),               # gx
        0.30 * np.random.randn(),               # gy
        0.10 * np.random.randn(),               # gz
    ], dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# 7. Live buffer & plot history
# ─────────────────────────────────────────────────────────────
live_buffer = np.zeros((WINDOW_SIZE, NUM_FEATURES))

# Tracks last alert state so we only write to serial on transitions,
# not every frame — avoids flooding the ESP32's receive buffer
_alert_active = False

def send_display_command(alert: bool):
    """Send ALERT:<name> or NORMAL to the ESP32 over the existing serial connection."""
    if ser is None:
        return
    try:
        if alert:
            name = injection['active'] or 'Anomaly'
            cmd  = f'ALERT:{name}\n'.encode('utf-8')
        else:
            cmd  = b'NORMAL\n'
        ser.write(cmd)
    except Exception as e:
        print(f"[display] Serial write failed: {e}")

HISTORY = 100
plot_time  = list(range(HISTORY))
plot_temp  = [36.6] * HISTORY
plot_hr    = [72.0] * HISTORY
plot_spo2  = [98.0] * HISTORY
plot_error = [0.0]  * HISTORY

# ─────────────────────────────────────────────────────────────
# 8. Figure layout
#    Left column : 4 data subplots
#    Right column: button panel
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.suptitle('Real-Time GRU Anomaly Detection', fontsize=15, fontweight='bold')

# GridSpec: plots on left (width 3), buttons on right (width 1)
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 2, figure=fig,
              left=0.07, right=0.98,
              top=0.93,  bottom=0.04,
              wspace=0.35, hspace=0.45,
              width_ratios=[3, 1])

ax_temp = fig.add_subplot(gs[0, 0])
ax_hr   = fig.add_subplot(gs[1, 0])
ax_spo2 = fig.add_subplot(gs[2, 0])
ax_mae  = fig.add_subplot(gs[3, 0])

# ── Data lines
line_temp,  = ax_temp.plot(plot_time, plot_temp,  color='#E27820', lw=1.5, label='Temp (°C)')
line_hr,    = ax_hr.plot(  plot_time, plot_hr,    color='#E24B4A', lw=1.5, label='Heart Rate (BPM)')
line_spo2,  = ax_spo2.plot(plot_time, plot_spo2,  color='#185FA5', lw=1.5, label='SpO2 (%)')
line_error, = ax_mae.plot( plot_time, plot_error, color='#534AB7', lw=1.5, label='Reconstruction Error (MAE)')

ax_mae.axhline(y=THRESHOLD, color='red', linestyle='--', lw=1, label='Threshold')

for ax, ylabel, ylim, lbl in [
    (ax_temp, '°C',  (20,  50),  line_temp),
    (ax_hr,   'BPM', (30, 180),  line_hr),
    (ax_spo2, '%',   (75, 105),  line_spo2),
    (ax_mae,  'MAE', (0,   0.3), line_error),
]:
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(labelsize=8)

# ── Status text (top of button column)
ax_status = fig.add_subplot(gs[0, 1])
ax_status.axis('off')
status_text = ax_status.text(
    0.5, 0.5, "Normal",
    transform=ax_status.transAxes,
    ha='center', va='center',
    fontsize=11, fontweight='bold',
    color='#3B6D11'
)
ax_status.set_title("Injection Status", fontsize=9, pad=4)

# ── Anomaly buttons  (rows 1–3 of button column)
# Pack all 8 anomaly buttons + 1 Normal button into the right column
btn_axes   = []
btn_objects = {}

n_anomalies  = len(ANOMALIES)
n_buttons    = n_anomalies + 1   # +1 for Normal
col_left     = 0.755
col_width    = 0.22
total_height = 0.62              # vertical span for all buttons
btn_h        = total_height / n_buttons
gap          = 0.005
top_start    = 0.20              # y of first button bottom (from figure bottom)

for i, name in enumerate(list(ANOMALIES.keys()) + ["NORMAL"]):
    y_pos = top_start + (n_buttons - 1 - i) * btn_h
    ax_b  = fig.add_axes([col_left, y_pos + gap, col_width, btn_h - gap * 2])

    if name == "NORMAL":
        color  = "#3B6D11"
        hcolor = "#27500A"
    else:
        color  = ANOMALIES[name]["color"]
        hcolor = color

    btn = Button(ax_b, name,
                 color=color,
                 hovercolor=hcolor)
    btn.label.set_fontsize(8)
    btn.label.set_color('white')
    btn.label.set_fontweight('bold')

    if name == "NORMAL":
        btn.on_clicked(lambda event: deactivate())
    else:
        btn.on_clicked(lambda event, n=name: activate(n))

    btn_objects[name] = btn
    btn_axes.append(ax_b)

plt.figtext(0.865, 0.965, "Inject Anomaly",
            ha='center', va='top', fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────
# 9. Animation update function
# ─────────────────────────────────────────────────────────────
def update(frame):
    global live_buffer, plot_temp, plot_hr, plot_spo2, plot_error

    # 1. Read raw sensor row from serial (or demo)
    # read_serial() returns None when it skips a header/garbage line
    raw_reading = read_serial()
    if raw_reading is None:
        return line_temp, line_hr, line_spo2, line_error

    # 2. Apply anomaly injection BEFORE scaling
    injected = inject(raw_reading)

    # 3. Update plot history with injected (visible) values
    plot_temp.append(injected[TEMP]);  plot_temp.pop(0)
    plot_hr.append(injected[HR]);      plot_hr.pop(0)
    plot_spo2.append(injected[SPO2]);  plot_spo2.pop(0)

    # 4. Scale and slide buffer
    scaled = scaler.transform(injected.reshape(1, -1))[0]
    live_buffer = np.roll(live_buffer, -1, axis=0)
    live_buffer[-1] = scaled

    # 5. Model inference
    input_tensor  = live_buffer.reshape(1, WINDOW_SIZE, NUM_FEATURES)
    reconstruction = autoencoder.predict(input_tensor, verbose=0)
    mae = float(np.mean(np.abs(input_tensor - reconstruction)))

    plot_error.append(mae); plot_error.pop(0)

    # 6. Threshold alert — only send serial command on state change
    global _alert_active
    is_alert = mae > THRESHOLD
    if is_alert:
        ax_mae.set_facecolor('#ffcccc')
        if not _alert_active:
            print(f"ALARM  MAE={mae:.4f}  anomaly={injection['active'] or 'unknown'}")
            send_display_command(alert=True)
            _alert_active = True
    else:
        ax_mae.set_facecolor('white')
        if _alert_active:
            send_display_command(alert=False)
            _alert_active = False

    # 7. Push new data to lines
    line_temp.set_ydata(plot_temp)
    line_hr.set_ydata(plot_hr)
    line_spo2.set_ydata(plot_spo2)
    line_error.set_ydata(plot_error)

    return line_temp, line_hr, line_spo2, line_error

# ─────────────────────────────────────────────────────────────
# 10. Run
# ─────────────────────────────────────────────────────────────
print("Starting live demo...")
ani = animation.FuncAnimation(
    fig, update,
    interval=50,
    blit=False,
    cache_frame_data=False
)
plt.show()
