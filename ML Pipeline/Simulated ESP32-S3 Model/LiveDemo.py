import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# 1. Load the Assets
print("Loading model and scaler...")
autoencoder = tf.keras.models.load_model('best_gru_autoencoder.keras')
scaler = joblib.load('sensor_scaler.gz')  # Saved from Phase 4

# Parameters
WINDOW_SIZE = 60
NUM_FEATURES = 9
THRESHOLD = 0.0826 # From Phase 5

# 2. Initialize the Live Buffer
# We start with a buffer full of zeros, or you can pre-fill it with a normal baseline
live_buffer = np.zeros((WINDOW_SIZE, NUM_FEATURES))

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Real-Time GRU Anomaly Detection Simulator', fontsize=16)

# Lists to hold the plotting data
plot_time = list(range(100))
plot_error = [0] * 100
plot_hr = [72] * 100 # Just plotting HR for visual simplicity

line_hr, = ax1.plot(plot_time, plot_hr, label='Simulated HR (BPM)', color='blue')
line_error, = ax2.plot(plot_time, plot_error, label='Reconstruction Error (MAE)', color='purple')
threshold_line = ax2.axhline(y=THRESHOLD, color='red', linestyle='--', label='Threshold')

ax1.set_ylim(50, 130)
ax1.set_ylabel('Heart Rate')
ax1.legend()

ax2.set_ylim(0, 0.3)
ax2.set_ylabel('MAE')
ax2.legend()

anomaly_trigger_time = 50 # Inject an anomaly at tick 50

def generate_live_reading(tick):
    """Simulates a single 20Hz reading from the I2C sensors"""
    # Base resting values
    temp = 37.0 + np.random.normal(0, 0.05)
    hr = 72.0 + np.random.normal(0, 1.0)
    spo2 = 98.5 + np.random.normal(0, 0.2)
    ax, ay, az = np.random.normal(0, 0.02), np.random.normal(0, 0.02), 1.0 + np.random.normal(0, 0.02)
    gx, gy, gz = np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 0.5)

    # Inject Anomaly dynamically
    if tick > anomaly_trigger_time and tick < anomaly_trigger_time + 60:
        hr += 30.0 # Sudden HR spike
        spo2 -= 10.0 # Sudden SpO2 drop

    return np.array([temp, hr, spo2, ax, ay, az, gx, gy, gz])

def update(frame):
    global live_buffer, plot_error, plot_hr
    
    # 1. Get a new raw reading
    new_reading = generate_live_reading(frame)
    
    # Update plot data for the top chart
    plot_hr.append(new_reading[1])
    plot_hr.pop(0)
    
    # 2. Scale the single reading (needs to be reshaped for the scaler)
    scaled_reading = scaler.transform(new_reading.reshape(1, -1))[0]
    
    # 3. Slide the buffer: remove oldest (index 0), append newest
    live_buffer = np.roll(live_buffer, -1, axis=0)
    live_buffer[-1] = scaled_reading
    
    # 4. Only run inference if the buffer is "full" of real data 
    # (In this sim it's always full, but good practice for real life)
    input_tensor = live_buffer.reshape(1, WINDOW_SIZE, NUM_FEATURES)
    
    # 5. Model Prediction
    reconstruction = autoencoder.predict(input_tensor, verbose=0)
    
    # 6. Calculate MAE
    mae = np.mean(np.abs(input_tensor - reconstruction))
    
    # Update plot data for the bottom chart
    plot_error.append(mae)
    plot_error.pop(0)
    
    # 7. Check Threshold & Update UI
    if mae > THRESHOLD:
        ax2.set_facecolor('#ffcccc') # Flash red
        print(f"[{frame}] ALARM! MAE: {mae:.4f}")
    else:
        ax2.set_facecolor('white')
    
    # Update lines
    line_hr.set_ydata(plot_hr)
    line_error.set_ydata(plot_error)
    
    return line_hr, line_error,

print("Starting live simulation...")
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()