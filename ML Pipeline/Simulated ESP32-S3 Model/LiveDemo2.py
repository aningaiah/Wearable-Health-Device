import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import time

# ---------------------------------------------------------
# 1. Load the Assets
# ---------------------------------------------------------
print("Loading model and scaler...")
autoencoder = tf.keras.models.load_model('best_gru_autoencoder.keras')
scaler = joblib.load('sensor_scaler.gz')  # Saved from Phase 4

# Parameters
WINDOW_SIZE = 60
NUM_FEATURES = 9
THRESHOLD = 0.0826 # From Phase 5

# ---------------------------------------------------------
# 2. Initialize Serial Connection
# ---------------------------------------------------------
print("Connecting to ESP32...")
try:
    # IMPORTANT: Change 'COM3' to your actual ESP32 port
    ser = serial.Serial('COM4', 115200, timeout=0.1) 
    ser.reset_input_buffer()
except Exception as e:
    print(f"Failed to connect to Serial: {e}")
    print("Please check your COM port and make sure your serial monitor is closed.")
    exit()

# ---------------------------------------------------------
# 3. Initialize the Live Buffer & UI
# ---------------------------------------------------------
live_buffer = np.zeros((WINDOW_SIZE, NUM_FEATURES))

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Real-Time GRU Anomaly Detection', fontsize=16)

# Lists to hold the plotting data
plot_time = list(range(100))
plot_error = [0] * 100
plot_temp = [37.0] * 100 # Changed to plot Temperature instead of dummy HR

line_temp, = ax1.plot(plot_time, plot_temp, label='Live Temp (°C)', color='orange')
line_error, = ax2.plot(plot_time, plot_error, label='Reconstruction Error (MAE)', color='purple')
threshold_line = ax2.axhline(y=THRESHOLD, color='red', linestyle='--', label='Threshold')

ax1.set_ylim(20, 50) # Temp range roughly 20C to 50C
ax1.set_ylabel('Temperature (°C)')
ax1.legend()

ax2.set_ylim(0, 0.3)
ax2.set_ylabel('MAE')
ax2.legend()

# ---------------------------------------------------------
# 4. Live Data Processing
# ---------------------------------------------------------
def generate_live_reading():
    """Reads a single line of live data from the ESP32 via Serial"""
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            data = [float(val) for val in line.split(',')]
            
            # Expecting 7 values from ESP32: temp, ax, ay, az, gx, gy, gz
            if len(data) == 7:
                temp = data[0]
                # Pad the missing sensors so the 9-feature Autoencoder doesn't crash
                hr = 72.0   
                spo2 = 98.5 
                
                ax, ay, az = data[1], data[2], data[3]
                gx, gy, gz = data[4], data[5], data[6]
                
                return np.array([temp, hr, spo2, ax, ay, az, gx, gy, gz])
                
        except ValueError:
            pass # Catch incomplete serial lines
        except Exception as e:
            print(f"Serial read error: {e}")

    # Fallback return: If no data is ready, return a baseline array 
    return np.array([37.0, 72.0, 98.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

def update(frame):
    global live_buffer, plot_error, plot_temp
    
    # 1. Get a new raw reading from the serial port
    new_reading = generate_live_reading()
    
    # Update plot data for the top chart (Temperature is index 0)
    plot_temp.append(new_reading[0])
    plot_temp.pop(0)
    
    # 2. Scale the single reading
    scaled_reading = scaler.transform(new_reading.reshape(1, -1))[0]
    
    # 3. Slide the buffer: remove oldest, append newest
    live_buffer = np.roll(live_buffer, -1, axis=0)
    live_buffer[-1] = scaled_reading
    
    # 4. Format for the model (Batch, Timesteps, Features)
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
        print(f"ALARM! MAE: {mae:.4f}")
    else:
        ax2.set_facecolor('white')
    
    # Update lines
    line_temp.set_ydata(plot_temp)
    line_error.set_ydata(plot_error)
    
    return line_temp, line_error,

# ---------------------------------------------------------
# 5. Run the Simulation
# ---------------------------------------------------------
print("Starting live simulation...")
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False, cache_frame_data=False)
plt.show()