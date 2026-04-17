import serial

# Change 'COM3' to your Arduino's specific port (e.g., '/dev/cu.usbmodem...' on Mac)
arduino_port = 'COM7' 
baud_rate = 115200

# Open the serial port
ser = serial.Serial(arduino_port, baud_rate)
print(f"Listening on {arduino_port}... Press Ctrl+C to stop.")

# Open a new CSV file to write the data
with open('sensor_log.csv', 'w') as file:
    try:
        while True:
            # Read a line from the serial port, decode it, and remove extra whitespace
            line = ser.readline().decode('utf-8').strip()
            
            # Print to the console so you can see it
            print(line)
            
            # Write the line to the CSV file with a newline character
            file.write(line + '\n')
            
    except KeyboardInterrupt:
        print("\nData logging stopped. File saved as sensor_log.csv")
        ser.close()