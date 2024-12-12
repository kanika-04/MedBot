import serial
import time

ser = serial.Serial('COM7', 9600)  

time.sleep(2)  

x_values = [0, 1, 1, 2, 2, 3, 3]
y_values = [1, 0, 1, 0, 1, 0, 1]
if len(x_values) == len(y_values):
    # Create a string in the format "x1,y1;x2,y2;x3,y3\n"
    xy_pairs = ""
    for i in range(len(x_values)):
        xy_pairs += f"{x_values[i]},{y_values[i]};"
    
    xy_pairs += "\n"
    
    ser.write(xy_pairs.encode())
    print(f"Sent: {xy_pairs}")

while ser.in_waiting > 0:
    print(ser.readline().decode('utf-8').rstrip())

# Close the connection
ser.close()
