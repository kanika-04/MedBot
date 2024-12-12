import serial
import time

def send_coordinates(x_values, y_values, port='COM6', baud_rate=9600):
    try:
        ser = serial.Serial(port, baud_rate)  
        time.sleep(2)  

        if len(x_values) == len(y_values):
            xy_pairs = ""
            for i in range(len(x_values)):
                xy_pairs += f"{x_values[i]},{y_values[i]};"
            xy_pairs += "\n"
            
            ser.write(xy_pairs.encode())
            print(f"Sent: {xy_pairs}")
            
            while ser.in_waiting > 0:
                print(ser.readline().decode('utf-8').rstrip())
        
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")
