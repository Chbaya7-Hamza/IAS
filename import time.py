
pip install adafruit-circuitpython-dht adafruit-circuitpython-bme280 mh-z19 adafruit-circuitpython-ads1x15


pip install matplotlib 



import time
import board
import adafruit_dht
import adafruit_bme280
import mh_z19  
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import csv
from datetime import datetime
import os


LOG_FILE = 'sensor_log.csv'
CALIBRATION_FILE = 'mq2_calibration.txt'
READ_INTERVAL = 5  


THRESHOLDS = {
    'co2_max': 1000,       
    'temp_max': 30.0,      
    'mq2_voltage_max': 1.5 
}



dht_device = adafruit_dht.DHT11(board.D4)


i2c = board.I2C()
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)



ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)  


def calibrate_mq2():
       if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            baseline = float(f.read().strip())
        print(f"Loaded MQ-2 baseline from file: {baseline:.2f}V")
        return baseline

    print("Calibrating MQ-2 sensor. Ensure the area has fresh air.")
    print("This will take 30 seconds...")
    readings = []
    for _ in range(30):
        try:
            readings.append(chan.voltage)
        except Exception as e:
            print(f"Error during calibration read: {e}")
        time.sleep(1)
    
    if not readings:
        print("Could not get any readings for calibration. Exiting.")
        return None

    baseline = sum(readings) / len(readings)
    print(f"Calibration complete. Baseline voltage: {baseline:.2f}V")
    
    with open(CALIBRATION_FILE, 'w') as f:
        f.write(str(baseline))
    print(f"Baseline saved to {CALIBRATION_FILE}")
    
    return baseline


def read_sensors(baseline):
        sensor_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dht_temp': None,
        'dht_humidity': None,
        'bme_temp': None,
        'bme_humidity': None,
        'bme_pressure': None,
        'co2_ppm': None,
        'mq2_voltage': None
    }

   
    try:
        sensor_data['dht_temp'] = dht_device.temperature
        sensor_data['dht_humidity'] = dht_device.humidity
    except RuntimeError as error:
        
        print(f"DHT11 error: {error.args[0]}")

  
    try:
        sensor_data['bme_temp'] = bme280.temperature
        sensor_data['bme_humidity'] = bme280.relative_humidity
        sensor_data['bme_pressure'] = bme280.pressure
    except Exception as error:
        print(f"BME280 error: {error}")

   
    try:
        co2_data = mh_z19.read()
        sensor_data['co2_ppm'] = co2_data['co2']
    except Exception as error:
        print(f"MH-Z19 error: {error}")

   
    try:
        sensor_data['mq2_voltage'] = chan.voltage
        if baseline:
            deviation = sensor_data['mq2_voltage'] - baseline
            sensor_data['mq2_deviation'] = deviation
    except Exception as error:
        print(f"MQ-2 error: {error}")
        
    return sensor_data

def log_data_to_csv(data, filename):
    
    fieldnames = ['timestamp', 'dht_temp', 'dht_humidity', 'bme_temp', 'bme_humidity', 'bme_pressure', 'co2_ppm', 'mq2_voltage', 'mq2_deviation']
    
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def check_alerts(data):
   
    alerts = []
    if data['co2_ppm'] and data['co2_ppm'] > THRESHOLDS['co2_max']:
        alerts.append(f"High CO2 level: {data['co2_ppm']} ppm")

    if data['dht_temp'] and data['dht_temp'] > THRESHOLDS['temp_max']:
        alerts.append(f"High temperature: {data['dht_temp']:.1f}°C")

    if data['mq2_voltage'] and data['mq2_voltage'] > THRESHOLDS['mq2_voltage_max']:
        alerts.append(f"Combustible gas detected: {data['mq2_voltage']:.2f}V")

    if alerts:
        print("\n!!! ALERTS DETECTED !!!")
        for alert in alerts:
            print(f"  - {alert}")
        print("-" * 20)



if __name__ == "__main__":
    print("Starting Environmental Monitor...")
    
   
    mq2_baseline = calibrate_mq2()
    if mq2_baseline is None:
        print("MQ-2 calibration failed. Exiting program.")
        exit()

    
    while True:
       
        sensor_readings = read_sensors(mq2_baseline)
        
    
        print("\n--- New Reading ---")
        print(f"Timestamp: {sensor_readings['timestamp']}")
        if sensor_readings['dht_temp'] is not None:
            print(f"DHT11   - Temp: {sensor_readings['dht_temp']:.1f}°C, Humidity: {sensor_readings['dht_humidity']:.1f}%")
        if sensor_readings['bme_temp'] is not None:
            print(f"BME280  - Temp: {sensor_readings['bme_temp']:.1f}°C, Humidity: {sensor_readings['bme_humidity']:.1f}%, Pressure: {sensor_readings['bme_pressure']:.1f} hPa")
        if sensor_readings['co2_ppm'] is not None:
            print(f"MH-Z19  - CO2: {sensor_readings['co2_ppm']} ppm")
        if sensor_readings['mq2_voltage'] is not None:
            print(f"MQ-2    - Voltage: {sensor_readings['mq2_voltage']:.2f}V (Deviation from baseline: {sensor_readings.get('mq2_deviation', 0):.2f}V)")
        
        
        check_alerts(sensor_readings)
        
       
        log_data_to_csv(sensor_readings, LOG_FILE)
        print(f"Data logged to {LOG_FILE}")

        
        time.sleep(READ_INTERVAL)
