sudo apt update && sudo apt upgrade

pip3 install flask flask-cors adafruit-circuitpython-dht adafruit-circuitpython-bme280 adafruit-circuitpython-ads1x15 mh-z19

sudo apt install sqlite3
sqlite3 sensors.db "CREATE TABLE IF NOT EXISTS readings (time TEXT, temperature REAL, humidity REAL, co2 INT, gas REAL);"


import time
import board
import json
import sqlite3
import adafruit_dht
import adafruit_bme280
import mh_z19
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from flask import Flask, jsonify


dht = adafruit_dht.DHT11(board.D4)
i2c = board.I2C()
bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)

ads = ADS.ADS1115(i2c)
mq2 = AnalogIn(ads, ADS.P0)


app = Flask(__name__)

@app.route("/latest")
def latest():
    temp, hum, co2, gas = read_all()
    status = ai(temp, co2, gas)
    return jsonify({
        "temperature": temp,
        "humidity": hum,
        "co2": co2,
        "gas": gas,
        "status": status
    })

@app.route("/history")
def history():
    con = sqlite3.connect("sensors.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM readings ORDER BY time DESC LIMIT 100")
    rows = cur.fetchall()
    con.close()
    return jsonify(rows)


def read_all():
    
    try:
        temp = dht.temperature
        hum = dht.humidity
    except:
        temp = hum = None

   
    if temp is None:
        temp = bme.temperature
    if hum is None:
        hum = bme.relative_humidity


    try:
        co2 = mh_z19.read()["co2"]
    except:
        co2 = None

   
    gas = mq2.voltage

    return temp, hum, co2, gas


def ai(temp, co2, gas):
    alerts = []

    if temp and temp > 50:
        alerts.append("🔥 Overheat risk")
    if co2 and co2 > 1500:
        alerts.append("🟣 High CO₂")
    if gas > 1.5:
        alerts.append("⚠️ Gas leak risk")

    return " | ".join(alerts) if alerts else "✔️ Environment Safe"



def log_to_db(temp, hum, co2, gas):
    con = sqlite3.connect("sensors.db")
    cur = con.cursor()
    cur.execute("INSERT INTO readings VALUES (datetime('now'), ?, ?, ?, ?)", (temp, hum, co2, gas))
    con.commit()
    con.close()



def loop():
    while True:
        t, h, c, g = read_all()
        log_to_db(t, h, c, g)
        time.sleep(3)


if __name__ == "__main__":
    import threading

    threading.Thread(target=loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000)

