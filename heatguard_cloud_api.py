from fastapi import FastAPI
from pydantic import BaseModel
from heatguard_ai_system.py import HeatGuardAI
from heatguard_n8n_connector import send_to_n8n

app = FastAPI(title="HeatGuard Cloud API")

ai = HeatGuardAI()

class WeatherInput(BaseModel):
    temperature: float
    humidity: float

@app.get("/")
def root():
    return {"message": "HeatGuard Cloud API is running"}

@app.post("/analyze")
def analyze_weather(data: WeatherInput):
    risk = ai.analyze_temperature(data.temperature, data.humidity)
    return {"risk_level": risk}

@app.post("/notify-n8n")
def notify_n8n(data: WeatherInput):
    risk = ai.analyze_temperature(data.temperature, data.humidity)
    response = send_to_n8n({"risk": risk})
    return {"risk": risk, "n8n_response": response}
heatguard_cloud_api.py