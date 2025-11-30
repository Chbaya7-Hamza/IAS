# heatguard_ai_system.py
class HeatGuardAISystem:
    def __init__(self, enable_n8n=False):
        self.enable_n8n = enable_n8n
        print(f"Dummy HeatGuard AI system initialized. n8n enabled: {self.enable_n8n}")

    def predict(self, data):
        print(f"Received data for prediction: {data}")
        return {"prediction": "dummy_result"}

    def process_sensor_data(self, sensor_data):
        print(f"Processing sensor data: {sensor_data}")

        temp = sensor_data.get("temperature", 0)
        gas = sensor_data.get("gas_level", 0)

        result = {"alert": None, "severity": "normal", "alerts": []}

        if temp > 80 or gas > 50:
            result["alert"] = "Critical alert triggered!"
            result["severity"] = "critical"
            result["alerts"].append({"message": "Critical alert triggered!"})
        elif temp > 60 or gas > 30:
            result["alert"] = "Warning alert triggered!"
            result["severity"] = "warning"
            result["alerts"].append({"message": "Warning alert triggered!"})
        else:
            result["alert"] = None
            result["severity"] = "normal"

        return result

    def generate_daily_report(self):
        print("Generating dummy daily report...")
        normal = 10
        warnings = 3
        critical = 2
        total = normal + warnings + critical

        # Simulate a report dictionary
        report = {
            "date": "2025-11-29",
            "summary": {
                "normal_readings": normal,
                "warnings": warnings,
                "critical": critical,
                "total_readings": total,
                "critical_alerts": critical,
                "warning_alerts": warnings  # add this key
            },
            "alerts": [
                {"message": "Critical alert triggered!"},
                {"message": "Warning alert triggered!"}
            ]
        }
        return report

