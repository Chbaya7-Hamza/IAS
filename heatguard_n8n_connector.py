import requests
import json
from datetime import datetime


class HeatGuardN8NConnector:
    """Connects HeatGuard AI to n8n workflows"""

    def __init__(self, n8n_base_url="http://localhost:5678"):
        self.base_url = n8n_base_url
        self.webhooks = {
            'critical': f"{n8n_base_url}/webhook/heatguard-critical",
            'report': f"{n8n_base_url}/webhook/heatguard-daily-report",
            'maintenance': f"{n8n_base_url}/webhook/heatguard-maintenance"
        }

    def send_critical_alert(self, alert_data):
        """Send critical alert to n8n"""
        try:
            response = requests.post(
                self.webhooks['critical'],
                json=alert_data,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Critical alert sent to n8n")
                return True
            else:
                print(f"✗ n8n error: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot reach n8n: {e}")
            return False

    def send_daily_report(self, report_data):
        """Send daily report to n8n"""
        try:
            response = requests.post(
                self.webhooks['report'],
                json=report_data,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Daily report sent to n8n")
                return True
            return False
        except Exception as e:
            print(f"✗ Error sending report: {e}")
            return False

    def send_maintenance_alert(self, maintenance_data):
        """Send maintenance prediction to n8n"""
        try:
            response = requests.post(
                self.webhooks['maintenance'],
                json=maintenance_data,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Maintenance alert sent to n8n")
                return True
            return False
        except Exception as e:
            print(f"✗ Error sending maintenance alert: {e}")
            return False


# Example Usage
if __name__ == "__main__":
    # Initialize connector
    connector = HeatGuardN8NConnector()

    # Test 1: Send critical alert
    critical_alert = {
        "severity": "critical",
        "temperature": 85.0,
        "gas_level": 70.0,
        "air_pressure": 1013.25,
        "humidity": 90.0,
        "sensor_id": "TEST_001",
        "timestamp": datetime.now().isoformat()
    }
    connector.send_critical_alert(critical_alert)

    # Test 2: Send daily report
    daily_report = {
        "date": datetime.now().date().isoformat(),
        "summary": {
            "total_readings": 288,
            "critical_alerts": 2,
            "warning_alerts": 8
        },
        "sensor_statistics": {
            "temperature_mean": 32.5,
            "temperature_max": 78.2
        },
        "ai_safety_analysis": "System operating normally with occasional temperature spikes.",
        "recommendations": ["Monitor cooling system", "All clear"],
        "kpis": {
            "anomaly_detection_accuracy": ">95%",
            "response_time": "< 3s"
        }
    }
    connector.send_daily_report(daily_report)

    # Test 3: Send maintenance alert
    maintenance_alert = {
        "status": "planned",
        "explanation": "Cooling efficiency decreasing gradually",
        "confidence": "medium"
    }
    connector.send_maintenance_alert(maintenance_alert)