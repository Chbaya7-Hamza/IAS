"""
Test HeatGuard AI + n8n Integration
"""

import requests
import json
from datetime import datetime
import time

N8N_WEBHOOK = "http://localhost:5678/webhook/test"


def test_n8n_connection():
    """Test if n8n is reachable"""
    print("=" * 70)
    print("Testing n8n Connection")
    print("=" * 70)

    try:
        response = requests.post(
            N8N_WEBHOOK,
            json={"test": "connection check"},
            timeout=5
        )

        if response.status_code == 200:
            print("✅ n8n is reachable!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ n8n returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach n8n: {e}")
        print("\nMake sure n8n is running:")
        print("  1. Open PowerShell")
        print("  2. Run: npx n8n")
        print("  3. Wait for: 'n8n ready on port 5678'")
        return False


def send_alert_to_n8n(alert_data):
    """Send alert to n8n"""
    try:
        response = requests.post(
            N8N_WEBHOOK,
            json=alert_data,
            timeout=5
        )

        if response.status_code == 200:
            print(f"  ✅ Sent to n8n: {alert_data['severity']}")
            return True
        else:
            print(f"  ❌ n8n error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_alert_scenarios():
    """Test different alert scenarios"""
    print("\n" + "=" * 70)
    print("Testing Alert Scenarios")
    print("=" * 70 + "\n")

    scenarios = [
        {
            "name": "Normal Reading",
            "data": {
                "severity": "normal",
                "temperature": 25.5,
                "gas_level": 15.0,
                "air_pressure": 1013.25,
                "humidity": 45.0,
                "sensor_id": "TEST_001",
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "name": "Warning Alert",
            "data": {
                "severity": "warning",
                "temperature": 65.0,
                "gas_level": 35.0,
                "air_pressure": 1013.25,
                "humidity": 75.0,
                "sensor_id": "TEST_001",
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "name": "Critical Alert",
            "data": {
                "severity": "critical",
                "temperature": 85.0,
                "gas_level": 70.0,
                "air_pressure": 1013.25,
                "humidity": 90.0,
                "sensor_id": "TEST_001",
                "timestamp": datetime.now().isoformat(),
                "message": "CRITICAL: Multiple thresholds exceeded!"
            }
        }
    ]

    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        send_alert_to_n8n(scenario['data'])
        time.sleep(1)  # Wait 1 second between tests

    print("\n✅ All scenarios sent to n8n!")
    print("Check n8n Executions to see the data")


def main():
    print("\n" + "=" * 70)
    print("🔗 HeatGuard AI + n8n Integration Test")
    print("=" * 70 + "\n")

    # Test connection
    if not test_n8n_connection():
        print("\n⚠️  Start n8n first, then run this test again")
        return

    # Test scenarios
    test_alert_scenarios()

    print("\n" + "=" * 70)
    print("✅ Integration Test Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Open n8n browser: http://localhost:5678")
    print("2. Click 'Executions' (left sidebar)")
    print("3. See your alerts processed by n8n!")
    print("4. Check the Code node output for logged data")


if __name__ == "__main__":
    main()