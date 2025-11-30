"""
Test Complete HeatGuard AI + n8n Integration
"""

from heatguard_ai_system import HeatGuardAISystem

import time
import random


def main():
    print("\n" + "=" * 70)
    print("🔗 Complete HeatGuard AI + n8n Integration Test")
    print("=" * 70)
    print("\nMake sure n8n is running with these workflows:")
    print("  1. HeatGuard Critical Alerts (webhook: heatguard-critical)")
    print("  2. HeatGuard Daily Reports (webhook: heatguard-report)")
    print("  3. HeatGuard Warnings (webhook: heatguard-warning)")
    print()
    input("Press Enter when n8n is ready...")

    # Initialize AI with n8n enabled
    print("\n" + "=" * 70)
    print("Initializing HeatGuard AI with n8n Integration")
    print("=" * 70)

    heatguard = HeatGuardAISystem(enable_n8n=True)

    # Test 1: Normal readings (should NOT send to n8n)
    print("\n" + "=" * 70)
    print("TEST 1: Normal Readings (No n8n alerts)")
    print("=" * 70)

    for i in range(3):
        normal_data = {
            'sensor_id': 'INTEGRATION_TEST',
            'temperature': random.uniform(20, 35),
            'gas_level': random.uniform(10, 20),
            'air_pressure': 1013.25,
            'humidity': random.uniform(40, 60)
        }

        print(f"\nReading {i + 1}: T={normal_data['temperature']:.1f}°C, Gas={normal_data['gas_level']:.1f}%")
        alert = heatguard.process_sensor_data(normal_data)
        print(f"  Result: {alert['severity']}")
        time.sleep(1)

    # Test 2: Warning readings (should send to n8n)
    print("\n" + "=" * 70)
    print("TEST 2: Warning Readings (Should appear in n8n)")
    print("=" * 70)

    for i in range(2):
        warning_data = {
            'sensor_id': 'INTEGRATION_TEST',
            'temperature': random.uniform(60, 70),
            'gas_level': random.uniform(30, 40),
            'air_pressure': 1013.25,
            'humidity': random.uniform(70, 80)
        }

        print(f"\nReading {i + 1}: T={warning_data['temperature']:.1f}°C, Gas={warning_data['gas_level']:.1f}%")
        alert = heatguard.process_sensor_data(warning_data)
        print(f"  Result: {alert['severity']}")
        time.sleep(2)

    # Test 3: Critical readings (should send to n8n)
    print("\n" + "=" * 70)
    print("TEST 3: Critical Readings (Should trigger n8n alerts)")
    print("=" * 70)

    for i in range(2):
        critical_data = {
            'sensor_id': 'INTEGRATION_TEST',
            'temperature': random.uniform(80, 90),
            'gas_level': random.uniform(60, 80),
            'air_pressure': 1013.25,
            'humidity': random.uniform(85, 95)
        }

        print(f"\nReading {i + 1}: T={critical_data['temperature']:.1f}°C, Gas={critical_data['gas_level']:.1f}%")
        alert = heatguard.process_sensor_data(critical_data)
        print(f"  Result: {alert['severity']}")
        if alert['alerts']:
            for a in alert['alerts']:
                print(f"    - {a['message']}")
        time.sleep(2)

    # Test 4: Generate daily report (should send to n8n)
    print("\n" + "=" * 70)
    print("TEST 4: Daily Report (Should appear in n8n)")
    print("=" * 70)

    print("\nGenerating daily report...")
    report = heatguard.generate_daily_report()

    if report:
        print(f"✓ Report generated for {report['date']}")
        print(f"  Total Readings: {report['summary']['total_readings']}")
        print(f"  Critical Alerts: {report['summary']['critical_alerts']}")
        print(f"  Warning Alerts: {report['summary']['warning_alerts']}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Integration Test Complete!")
    print("=" * 70)
    print("\nWhat to check in n8n:")
    print("1. Open n8n: http://localhost:5678")
    print("2. Click 'Executions' (left sidebar)")
    print("3. You should see:")
    print("   - 2 warning alerts")
    print("   - 2 critical alerts")
    print("   - 1 daily report")
    print("4. Click each execution to see the data")
    print("5. Check console logs in Code nodes")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()