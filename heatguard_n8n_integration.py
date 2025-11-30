"""
n8n Connector for HeatGuard AI
Sends alerts and reports to n8n workflows
"""

import requests
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class N8NConnector:
    """
    Connects HeatGuard AI to n8n workflows
    """

    def __init__(self, n8n_base_url="http://localhost:5678"):
        """
        Initialize n8n connector

        Args:
            n8n_base_url: Base URL of your n8n instance
        """
        self.base_url = n8n_base_url

        # Webhook endpoints
        self.webhooks = {
            'critical': f"{n8n_base_url}/webhook/heatguard-critical",
            'warning': f"{n8n_base_url}/webhook/heatguard-warning",
            'normal': f"{n8n_base_url}/webhook/heatguard-normal",
            'daily_report': f"{n8n_base_url}/webhook/heatguard-report",
            'maintenance': f"{n8n_base_url}/webhook/heatguard-maintenance"
        }

        logger.info(f"N8N Connector initialized: {n8n_base_url}")

    def send_alert(self, alert_data, alert_type='critical'):
        """
        Send alert to n8n workflow

        Args:
            alert_data: Dictionary with alert information
            alert_type: 'critical', 'warning', or 'normal'

        Returns:
            True if successful, False otherwise
        """
        webhook_url = self.webhooks.get(alert_type, self.webhooks['critical'])

        try:
            # Add metadata
            payload = {
                **alert_data,
                'n8n_timestamp': datetime.now().isoformat(),
                'alert_type': alert_type
            }

            # Send to n8n
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"✓ {alert_type.upper()} alert sent to n8n")
                print(f"✓ Alert sent to n8n: {alert_type}")
                return True
            else:
                logger.error(f"n8n returned status {response.status_code}")
                print(f"✗ n8n error: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to n8n")
            print("✗ n8n timeout - check if n8n is running")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to n8n")
            print("✗ Cannot connect to n8n - is it running?")
            return False
        except Exception as e:
            logger.error(f"Error sending to n8n: {e}")
            print(f"✗ Error: {e}")
            return False

    def send_daily_report(self, report_data):
        """
        Send daily report to n8n

        Args:
            report_data: Dictionary with report information

        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                **report_data,
                'report_type': 'daily',
                'sent_at': datetime.now().isoformat()
            }

            response = requests.post(
                self.webhooks['daily_report'],
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                logger.info("✓ Daily report sent to n8n")
                print("✓ Daily report sent to n8n")
                return True
            else:
                logger.error(f"n8n returned status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending report to n8n: {e}")
            print(f"✗ Error sending report: {e}")
            return False

    def send_maintenance_alert(self, maintenance_data):
        """
        Send maintenance prediction to n8n

        Args:
            maintenance_data: Dictionary with maintenance info

        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                **maintenance_data,
                'alert_type': 'maintenance',
                'sent_at': datetime.now().isoformat()
            }

            response = requests.post(
                self.webhooks['maintenance'],
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                logger.info("✓ Maintenance alert sent to n8n")
                print("✓ Maintenance alert sent to n8n")
                return True
            else:
                logger.error(f"n8n returned status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending maintenance alert: {e}")
            print(f"✗ Error: {e}")
            return False

    def test_connection(self):
        """
        Test connection to n8n

        Returns:
            True if n8n is reachable, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}",
                timeout=5
            )
            return response.status_code in [200, 302]
        except:
            return False


# Test the connector
if __name__ == "__main__":
    print("=" * 70)
    print("Testing N8N Connector")
    print("=" * 70)

    # Initialize
    connector = N8NConnector()

    # Test connection
    print("\n1. Testing connection to n8n...")
    if connector.test_connection():
        print("   ✓ n8n is reachable")
    else:
        print("   ✗ Cannot reach n8n - make sure it's running!")
        print("   Run: npx n8n")
        exit()

    # Test critical alert
    print("\n2. Sending test critical alert...")
    test_alert = {
        'severity': 'critical',
        'temperature': 85.0,
        'gas_level': 70.0,
        'sensor_id': 'TEST_001',
        'message': 'Test critical alert from connector'
    }
    connector.send_alert(test_alert, 'critical')

    # Test daily report
    print("\n3. Sending test daily report...")
    test_report = {
        'date': datetime.now().date().isoformat(),
        'summary': {
            'total_readings': 100,
            'critical_alerts': 2,
            'warning_alerts': 5
        }
    }
    connector.send_daily_report(test_report)

    print("\n" + "=" * 70)
    print("✓ Test complete! Check n8n Executions to see the data")
    print("=" * 70)