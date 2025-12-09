"""
Kill Switch Lambda Handler

Emergency shutdown - closes all positions and disables trading.
Uses the KillSwitchService from tradingbot-core for shared logic.
"""
import os
import json
import logging
import boto3
import sys

# Add Lambda layer path
sys.path.insert(0, '/opt/python')

# Add src directory to path (Lambda root is src/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from alpaca_client.client import AlpacaClient
from tradingbot_core.services import KillSwitchService, SystemStateService

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
secrets_manager = boto3.client('secretsmanager')

SYSTEM_STATE_TABLE = os.environ['SYSTEM_STATE_TABLE']
ALERT_TOPIC_ARN = os.environ['ALERT_TOPIC_ARN']
ALPACA_SECRET_NAME = os.environ['ALPACA_SECRET_NAME']


def lambda_handler(event, context):
    """
    Kill switch Lambda handler

    Args:
        event: Lambda event with optional 'reason' and 'activated_by' fields
        context: Lambda context

    Returns:
        Response with execution summary
    """
    logger.warning("KILL SWITCH ACTIVATED")

    try:
        # Get reason from event
        reason = event.get('reason', 'Manual activation')
        activated_by = event.get('activated_by', 'user')

        # Initialize services
        alpaca_client = get_alpaca_client()
        system_state_service = SystemStateService(dynamodb, SYSTEM_STATE_TABLE)
        kill_switch_service = KillSwitchService(alpaca_client, system_state_service)

        # Activate kill switch with alert callback
        result = kill_switch_service.activate(
            reason=reason,
            activated_by=activated_by,
            alert_callback=send_alert
        )

        if result.success:
            logger.warning("Kill switch execution completed successfully")
            return {
                'statusCode': 200,
                'body': json.dumps(result.to_dict())
            }
        else:
            logger.error(f"Kill switch execution had issues: {result.error}")
            return {
                'statusCode': 500,
                'body': json.dumps(result.to_dict())
            }

    except Exception as e:
        logger.error(f"Error in kill switch: {e}", exc_info=True)
        send_alert(f"KILL SWITCH ERROR: {str(e)}", "CRITICAL")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def get_alpaca_client() -> AlpacaClient:
    """Get Alpaca client from Secrets Manager"""
    try:
        secret = secrets_manager.get_secret_value(SecretId=ALPACA_SECRET_NAME)
        secret_data = json.loads(secret['SecretString'])

        return AlpacaClient(
            secret_data['api_key'],
            secret_data['secret_key'],
            secret_data.get('paper', True)
        )

    except Exception as e:
        logger.error(f"Error getting Alpaca credentials: {e}")
        raise


def send_alert(message: str, level: str = "INFO"):
    """Send SNS alert"""
    try:
        sns.publish(
            TopicArn=ALERT_TOPIC_ARN,
            Subject=f"Trading Bot Alert [{level}]",
            Message=message
        )
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
