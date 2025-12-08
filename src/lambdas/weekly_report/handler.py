"""
Weekly Report Lambda Handler

Thin handler that fetches data and delegates report generation to Core.
"""
import os
import json
import logging
import boto3
from datetime import datetime, timedelta
import sys

# Add Lambda layer path
sys.path.insert(0, '/opt/python')

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tradingbot_core import create_broker
from tradingbot_core.reporting import (
    ReportData,
    AccountData,
    PositionData,
    TaxData,
    WeeklyReportGenerator,
    fetch_market_indices_for_period,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
secrets_manager = boto3.client('secretsmanager')

TRADES_TABLE = os.environ['TRADES_TABLE']
RISK_METRICS_TABLE = os.environ['RISK_METRICS_TABLE']
TAX_OBLIGATIONS_TABLE = os.environ.get('TAX_OBLIGATIONS_TABLE', '')
ALERT_TOPIC_ARN = os.environ['ALERT_TOPIC_ARN']
ALPACA_SECRET_NAME = os.environ['ALPACA_SECRET_NAME']


def lambda_handler(event, context):
    """Weekly report Lambda handler."""
    logger.info("Weekly report generation started")

    try:
        broker = get_broker_client()
        account = broker.get_account()
        positions = broker.get_positions()
        trades = get_week_trades()
        equity_history = get_historical_equity_values(days_back=7)
        market_indices = fetch_market_indices_for_period(broker, period_days=5)
        tax_obligations = get_week_tax_obligations()

        report_data = ReportData(
            account=AccountData(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
            ),
            positions=[
                PositionData(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_pl_pct=float(p.unrealized_plpc) * 100
                )
                for p in positions
            ],
            trades=trades,
            equity_history=equity_history,
            market_indices=market_indices,
            tax_obligations=tax_obligations,
            bot_name="Swing Trader Bot"
        )

        generator = WeeklyReportGenerator(report_data)
        report = generator.generate()
        send_report(report)

        logger.info("Weekly report sent successfully")
        return {'statusCode': 200, 'body': json.dumps({'message': 'Weekly report sent'})}

    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}


def get_broker_client():
    """Get Alpaca broker client."""
    secret = secrets_manager.get_secret_value(SecretId=ALPACA_SECRET_NAME)
    secret_data = json.loads(secret['SecretString'])
    return create_broker(
        broker_type="alpaca",
        api_key=secret_data['api_key'],
        secret_key=secret_data['secret_key'],
        paper=secret_data.get('paper', True)
    )


def get_week_trades():
    """Get past week's trades from DynamoDB."""
    table = dynamodb.Table(TRADES_TABLE)
    start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()

    try:
        response = table.scan(
            FilterExpression='#ts >= :start',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={':start': start_date}
        )
        return response.get('Items', [])
    except Exception as e:
        logger.error(f"Error getting week's trades: {e}")
        return []


def get_week_tax_obligations():
    """Get past week's tax obligations for win rate calculation."""
    if not TAX_OBLIGATIONS_TABLE:
        return []

    try:
        table = dynamodb.Table(TAX_OBLIGATIONS_TABLE)
        week_start = (datetime.utcnow() - timedelta(days=7)).isoformat()

        response = table.scan(
            FilterExpression='sell_date >= :start',
            ExpressionAttributeValues={':start': week_start}
        )

        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='sell_date >= :start',
                ExpressionAttributeValues={':start': week_start},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        return items
    except Exception as e:
        logger.error(f"Error getting tax obligations: {e}")
        return []


def get_historical_equity_values(days_back=7):
    """Get historical equity values from risk metrics table."""
    try:
        table = dynamodb.Table(RISK_METRICS_TABLE)
        start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        response = table.scan(
            FilterExpression='#ts >= :start',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={':start': start_date}
        )

        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#ts >= :start',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':start': start_date},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        daily_values = {}
        for item in items:
            if 'portfolio_value' in item and item['portfolio_value'] > 0:
                timestamp = item.get('timestamp', '')
                if timestamp:
                    date_str = timestamp[:10]
                    equity = float(item['portfolio_value'])
                    if date_str not in daily_values or timestamp > daily_values[date_str][0]:
                        daily_values[date_str] = (timestamp, equity)

        equity_history = [(date_str, values[1]) for date_str, values in daily_values.items()]
        equity_history.sort(key=lambda x: x[0])
        return equity_history

    except Exception as e:
        logger.error(f"Error fetching historical equity: {e}")
        return []


def send_report(report: str):
    """Send report via SNS."""
    sns.publish(
        TopicArn=ALERT_TOPIC_ARN,
        Subject=f"Weekly Trading Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
        Message=report
    )
    logger.info("Weekly report sent successfully")
