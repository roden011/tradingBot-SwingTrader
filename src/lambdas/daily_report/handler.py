"""
Daily Report Lambda Handler

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
    DailyReportGenerator,
    fetch_market_indices,
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
    """Daily report Lambda handler."""
    logger.info("Daily report generation started")

    try:
        # Get broker client
        broker = get_broker_client()

        # Fetch all data
        account = broker.get_account()
        positions = broker.get_positions()
        trades = get_today_trades()
        equity_history = get_historical_equity_values(days_back=30)
        market_indices = fetch_market_indices(broker, days_back=7)

        # Get tax data if available
        tax_data = None
        qtd_tax_data = None
        if TAX_OBLIGATIONS_TABLE:
            tax_data = get_today_tax_summary()
            qtd_tax_data = get_qtd_tax_summary()

        # Convert to ReportData format
        report_data = ReportData(
            account=AccountData(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                last_equity=float(account.last_equity) if hasattr(account, 'last_equity') else None
            ),
            positions=[
                PositionData(
                    symbol=p.symbol,
                    qty=float(p.quantity),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_pl_pct=float(p.unrealized_pl_pct) * 100
                )
                for p in positions
            ],
            trades=trades,
            equity_history=equity_history,
            market_indices=market_indices,
            tax_data=tax_data,
            qtd_tax_data=qtd_tax_data,
            bot_name="Swing Trader Bot"
        )

        # Generate report using Core
        generator = DailyReportGenerator(report_data)
        report = generator.generate()

        # Send report
        send_report(report)

        logger.info("Daily report sent successfully")
        return {'statusCode': 200, 'body': json.dumps({'message': 'Daily report sent'})}

    except Exception as e:
        logger.error(f"Error generating daily report: {e}", exc_info=True)
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


def get_today_trades():
    """Get today's trades from DynamoDB."""
    table = dynamodb.Table(TRADES_TABLE)
    today = datetime.utcnow().date().isoformat()

    try:
        response = table.scan(
            FilterExpression='begins_with(#ts, :today)',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={':today': today}
        )
        return response.get('Items', [])
    except Exception as e:
        logger.error(f"Error getting today's trades: {e}")
        return []


def get_historical_equity_values(days_back=30):
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

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#ts >= :start',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':start': start_date},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        # Group by day, take last value per day
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


def get_today_tax_summary():
    """Get today's tax summary from tax obligations table."""
    if not TAX_OBLIGATIONS_TABLE:
        return None

    try:
        table = dynamodb.Table(TAX_OBLIGATIONS_TABLE)
        today = datetime.utcnow().date().isoformat()

        response = table.scan(
            FilterExpression='begins_with(sell_date, :today)',
            ExpressionAttributeValues={':today': today}
        )

        items = response.get('Items', [])

        if not items:
            return TaxData(
                num_trades=0,
                total_gains=0.0,
                total_losses=0.0,
                net_gains=0.0,
                tax_owed=0.0,
                period_label=f"Today ({today})"
            )

        # Calculate today's summary
        total_gains = sum(
            float(t.get('realized_gain_loss', 0))
            for t in items if float(t.get('realized_gain_loss', 0)) > 0
        )
        total_losses = sum(
            abs(float(t.get('realized_gain_loss', 0)))
            for t in items if float(t.get('realized_gain_loss', 0)) < 0
        )
        net_gains = total_gains - total_losses
        tax_owed = sum(float(t.get('tax_owed', 0)) for t in items)

        return TaxData(
            num_trades=len(items),
            total_gains=total_gains,
            total_losses=total_losses,
            net_gains=net_gains,
            tax_owed=tax_owed,
            period_label=f"Today ({today})"
        )

    except Exception as e:
        logger.error(f"Error getting today's tax summary: {e}")
        return None


def get_qtd_tax_summary():
    """Get quarter-to-date tax summary from tax obligations table."""
    if not TAX_OBLIGATIONS_TABLE:
        return None

    try:
        table = dynamodb.Table(TAX_OBLIGATIONS_TABLE)
        now = datetime.utcnow()

        # Calculate quarter start date
        current_quarter = (now.month - 1) // 3 + 1
        quarter_start_month = (current_quarter - 1) * 3 + 1
        quarter_start = datetime(now.year, quarter_start_month, 1).isoformat()

        response = table.scan(
            FilterExpression='sell_date >= :start',
            ExpressionAttributeValues={':start': quarter_start}
        )

        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='sell_date >= :start',
                ExpressionAttributeValues={':start': quarter_start},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        if not items:
            return TaxData(
                num_trades=0,
                total_gains=0.0,
                total_losses=0.0,
                net_gains=0.0,
                tax_owed=0.0,
                short_term_gains=0.0,
                long_term_gains=0.0,
                period_label=f"Quarter {current_quarter} {now.year} (QTD)"
            )

        # Calculate QTD summary
        short_term = sum(
            float(t.get('realized_gain_loss', 0))
            for t in items
            if float(t.get('realized_gain_loss', 0)) > 0 and not t.get('is_long_term', False)
        )
        long_term = sum(
            float(t.get('realized_gain_loss', 0))
            for t in items
            if float(t.get('realized_gain_loss', 0)) > 0 and t.get('is_long_term', False)
        )
        total_gains = short_term + long_term
        total_losses = sum(
            abs(float(t.get('realized_gain_loss', 0)))
            for t in items if float(t.get('realized_gain_loss', 0)) < 0
        )
        net_gains = total_gains - total_losses
        tax_owed = sum(float(t.get('tax_owed', 0)) for t in items)

        return TaxData(
            num_trades=len(items),
            total_gains=total_gains,
            total_losses=total_losses,
            net_gains=net_gains,
            tax_owed=tax_owed,
            short_term_gains=short_term,
            long_term_gains=long_term,
            period_label=f"Quarter {current_quarter} {now.year} (QTD)"
        )

    except Exception as e:
        logger.error(f"Error getting QTD tax summary: {e}")
        return None


def send_report(report: str):
    """Send report via SNS."""
    sns.publish(
        TopicArn=ALERT_TOPIC_ARN,
        Subject=f"Daily Trading Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
        Message=report
    )
    logger.info("Report sent successfully")
