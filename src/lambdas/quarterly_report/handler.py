"""
Quarterly Tax Report Lambda Handler

Thin handler that fetches data and delegates report generation to Core.
"""
import os
import json
import logging
import boto3
from datetime import datetime
import sys

# Add Lambda layer path
sys.path.insert(0, '/opt/python')

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tradingbot_core.reporting import (
    ReportData,
    AccountData,
    TaxData,
    QuarterlyReportGenerator,
)

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

TAX_OBLIGATIONS_TABLE = os.environ['TAX_OBLIGATIONS_TABLE']
ALERT_TOPIC_ARN = os.environ['ALERT_TOPIC_ARN']


def lambda_handler(event, context):
    """Quarterly tax report Lambda handler."""
    logger.info("Quarterly tax report generation started")

    try:
        year = event.get('year', datetime.utcnow().year)
        quarter = event.get('quarter', get_current_quarter())

        tax_obligations = get_quarterly_tax_obligations(year, quarter)
        tax_data = calculate_tax_summary(tax_obligations)

        report_data = ReportData(
            account=AccountData(equity=0, cash=0, buying_power=0),
            tax_obligations=tax_obligations,
            tax_data=tax_data,
            bot_name="Swing Trader Bot"
        )

        generator = QuarterlyReportGenerator(report_data, year=year, quarter=quarter)
        report = generator.generate()
        send_report(report, year, quarter)

        logger.info(f"Quarterly tax report sent successfully for Q{quarter} {year}")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': f'Quarterly tax report sent for Q{quarter} {year}'})
        }

    except Exception as e:
        logger.error(f"Error generating quarterly tax report: {e}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}


def get_current_quarter():
    """Get current quarter (1-4)."""
    month = datetime.utcnow().month
    return (month - 1) // 3 + 1


def get_quarterly_tax_obligations(year, quarter):
    """Get tax obligations for a specific quarter."""
    try:
        table = dynamodb.Table(TAX_OBLIGATIONS_TABLE)

        quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

        start_month, start_day = quarter_starts[quarter]
        end_month, end_day = quarter_ends[quarter]

        start_date = datetime(year, start_month, start_day).isoformat()
        end_date = datetime(year, end_month, end_day, 23, 59, 59).isoformat()

        response = table.scan(
            FilterExpression='sell_date BETWEEN :start AND :end',
            ExpressionAttributeValues={':start': start_date, ':end': end_date}
        )

        items = response.get('Items', [])
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='sell_date BETWEEN :start AND :end',
                ExpressionAttributeValues={':start': start_date, ':end': end_date},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        return items

    except Exception as e:
        logger.error(f"Error getting quarterly tax obligations: {e}")
        return []


def calculate_tax_summary(tax_obligations):
    """Calculate tax summary from obligations."""
    if not tax_obligations:
        return TaxData(
            num_trades=0, total_gains=0, total_losses=0,
            net_gains=0, tax_owed=0, short_term_gains=0, long_term_gains=0
        )

    short_term = 0.0
    long_term = 0.0
    losses = 0.0
    tax_owed = 0.0

    for item in tax_obligations:
        gain_loss = float(item.get('realized_gain_loss', 0))
        is_long_term = item.get('is_long_term', False)

        if gain_loss > 0:
            if is_long_term:
                long_term += gain_loss
            else:
                short_term += gain_loss
        else:
            losses += abs(gain_loss)

        tax_owed += float(item.get('tax_owed', 0))

    return TaxData(
        num_trades=len(tax_obligations),
        total_gains=short_term + long_term,
        total_losses=losses,
        net_gains=(short_term + long_term) - losses,
        tax_owed=tax_owed,
        short_term_gains=short_term,
        long_term_gains=long_term
    )


def send_report(report: str, year: int, quarter: int):
    """Send report via SNS."""
    sns.publish(
        TopicArn=ALERT_TOPIC_ARN,
        Subject=f"Quarterly Tax Report - Q{quarter} {year}",
        Message=report
    )
    logger.info(f"Quarterly tax report sent for Q{quarter} {year}")
