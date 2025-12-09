"""
DayTradeRepository - DynamoDB persistence layer for day trade tracking

Handles storage and retrieval of day trade records for PDT compliance
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class DayTradeRepository:
    """
    Repository for day trade records in DynamoDB
    """

    def __init__(self, day_trades_table):
        """
        Initialize repository with DynamoDB table

        Args:
            day_trades_table: boto3 DynamoDB Table resource
        """
        self.table = day_trades_table

    def record_day_trade(
        self,
        date: str,
        symbol: str,
        side: str,
        timestamp: str,
        account_value: float
    ) -> bool:
        """
        Record a day trade to DynamoDB

        Args:
            date: Trade date in ISO format (YYYY-MM-DD)
            symbol: Stock symbol
            side: 'buy' or 'sell'
            timestamp: Full timestamp in ISO format
            account_value: Account value at time of trade

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Calculate TTL (30 days from now)
            ttl = int((datetime.utcnow() + timedelta(days=30)).timestamp())

            self.table.put_item(
                Item={
                    'date': date,
                    'symbol': symbol,
                    'side': side,
                    'timestamp': timestamp,
                    'account_value': Decimal(str(account_value)),
                    'ttl': ttl
                }
            )

            logger.info(f"Recorded day trade: {symbol} {side} on {date}")
            return True

        except Exception as e:
            logger.error(f"Failed to record day trade for {symbol}: {e}")
            return False

    def get_recent_day_trades(self, days: int = 7) -> List[Dict]:
        """
        Retrieve day trades from last N days

        Args:
            days: Number of days to look back

        Returns:
            List of day trade records
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

            response = self.table.scan(
                FilterExpression='#date >= :cutoff',
                ExpressionAttributeNames={
                    '#date': 'date'
                },
                ExpressionAttributeValues={
                    ':cutoff': cutoff_date
                }
            )

            day_trades = response.get('Items', [])

            # Convert Decimal to float for consistency
            for trade in day_trades:
                if 'account_value' in trade and isinstance(trade['account_value'], Decimal):
                    trade['account_value'] = float(trade['account_value'])

            logger.debug(f"Retrieved {len(day_trades)} day trades from last {days} days")
            return day_trades

        except Exception as e:
            logger.error(f"Failed to retrieve recent day trades: {e}")
            return []

    def get_day_trades_for_symbol(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Retrieve day trades for a specific symbol from last N days

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of day trade records for the symbol
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

            response = self.table.scan(
                FilterExpression='#date >= :cutoff AND symbol = :symbol',
                ExpressionAttributeNames={
                    '#date': 'date'
                },
                ExpressionAttributeValues={
                    ':cutoff': cutoff_date,
                    ':symbol': symbol
                }
            )

            day_trades = response.get('Items', [])

            # Convert Decimal to float
            for trade in day_trades:
                if 'account_value' in trade and isinstance(trade['account_value'], Decimal):
                    trade['account_value'] = float(trade['account_value'])

            return day_trades

        except Exception as e:
            logger.error(f"Failed to retrieve day trades for {symbol}: {e}")
            return []

    def count_day_trades_since(self, date: str) -> int:
        """
        Count total day trades since a given date

        Args:
            date: Cutoff date in ISO format (YYYY-MM-DD)

        Returns:
            int: Count of day trades
        """
        try:
            response = self.table.scan(
                FilterExpression='#date >= :cutoff',
                ExpressionAttributeNames={
                    '#date': 'date'
                },
                ExpressionAttributeValues={
                    ':cutoff': date
                },
                Select='COUNT'
            )

            return response.get('Count', 0)

        except Exception as e:
            logger.error(f"Failed to count day trades since {date}: {e}")
            return 0
