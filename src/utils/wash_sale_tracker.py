"""
Wash Sale Rule Tracker

Tracks wash sale violations for tax compliance (live trading only).

Wash Sale Rule (IRS):
- If you sell a security at a LOSS
- And buy the same security within 30 days (before or after)
- You cannot claim the loss for tax purposes

This tracker prevents repurchasing securities within 30 days of selling at a loss.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WashSaleCheck:
    """Result of wash sale check"""
    can_buy: bool
    symbol: str
    blocked_reason: Optional[str] = None
    warning_message: Optional[str] = None
    loss_trades: List[Dict] = None
    days_until_clear: Optional[int] = None

    def __post_init__(self):
        if self.loss_trades is None:
            self.loss_trades = []

    def __repr__(self):
        if self.can_buy:
            return f"WashSaleCheck({self.symbol}, OK)"
        else:
            return f"WashSaleCheck({self.symbol}, BLOCKED, {self.days_until_clear} days)"


class WashSaleTracker:
    """
    Track wash sales to prevent tax-inefficient trades

    Note: Only relevant for live trading with real money.
    Paper trading doesn't have tax implications.
    """

    def __init__(self, enabled: bool = True, wash_sale_period_days: int = 30):
        """
        Initialize wash sale tracker

        Args:
            enabled: Whether to enforce wash sale rules (disable for paper trading)
            wash_sale_period_days: Days to wait after loss before repurchasing (30 default per IRS)
        """
        self.enabled = enabled
        self.wash_sale_period_days = wash_sale_period_days

    def check_wash_sale(
        self,
        symbol: str,
        side: str,
        closed_trades: List[Dict]
    ) -> WashSaleCheck:
        """
        Check if buying this symbol would trigger a wash sale

        Args:
            symbol: Symbol to buy
            side: Must be 'buy' (only checks buys)
            closed_trades: List of closed trades with P&L

        Returns:
            WashSaleCheck with compliance status
        """
        if not self.enabled:
            return WashSaleCheck(
                can_buy=True,
                symbol=symbol,
                warning_message="Wash sale tracking disabled"
            )

        if side != 'buy':
            # Wash sale only applies to buying after selling at loss
            return WashSaleCheck(can_buy=True, symbol=symbol)

        # Find recent closed trades for this symbol with losses
        cutoff_date = datetime.utcnow() - timedelta(days=self.wash_sale_period_days)

        loss_trades = []
        for trade in closed_trades:
            if trade.get('symbol') != symbol:
                continue

            if trade.get('side') != 'sell':
                continue

            # Check if it's a loss
            realized_pl = trade.get('realized_pl', 0)
            if realized_pl >= 0:
                continue  # Not a loss

            # Check if within wash sale period
            timestamp = trade.get('timestamp', '')
            if not timestamp:
                continue

            try:
                trade_date = datetime.fromisoformat(timestamp)
            except:
                continue

            if trade_date < cutoff_date:
                continue  # Too old

            # This is a recent loss trade
            days_ago = (datetime.utcnow() - trade_date).days
            days_remaining = self.wash_sale_period_days - days_ago

            loss_trades.append({
                'timestamp': timestamp,
                'realized_pl': realized_pl,
                'days_ago': days_ago,
                'days_remaining': days_remaining
            })

        if loss_trades:
            # Found recent loss trades - would be wash sale
            most_recent_loss = loss_trades[0]  # Assumes sorted by date
            days_remaining = most_recent_loss.get('days_remaining', 0)
            total_losses = sum(t.get('realized_pl', 0) for t in loss_trades)

            logger.warning(
                f"⚠ WASH SALE: Cannot buy {symbol} - sold at loss ${total_losses:.2f} "
                f"within last {self.wash_sale_period_days} days. "
                f"Wait {days_remaining} more days to avoid wash sale rule."
            )

            return WashSaleCheck(
                can_buy=False,
                symbol=symbol,
                blocked_reason=(
                    f"Wash sale rule: Sold {symbol} at loss ${total_losses:.2f} "
                    f"{most_recent_loss.get('days_ago')} days ago. "
                    f"Must wait {days_remaining} more days before repurchasing."
                ),
                loss_trades=loss_trades,
                days_until_clear=days_remaining
            )
        else:
            # No recent loss trades - safe to buy
            return WashSaleCheck(can_buy=True, symbol=symbol)

    def get_blocked_symbols(self, closed_trades: List[Dict]) -> List[str]:
        """
        Get list of symbols currently blocked by wash sale rule

        Args:
            closed_trades: List of closed trades

        Returns:
            List of symbols that cannot be purchased due to wash sale
        """
        if not self.enabled:
            return []

        # Get unique symbols from recent loss trades
        cutoff_date = datetime.utcnow() - timedelta(days=self.wash_sale_period_days)

        blocked_symbols = set()

        for trade in closed_trades:
            if trade.get('side') != 'sell':
                continue

            realized_pl = trade.get('realized_pl', 0)
            if realized_pl >= 0:
                continue  # Not a loss

            timestamp = trade.get('timestamp', '')
            if not timestamp:
                continue

            try:
                trade_date = datetime.fromisoformat(timestamp)
            except:
                continue

            if trade_date < cutoff_date:
                continue

            symbol = trade.get('symbol')
            if symbol:
                blocked_symbols.add(symbol)

        if blocked_symbols:
            logger.info(
                f"Wash sale: {len(blocked_symbols)} symbols blocked: {blocked_symbols}"
            )

        return list(blocked_symbols)
