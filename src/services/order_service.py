"""
Order Service

Handles order creation, submission, and tracking.
Delegates to AlpacaClient for actual order execution.
Integrates with TaxService to calculate tax obligations on sells.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from alpaca_client.client import AlpacaClient
from tradingbot_core import OrderResult, OrderStatus
from tradingbot_core import Trade
from tradingbot_core.models.utils import convert_floats_to_decimal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderService:
    """
    Service for handling order operations including creation,
    submission, validation, trade recording, and tax calculation.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        dynamodb_client=None,
        trades_table_name: str = None,
        positions_table_name: str = None,
        tax_service=None
    ):
        """
        Initialize order service

        Args:
            alpaca_client: AlpacaClient instance
            dynamodb_client: DynamoDB client for recording trades
            trades_table_name: Name of the trades DynamoDB table
            positions_table_name: Name of the positions DynamoDB table (for cost basis)
            tax_service: TaxService instance (optional, for tax tracking)
        """
        self.alpaca_client = alpaca_client
        self.dynamodb = dynamodb_client
        self.trades_table_name = trades_table_name
        self.positions_table_name = positions_table_name
        self.tax_service = tax_service
        self.trades_table = None
        self.positions_table = None

        if dynamodb_client and trades_table_name:
            self.trades_table = dynamodb_client.Table(trades_table_name)

        if dynamodb_client and positions_table_name:
            self.positions_table = dynamodb_client.Table(positions_table_name)

        logger.info(
            f"OrderService initialized with trades table: {trades_table_name}, "
            f"positions table: {positions_table_name}, "
            f"tax tracking: {'enabled' if tax_service else 'disabled'}"
        )

    def create_market_buy_order(
        self,
        symbol: str,
        quantity: float,
        expected_price: float,
        strategy_signals: Dict = None,
        consensus_score: float = 0.0
    ) -> Dict:
        """
        Create and submit a market buy order

        Args:
            symbol: Symbol to buy
            quantity: Quantity to buy
            expected_price: Expected price (for slippage check)
            strategy_signals: Strategy signals that triggered this order
            consensus_score: Consensus score for this trade

        Returns:
            Dict with execution result
        """
        logger.info(
            f"Creating market BUY order: {quantity} {symbol} @ ~${expected_price:.2f} "
            f"(cost: ${quantity * expected_price:,.2f}, consensus: {consensus_score:.2f})"
        )

        # Submit order to Alpaca
        order_result = self.alpaca_client.place_market_order_with_confirmation(
            symbol=symbol,
            qty=quantity,
            side='buy',
            expected_price=expected_price,
            wait_for_fill=True,
            max_wait_seconds=30
        )

        # Record trade
        trade = Trade.create_new(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            order_type='market',
            strategy_signals=strategy_signals or {},
            consensus_score=consensus_score,
            risk_checks_passed=True,
            risk_check_details={'order_type': 'market_buy'}
        )

        if order_result.status == OrderStatus.FILLED:
            trade.mark_filled(
                filled_price=order_result.filled_avg_price,
                filled_quantity=order_result.filled_quantity,
                broker_order_id=order_result.order_id,
                slippage_dollars=order_result.slippage_dollars,
                slippage_percentage=order_result.slippage_percentage
            )
            logger.info(
                f"✓ BUY executed: {order_result.filled_quantity} {symbol} @ "
                f"${order_result.filled_avg_price:.2f}"
            )
        else:
            trade.mark_rejected(
                f"{order_result.rejection_reason}: {order_result.error_message}"
            )
            logger.warning(
                f"✗ BUY rejected: {symbol} - {order_result.rejection_reason}"
            )

        # Save trade to DynamoDB
        self._save_trade(trade)

        return {
            'executed': order_result.status == OrderStatus.FILLED,
            'order_result': order_result,
            'trade': trade,
            'quantity': order_result.filled_quantity if order_result.status == OrderStatus.FILLED else 0,
            'filled_price': order_result.filled_avg_price if order_result.status == OrderStatus.FILLED else 0.0
        }

    def create_market_sell_order(
        self,
        symbol: str,
        quantity: float,
        expected_price: float,
        strategy_signals: Dict = None,
        consensus_score: float = 0.0,
        sell_reason: str = "strategy_signal"
    ) -> Dict:
        """
        Create and submit a market sell order

        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            expected_price: Expected price (for slippage check)
            strategy_signals: Strategy signals that triggered this order
            consensus_score: Consensus score for this trade
            sell_reason: Reason for selling (e.g., "stop_loss", "take_profit", "eod_exit")

        Returns:
            Dict with execution result
        """
        logger.info(
            f"Creating market SELL order: {quantity} {symbol} @ ~${expected_price:.2f} "
            f"(proceeds: ${quantity * expected_price:,.2f}, reason: {sell_reason})"
        )

        # Submit order to Alpaca
        order_result = self.alpaca_client.place_market_order_with_confirmation(
            symbol=symbol,
            qty=quantity,
            side='sell',
            expected_price=expected_price,
            wait_for_fill=True,
            max_wait_seconds=30
        )

        # Record trade
        trade = Trade.create_new(
            symbol=symbol,
            side='sell',
            quantity=quantity,
            order_type='market',
            strategy_signals=strategy_signals or {sell_reason: 'triggered'},
            consensus_score=consensus_score,
            risk_checks_passed=True,
            risk_check_details={'order_type': 'market_sell', 'reason': sell_reason}
        )

        if order_result.status == OrderStatus.FILLED:
            trade.mark_filled(
                filled_price=order_result.filled_avg_price,
                filled_quantity=order_result.filled_quantity,
                broker_order_id=order_result.order_id,
                slippage_dollars=order_result.slippage_dollars,
                slippage_percentage=order_result.slippage_percentage
            )
            logger.info(
                f"✓ SELL executed: {order_result.filled_quantity} {symbol} @ "
                f"${order_result.filled_avg_price:.2f}"
            )

            # Calculate and save tax obligation
            if self.tax_service:
                self._calculate_and_save_tax(
                    symbol=symbol,
                    sell_quantity=order_result.filled_quantity,
                    sell_price=order_result.filled_avg_price,
                    trade_id=trade.trade_id
                )
        else:
            trade.mark_rejected(
                f"{order_result.rejection_reason}: {order_result.error_message}"
            )
            logger.warning(
                f"✗ SELL rejected: {symbol} - {order_result.rejection_reason}"
            )

        # Save trade to DynamoDB
        self._save_trade(trade)

        return {
            'executed': order_result.status == OrderStatus.FILLED,
            'order_result': order_result,
            'trade': trade,
            'quantity': order_result.filled_quantity if order_result.status == OrderStatus.FILLED else 0,
            'filled_price': order_result.filled_avg_price if order_result.status == OrderStatus.FILLED else 0.0
        }

    def create_rejected_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        rejection_reason: str,
        strategy_signals: Dict = None,
        consensus_score: float = 0.0
    ) -> Trade:
        """
        Create a rejected trade record (for trades blocked by risk checks, etc.)

        Args:
            symbol: Symbol
            side: 'buy' or 'sell'
            quantity: Intended quantity
            rejection_reason: Why the trade was rejected
            strategy_signals: Strategy signals
            consensus_score: Consensus score

        Returns:
            Trade object (already saved to DynamoDB)
        """
        trade = Trade.create_new(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type='market',
            strategy_signals=strategy_signals or {},
            consensus_score=consensus_score,
            risk_checks_passed=False,
            risk_check_details={'rejection_reason': rejection_reason}
        )

        trade.mark_rejected(rejection_reason)
        self._save_trade(trade)

        logger.info(f"✗ Trade rejected: {side.upper()} {quantity} {symbol} - {rejection_reason}")

        return trade

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """
        Cancel all open orders (optionally for a specific symbol)

        Args:
            symbol: Symbol to cancel orders for (None = all symbols)

        Returns:
            True if successful
        """
        try:
            if symbol:
                logger.info(f"Cancelling all orders for {symbol}")
            else:
                logger.warning("Cancelling ALL open orders")

            # Alpaca API to cancel orders
            self.alpaca_client.trading_client.cancel_orders()
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List:
        """
        Get all open orders

        Args:
            symbol: Symbol to filter by (None = all symbols)

        Returns:
            List of open orders
        """
        try:
            # Get open orders from Alpaca
            orders = self.alpaca_client.trading_client.get_orders(status='open')

            if symbol:
                orders = [o for o in orders if o.symbol == symbol]

            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def close_position(
        self,
        symbol: str,
        current_price: float,
        reason: str = "manual_close"
    ) -> Dict:
        """
        Close an entire position (sell all shares)

        Args:
            symbol: Symbol to close
            current_price: Current price
            reason: Reason for closing

        Returns:
            Dict with execution result
        """
        try:
            # Get position from Alpaca
            position = self.alpaca_client.get_position(symbol)

            if not position:
                logger.warning(f"No position found for {symbol}")
                return {'executed': False, 'reason': 'No position found'}

            quantity = float(position.qty)

            logger.info(f"Closing entire position: {quantity} {symbol} @ ${current_price:.2f}")

            return self.create_market_sell_order(
                symbol=symbol,
                quantity=quantity,
                expected_price=current_price,
                strategy_signals={reason: 'triggered'},
                consensus_score=-1.0,  # Forced close
                sell_reason=reason
            )

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return {'executed': False, 'reason': str(e)}

    def _save_trade(self, trade: Trade) -> bool:
        """
        Save trade to DynamoDB

        Args:
            trade: Trade object to save

        Returns:
            True if successful
        """
        if not self.trades_table:
            logger.warning("Trades table not configured, skipping trade save")
            return False

        try:
            trade_dict = trade.to_dynamodb_item()
            # Note: to_dynamodb_item() already calls convert_floats_to_decimal()

            self.trades_table.put_item(Item=trade_dict)
            logger.debug(f"Saved trade: {trade.side.upper()} {trade.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error saving trade {trade.symbol}: {e}")
            return False

    def save_trade(self, trade: Trade) -> bool:
        """
        Public method to save trade to DynamoDB

        Args:
            trade: Trade object to save

        Returns:
            True if successful
        """
        return self._save_trade(trade)

    def get_cost(self, quantity: float, price: float) -> float:
        """
        Calculate total cost for a buy order

        Args:
            quantity: Quantity to buy
            price: Price per share

        Returns:
            Total cost
        """
        return quantity * price

    def get_proceeds(self, quantity: float, price: float) -> float:
        """
        Calculate expected proceeds for a sell order

        Args:
            quantity: Quantity to sell
            price: Price per share

        Returns:
            Total proceeds
        """
        return quantity * price

    def validate_order_parameters(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Validate order parameters before submission

        Args:
            symbol: Symbol
            quantity: Quantity
            price: Price

        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"

        if price <= 0:
            return False, "Price must be positive"

        if not symbol or len(symbol) == 0:
            return False, "Symbol must be provided"

        # Check for fractional shares (Alpaca supports some, but not all)
        if quantity != int(quantity) and quantity < 1.0:
            return False, "Fractional shares must be at least 1.0 total"

        return True, None

    def _calculate_and_save_tax(
        self,
        symbol: str,
        sell_quantity: float,
        sell_price: float,
        trade_id: str
    ) -> bool:
        """
        Calculate and save tax obligation for a sell trade

        Args:
            symbol: Symbol sold
            sell_quantity: Quantity sold
            sell_price: Sell price per share
            trade_id: Trade ID

        Returns:
            True if successful
        """
        try:
            # Get position from DynamoDB to get cost basis and entry date
            if not self.positions_table:
                logger.warning("Positions table not configured, skipping tax calculation")
                return False

            response = self.positions_table.get_item(Key={'symbol': symbol})
            position_item = response.get('Item')

            if not position_item:
                logger.warning(f"No position found for {symbol}, skipping tax calculation")
                return False

            # Extract position data
            avg_entry_price = float(position_item.get('avg_entry_price', 0))
            entry_timestamp = position_item.get('entry_timestamp', '')

            # Calculate cost basis for the sold quantity
            # Use avg_entry_price directly - the stored cost_basis can become stale
            # when positions are synced from Alpaca (quantity updates but cost_basis doesn't)
            sold_cost_basis = sell_quantity * avg_entry_price

            # Calculate tax obligation
            tax_obligation = self.tax_service.calculate_tax_on_sale(
                symbol=symbol,
                sell_quantity=sell_quantity,
                sell_price=sell_price,
                sell_date=datetime.utcnow().isoformat(),
                trade_id=trade_id,
                cost_basis=sold_cost_basis,
                buy_date=entry_timestamp
            )

            if tax_obligation:
                # Save to DynamoDB
                return self.tax_service.save_tax_obligation(tax_obligation)

            return False

        except Exception as e:
            logger.error(f"Error calculating tax for {symbol}: {e}")
            return False

    def _get_buy_dates_for_position(self, symbol: str) -> List[str]:
        """
        Get buy dates for a position (for FIFO cost basis calculation)

        Args:
            symbol: Symbol

        Returns:
            List of buy dates (ISO format)
        """
        if not self.trades_table:
            return []

        try:
            # Query trades table for buy trades for this symbol
            response = self.trades_table.query(
                IndexName='symbol-timestamp-index',
                KeyConditionExpression='symbol = :symbol',
                FilterExpression='side = :side AND #status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':symbol': symbol,
                    ':side': 'buy',
                    ':status': 'filled'
                }
            )

            items = response.get('Items', [])
            buy_dates = [item.get('timestamp', '') for item in items]
            buy_dates.sort()  # FIFO order

            return buy_dates

        except Exception as e:
            logger.error(f"Error getting buy dates for {symbol}: {e}")
            return []
