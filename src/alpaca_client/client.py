"""
Alpaca API Client Wrapper
"""
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus as AlpacaOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Import order result model
# Note: No path manipulation needed in Lambda
from tradingbot_core import OrderResult, OrderStatus, RejectionReason

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Wrapper around Alpaca API for trading and market data
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca client

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )

        # Initialize data client
        self.data_client = StockHistoricalDataClient(
            api_key=api_key, secret_key=secret_key
        )

        logger.info(
            f"Alpaca client initialized (paper={paper})"
        )

    def get_account(self):
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            logger.info(f"Account status: {account.status}, equity: ${account.equity}")
            return account
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            raise

    def get_positions(self) -> List:
        """Get all current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            logger.info(f"Retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def get_position(self, symbol: str):
        """Get position for a specific symbol"""
        try:
            position = self.trading_client.get_open_position(symbol)
            return position
        except Exception as e:
            logger.debug(f"No position for {symbol}: {e}")
            return None

    def _fetch_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
        batch_num: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch a batch of symbols (internal method for parallel execution)

        Args:
            symbols: List of symbols in this batch
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            batch_num: Batch number (for logging)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                feed='iex',  # Use IEX feed for paper trading (free tier)
            )

            bars = self.data_client.get_stock_bars(request_params)

            # Convert to dictionary of DataFrames
            result = {}
            for symbol in symbols:
                if symbol in bars.data:
                    df = pd.DataFrame([
                        {
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                        }
                        for bar in bars.data[symbol]
                    ])
                    df.set_index('timestamp', inplace=True)
                    result[symbol] = df
                else:
                    logger.debug(f"No data received for {symbol} in batch {batch_num}")
                    result[symbol] = pd.DataFrame()

            logger.debug(f"Batch {batch_num}: Retrieved {len(result)} symbols")
            return result

        except Exception as e:
            logger.error(f"Error fetching batch {batch_num}: {e}")
            # Return empty DataFrames for all symbols in this batch
            return {symbol: pd.DataFrame() for symbol in symbols}

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        enable_parallel: bool = True,
        max_workers: int = 5,
        batch_size: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical price bars for symbols with parallel fetching

        Args:
            symbols: List of symbols
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (default: Day)
            enable_parallel: Enable parallel batch fetching (default: True)
            max_workers: Number of parallel workers (default: 5)
            batch_size: Symbols per batch (default: 100)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        if not symbols:
            return {}

        # If parallel is disabled or only one batch, use sequential fetch
        if not enable_parallel or len(symbols) <= batch_size:
            return self._fetch_batch(symbols, start, end, timeframe, 1)

        # Split symbols into batches
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

        logger.info(
            f"Fetching historical data for {len(symbols)} symbols in {len(batches)} batches "
            f"(parallel workers={max_workers})"
        )

        result = {}

        # Fetch batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch fetch tasks
            future_to_batch = {
                executor.submit(self._fetch_batch, batch, start, end, timeframe, i + 1): i + 1
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_result = future.result()
                    result.update(batch_result)
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")

        logger.info(f"Retrieved historical data for {len(result)} symbols (parallel fetch)")
        return result

    def get_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe=None,
        enable_parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Adapter method to match Core's BrokerClient interface.

        Wraps get_historical_bars to provide the interface expected by
        tradingbot_core's MarketDataService.

        Args:
            symbols: List of symbols
            start: Start datetime
            end: End datetime
            timeframe: Core's Timeframe enum (ignored, always uses daily bars)
            enable_parallel: Enable parallel fetching

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        return self.get_historical_bars(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=TimeFrame.Day,
            enable_parallel=enable_parallel
        )

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for symbols

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to latest price
        """
        try:
            # Get the last bar for each symbol
            end = datetime.utcnow()
            start = end - timedelta(days=1)

            bars = self.get_historical_bars(symbols, start, end, TimeFrame.Minute)

            prices = {}
            for symbol, df in bars.items():
                if not df.empty:
                    prices[symbol] = float(df['close'].iloc[-1])
                else:
                    logger.warning(f"No price data for {symbol}")
                    prices[symbol] = 0.0

            return prices

        except Exception as e:
            logger.error(f"Error getting latest prices: {e}")
            raise

    def place_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "day"
    ) -> Optional[str]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            time_in_force: Time in force (default 'day')

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

            market_order_data = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif
            )

            order = self.trading_client.submit_order(order_data=market_order_data)

            logger.info(
                f"Market order placed: {side} {qty} {symbol}, order_id={order.id}"
            )
            return order.id

        except Exception as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Optional[str]:
        """
        Place a limit order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            limit_price: Limit price
            time_in_force: Time in force (default 'day')

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )

            order = self.trading_client.submit_order(order_data=limit_order_data)

            logger.info(
                f"Limit order placed: {side} {qty} {symbol} @ ${limit_price}, order_id={order.id}"
            )
            return order.id

        except Exception as e:
            logger.error(f"Error placing limit order for {symbol}: {e}")
            return None

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.trading_client.cancel_orders()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions (emergency exit)"""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.warning("ALL POSITIONS CLOSED (EMERGENCY EXIT)")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def close_position(self, symbol: str, qty: Optional[float] = None) -> bool:
        """
        Close a position

        Args:
            symbol: Stock symbol
            qty: Quantity to close (None = close all)

        Returns:
            True if successful
        """
        try:
            if qty:
                self.trading_client.close_position(symbol, qty=qty)
            else:
                self.trading_client.close_position(symbol)

            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    def get_order_status(self, order_id: str):
        """Get order status"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    def _map_rejection_reason(self, error_message: str) -> RejectionReason:
        """Map Alpaca error messages to rejection reasons"""
        error_lower = error_message.lower()

        if "insufficient" in error_lower or "buying power" in error_lower:
            return RejectionReason.INSUFFICIENT_FUNDS
        elif "not tradable" in error_lower or "not tradeable" in error_lower:
            return RejectionReason.STOCK_NOT_TRADEABLE
        elif "halt" in error_lower:
            return RejectionReason.STOCK_HALTED
        elif "quantity" in error_lower or "notional" in error_lower:
            return RejectionReason.QUANTITY_TOO_SMALL
        elif "duplicate" in error_lower:
            return RejectionReason.DUPLICATE_ORDER
        elif "rate" in error_lower or "429" in error_message:
            return RejectionReason.RATE_LIMIT
        else:
            return RejectionReason.UNKNOWN

    def wait_for_order_fill(
        self,
        order_id: str,
        max_wait_seconds: int = 30,
        poll_interval_seconds: float = 0.5
    ) -> Optional[OrderResult]:
        """
        Poll order status until filled or timeout

        Args:
            order_id: Order ID to monitor
            max_wait_seconds: Maximum time to wait for fill
            poll_interval_seconds: Time between status checks

        Returns:
            OrderResult with fill details or None if timeout
        """
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < max_wait_seconds:
            attempt += 1

            try:
                order = self.trading_client.get_order_by_id(order_id)

                # Check order status
                if order.status == AlpacaOrderStatus.FILLED:
                    logger.info(
                        f"Order {order_id} filled: {order.filled_qty} @ ${order.filled_avg_price}"
                    )
                    return OrderResult(
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side.value,
                        requested_quantity=float(order.qty),
                        expected_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        status=OrderStatus.FILLED,
                        filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
                        filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                        filled_at=order.filled_at.isoformat() if order.filled_at else None
                    )

                elif order.status == AlpacaOrderStatus.PARTIALLY_FILLED:
                    # Still partially filled, keep waiting
                    logger.debug(
                        f"Order {order_id} partially filled: {order.filled_qty}/{order.qty}, "
                        f"waiting... (attempt {attempt})"
                    )

                elif order.status in [
                    AlpacaOrderStatus.REJECTED,
                    AlpacaOrderStatus.CANCELED,
                    AlpacaOrderStatus.EXPIRED
                ]:
                    # Order failed
                    logger.warning(f"Order {order_id} failed with status: {order.status}")

                    # Handle partial fill before rejection
                    if order.filled_qty and float(order.filled_qty) > 0:
                        return OrderResult(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side.value,
                            requested_quantity=float(order.qty),
                            expected_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                            status=OrderStatus.PARTIALLY_FILLED,
                            filled_quantity=float(order.filled_qty),
                            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                            rejection_reason=self._map_rejection_reason(str(order.status)),
                            error_message=f"Order {order.status.value}: partially filled before failure",
                            submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                            filled_at=order.filled_at.isoformat() if order.filled_at else None
                        )
                    else:
                        return OrderResult(
                            order_id=order.id,
                            symbol=order.symbol,
                            side=order.side.value,
                            requested_quantity=float(order.qty),
                            expected_price=0.0,
                            status=OrderStatus.REJECTED,
                            rejection_reason=self._map_rejection_reason(str(order.status)),
                            error_message=f"Order {order.status.value}",
                            submitted_at=order.submitted_at.isoformat() if order.submitted_at else None
                        )

            except Exception as e:
                logger.error(f"Error polling order {order_id}: {e}")
                time.sleep(poll_interval_seconds)
                continue

            # Wait before next poll
            time.sleep(poll_interval_seconds)

        # Timeout - check final status
        logger.warning(f"Order {order_id} fill timeout after {max_wait_seconds}s")

        try:
            order = self.trading_client.get_order_by_id(order_id)

            # Return partial fill result if any quantity was filled
            if order.filled_qty and float(order.filled_qty) > 0:
                logger.warning(
                    f"Order {order_id} partially filled at timeout: "
                    f"{order.filled_qty}/{order.qty}"
                )
                return OrderResult(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side.value,
                    requested_quantity=float(order.qty),
                    expected_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    status=OrderStatus.PARTIALLY_FILLED,
                    filled_quantity=float(order.filled_qty),
                    filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    error_message=f"Timeout after {max_wait_seconds}s",
                    submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                    filled_at=order.filled_at.isoformat() if order.filled_at else None
                )
        except Exception as e:
            logger.error(f"Error getting final order status for {order_id}: {e}")

        return None

    def place_market_order_with_confirmation(
        self,
        symbol: str,
        qty: float,
        side: str,
        expected_price: float,
        time_in_force: str = "day",
        wait_for_fill: bool = True,
        max_wait_seconds: int = 30
    ) -> OrderResult:
        """
        Place market order and wait for fill confirmation

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            expected_price: Expected execution price (for slippage calculation)
            time_in_force: Time in force (default 'day')
            wait_for_fill: Whether to wait for fill confirmation
            max_wait_seconds: Maximum time to wait for fill

        Returns:
            OrderResult with complete fill details
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

            market_order_data = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif
            )

            order = self.trading_client.submit_order(order_data=market_order_data)

            logger.info(
                f"Market order submitted: {side} {qty} {symbol}, order_id={order.id}"
            )

            # Wait for fill if requested
            if wait_for_fill:
                result = self.wait_for_order_fill(order.id, max_wait_seconds)

                if result:
                    # Update expected price for slippage calculation
                    result.expected_price = expected_price
                    result.__post_init__()  # Recalculate slippage

                    # Log slippage if significant
                    if abs(result.slippage_percentage) > 0.1:
                        logger.warning(
                            f"Significant slippage on {symbol}: "
                            f"expected ${expected_price:.2f}, "
                            f"filled @ ${result.filled_avg_price:.2f} "
                            f"({result.slippage_percentage:+.2f}%)"
                        )

                    return result
                else:
                    # Timeout without result
                    return OrderResult(
                        order_id=order.id,
                        symbol=symbol,
                        side=side,
                        requested_quantity=qty,
                        expected_price=expected_price,
                        status=OrderStatus.FAILED,
                        error_message="Timeout waiting for fill",
                        submitted_at=order.submitted_at.isoformat() if order.submitted_at else None
                    )
            else:
                # Return immediate result without waiting
                return OrderResult(
                    order_id=order.id,
                    symbol=symbol,
                    side=side,
                    requested_quantity=qty,
                    expected_price=expected_price,
                    status=OrderStatus.PENDING,
                    submitted_at=order.submitted_at.isoformat() if order.submitted_at else None
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error placing market order for {symbol}: {error_msg}")

            rejection_reason = self._map_rejection_reason(error_msg)

            # Determine if error is retryable
            is_retryable = rejection_reason in [
                RejectionReason.RATE_LIMIT,
                RejectionReason.UNKNOWN
            ]

            return OrderResult(
                order_id=None,
                symbol=symbol,
                side=side,
                requested_quantity=qty,
                expected_price=expected_price,
                status=OrderStatus.REJECTED,
                rejection_reason=rejection_reason,
                error_message=error_msg,
                is_retryable=is_retryable
            )

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
