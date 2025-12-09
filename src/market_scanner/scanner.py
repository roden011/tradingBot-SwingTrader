"""
Market Scanner for Dynamic Stock Discovery
Scans the market for liquid, tradeable stocks that meet criteria
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
import time

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Scans the market to discover tradeable stocks dynamically
    """

    def __init__(self, alpaca_client, config: Optional[Dict] = None, dynamodb_table=None):
        """
        Initialize market scanner

        Args:
            alpaca_client: AlpacaClient instance
            config: Market scanner configuration dict
            dynamodb_table: Optional DynamoDB Table resource for caching
        """
        self.alpaca_client = alpaca_client
        self.dynamodb_table = dynamodb_table

        # Extract config
        config = config or {}
        self.batch_size = config.get('batch_size', 100)
        self.min_data_points = config.get('min_data_points', 3)

        # Cache configuration
        self.cache_enabled = config.get('cache_enabled', True) and dynamodb_table is not None
        self.cache_ttl_seconds = config.get('cache_ttl_seconds', 3600)  # Default: 1 hour

        # Pre-filter configuration
        self.pre_filter_enabled = config.get('pre_filter_enabled', True)
        self.pre_filter_min_movement_percent = config.get('pre_filter_min_movement_percent', 0.5)
        self.pre_filter_min_volume_ratio = config.get('pre_filter_min_volume_ratio', 0.5)
        self.pre_filter_max_spread_percent = config.get('pre_filter_max_spread_percent', 0.5)
        self.pre_filter_min_atr = config.get('pre_filter_min_atr', 0.50)

        # Intraday volume screener configuration
        self.intraday_screener_enabled = config.get('intraday_screener_enabled', True)
        self.intraday_volume_multiplier = config.get('intraday_volume_multiplier', 5.0)
        self.intraday_screener_sample_size = config.get('intraday_screener_sample_size', 1000)

        logger.info(
            f"Market scanner initialized: batch_size={self.batch_size}, "
            f"min_data_points={self.min_data_points}, "
            f"cache_enabled={self.cache_enabled}, "
            f"cache_ttl={self.cache_ttl_seconds}s, "
            f"pre_filter_enabled={self.pre_filter_enabled}, "
            f"intraday_screener_enabled={self.intraday_screener_enabled}"
        )

    def _get_cache_key(self) -> str:
        """Generate cache key for universe cache"""
        import os
        env = os.environ.get('ENVIRONMENT', 'dev')
        profile = os.environ.get('PROFILE', 'aggressive')
        stage = os.environ.get('STAGE', 'blue')
        return f"trading_universe_{env}_{profile}_{stage}"

    def _get_cached_universe(self) -> Optional[List[str]]:
        """
        Get cached trading universe from DynamoDB if available and not expired

        Returns:
            List of symbols if cache hit, None if cache miss or expired
        """
        if not self.cache_enabled:
            return None

        try:
            cache_key = self._get_cache_key()
            logger.info(f"Checking cache for key: {cache_key}")

            response = self.dynamodb_table.get_item(Key={'cache_key': cache_key})

            if 'Item' not in response:
                logger.info("Cache miss: No cached universe found")
                return None

            item = response['Item']
            cached_time = item.get('timestamp', 0)
            current_time = int(time.time())
            age_seconds = current_time - cached_time

            # Check if cache is expired
            if age_seconds > self.cache_ttl_seconds:
                logger.info(f"Cache expired: {age_seconds}s old (TTL: {self.cache_ttl_seconds}s)")
                return None

            symbols = item.get('symbols', [])
            logger.info(f"✅ Cache HIT: {len(symbols)} symbols (age: {age_seconds}s)")
            return symbols

        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None

    def _save_to_cache(self, symbols: List[str]) -> None:
        """
        Save trading universe to DynamoDB cache

        Args:
            symbols: List of symbols to cache
        """
        if not self.cache_enabled:
            return

        try:
            cache_key = self._get_cache_key()
            current_time = int(time.time())
            ttl = current_time + self.cache_ttl_seconds

            self.dynamodb_table.put_item(
                Item={
                    'cache_key': cache_key,
                    'symbols': symbols,
                    'timestamp': current_time,
                    'ttl': ttl,
                    'count': len(symbols)
                }
            )

            logger.info(f"💾 Cached {len(symbols)} symbols (TTL: {self.cache_ttl_seconds}s)")

        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def get_tradeable_universe(
        self,
        min_price: float = 5.0,
        max_price: float = 1000.0,
        min_volume: int = 1_000_000,
        max_symbols: int = 50,
        exclude_symbols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get a list of tradeable stocks based on liquidity and price filters

        Uses DynamoDB caching to avoid expensive market scans on every execution.
        Cache TTL is configurable (default: 1 hour).

        Args:
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average daily volume
            max_symbols: Maximum number of symbols to return
            exclude_symbols: Symbols to exclude (blacklist)

        Returns:
            List of stock symbols
        """
        # Check cache first
        cached_symbols = self._get_cached_universe()
        if cached_symbols is not None:
            # Apply exclusions to cached result
            if exclude_symbols:
                cached_symbols = [s for s in cached_symbols if s not in exclude_symbols]
            return cached_symbols

        # Cache miss - perform full market scan
        logger.info("Scanning market for tradeable stocks...")

        try:
            # Get all active, tradeable US equities
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )

            assets = self.alpaca_client.trading_client.get_all_assets(request)

            # Filter for tradeable stocks
            tradeable_symbols = []
            for asset in assets:
                # Must be tradeable and not fractionable (for simplicity)
                if not asset.tradable:
                    continue

                # Note: Removed easy_to_borrow check - we're buying long, not shorting
                # Hard-to-borrow stocks often have the best momentum for day trading

                # Exclude specific exchanges if needed (e.g., OTC)
                if asset.exchange in ['OTC', 'OTCBB']:
                    continue

                # Add to list
                tradeable_symbols.append(asset.symbol)

            logger.info(f"Found {len(tradeable_symbols)} tradeable stocks")

            # Exclude blacklisted symbols
            if exclude_symbols:
                tradeable_symbols = [
                    s for s in tradeable_symbols if s not in exclude_symbols
                ]

            # Get recent price and volume data to filter
            filtered_symbols = self._filter_by_liquidity(
                tradeable_symbols, min_price, max_price, min_volume, max_symbols
            )

            logger.info(f"Filtered to {len(filtered_symbols)} liquid stocks")

            # Save to cache for future executions
            self._save_to_cache(filtered_symbols)

            return filtered_symbols

        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            # Return a default universe as fallback
            return self._get_default_universe(exclude_symbols)

    def _filter_by_liquidity(
        self,
        symbols: List[str],
        min_price: float,
        max_price: float,
        min_volume: int,
        max_symbols: int,
    ) -> List[str]:
        """
        Filter symbols by price and volume (liquidity)

        Args:
            symbols: List of symbols to filter
            min_price: Minimum price
            max_price: Maximum price
            min_volume: Minimum average volume
            max_symbols: Maximum symbols to return

        Returns:
            Filtered list of symbols
        """
        # Fetch recent data for liquidity check (last 5 days)
        end = datetime.utcnow()
        start = end - timedelta(days=7)

        # Process in batches to avoid rate limits (Alpaca allows ~200 symbols at once)
        batch_size = self.batch_size
        filtered = []

        # Tracking statistics
        stats = {
            'total_symbols_input': len(symbols),
            'batches_processed': 0,
            'batches_failed': 0,
            'symbols_in_response': 0,
            'rejected_no_data': 0,
            'rejected_insufficient_days': 0,
            'rejected_price_too_low': 0,
            'rejected_price_too_high': 0,
            'rejected_volume': 0,
            'passed_filter': 0
        }

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                historical_data = self.alpaca_client.get_historical_bars(
                    batch, start, end
                )
                stats['batches_processed'] += 1
                stats['symbols_in_response'] += len(historical_data)

                # Log how many symbols were requested vs returned
                logger.info(f"Batch {batch_num}: requested {len(batch)} symbols, received {len(historical_data)} responses")

                for symbol, df in historical_data.items():
                    if df.empty:
                        stats['rejected_no_data'] += 1
                        continue

                    if len(df) < self.min_data_points:
                        stats['rejected_insufficient_days'] += 1
                        continue

                    # Check price range
                    avg_price = df['close'].mean()
                    if avg_price < min_price:
                        stats['rejected_price_too_low'] += 1
                        continue
                    if avg_price > max_price:
                        stats['rejected_price_too_high'] += 1
                        continue

                    # Check volume
                    avg_volume = df['volume'].mean()
                    if avg_volume < min_volume:
                        stats['rejected_volume'] += 1
                        continue

                    # Calculate liquidity score (volume * price)
                    liquidity_score = avg_volume * avg_price
                    stats['passed_filter'] += 1

                    filtered.append({
                        'symbol': symbol,
                        'price': avg_price,
                        'volume': avg_volume,
                        'liquidity_score': liquidity_score,
                    })

            except Exception as e:
                stats['batches_failed'] += 1
                logger.warning(f"Error fetching data for batch {batch_num}: {e}")
                continue

        # Log detailed statistics
        logger.info(f"Filtering statistics: {stats}")
        logger.info(f"Filter criteria: price ${min_price}-${max_price}, min_volume {min_volume:,}, max_symbols {max_symbols}")

        # Calculate missing symbols
        symbols_missing = stats['total_symbols_input'] - stats['symbols_in_response']
        if symbols_missing > 0:
            logger.warning(f"{symbols_missing} symbols had no data returned from Alpaca (possibly IEX feed limitations)")

        # Sort by liquidity score and take top N
        filtered.sort(key=lambda x: x['liquidity_score'], reverse=True)
        top_symbols = [s['symbol'] for s in filtered[:max_symbols]]

        return top_symbols

    def _get_default_universe(
        self, exclude_symbols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get default trading universe as fallback

        Args:
            exclude_symbols: Symbols to exclude

        Returns:
            List of default symbols
        """
        # High-liquidity stocks as fallback
        default = [
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # FAANG
            'NVDA', 'TSLA', 'AMD', 'NFLX',  # Tech
            'JPM', 'BAC', 'WFC', 'GS',  # Financials
            'JNJ', 'UNH', 'PFE', 'ABBV',  # Healthcare
            'XOM', 'CVX', 'COP',  # Energy
            'DIS', 'NKE', 'MCD', 'SBUX',  # Consumer
        ]

        if exclude_symbols:
            default = [s for s in default if s not in exclude_symbols]

        return default[:30]  # Limit to 30 for testing

    def scan_intraday_volume_spikes(
        self,
        min_price: float = 5.0,
        max_price: float = 1000.0,
        exclude_symbols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Scan for stocks with unusually high volume TODAY (intraday volume spike detection)

        This catches momentum stocks that wouldn't normally be in the universe due to
        low historical average volume, but are experiencing a breakout with high volume today.

        Args:
            min_price: Minimum stock price
            max_price: Maximum stock price
            exclude_symbols: Symbols to exclude (blacklist)

        Returns:
            List of symbols with volume spikes
        """
        if not self.intraday_screener_enabled:
            logger.info("Intraday volume screener disabled")
            return []

        logger.info(f"Scanning for intraday volume spikes (sample size: {self.intraday_screener_sample_size})...")

        try:
            # Get all tradeable stocks
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )

            assets = self.alpaca_client.trading_client.get_all_assets(request)

            # Filter for tradeable stocks (exclude OTC, non-tradeable)
            tradeable_symbols = []
            for asset in assets:
                if not asset.tradable:
                    continue
                if asset.exchange in ['OTC', 'OTCBB']:
                    continue
                tradeable_symbols.append(asset.symbol)

            # Exclude blacklisted symbols
            if exclude_symbols:
                tradeable_symbols = [s for s in tradeable_symbols if s not in exclude_symbols]

            # Sample a diverse set of stocks (evenly distributed across the alphabet for diversity)
            import random
            random.seed(datetime.utcnow().hour)  # Same sample for the hour
            sample_symbols = random.sample(
                tradeable_symbols,
                min(self.intraday_screener_sample_size, len(tradeable_symbols))
            )

            logger.info(f"Sampling {len(sample_symbols)} stocks from {len(tradeable_symbols)} tradeable stocks")

            # Fetch historical data (last 7 days for volume average)
            end = datetime.utcnow()
            start = end - timedelta(days=10)

            historical_data = self.alpaca_client.get_historical_bars(sample_symbols, start, end)

            # Fetch today's intraday data (since market open)
            market_open = datetime.utcnow().replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM ET in UTC
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            intraday_data = self.alpaca_client.get_historical_bars(
                sample_symbols,
                market_open,
                datetime.utcnow(),
                timeframe=TimeFrame(5, TimeFrameUnit.Minute)
            )

            # Find volume spikes
            volume_spikes = []
            stats = {
                'sampled': len(sample_symbols),
                'with_data': 0,
                'volume_spikes': 0,
                'rejected_price': 0,
                'rejected_no_spike': 0,
            }

            for symbol in sample_symbols:
                hist_df = historical_data.get(symbol)
                intraday_df = intraday_data.get(symbol)

                # Need both historical and intraday data
                if hist_df is None or hist_df.empty or len(hist_df) < 3:
                    continue
                if intraday_df is None or intraday_df.empty:
                    continue

                stats['with_data'] += 1

                # Check price range
                current_price = intraday_df['close'].iloc[-1]
                if current_price < min_price or current_price > max_price:
                    stats['rejected_price'] += 1
                    continue

                # Calculate historical average daily volume
                avg_daily_volume = hist_df['volume'].mean()

                # Calculate today's volume so far
                today_volume = intraday_df['volume'].sum()

                # Check for volume spike (today's volume > multiplier * average)
                if avg_daily_volume > 0:
                    volume_ratio = today_volume / avg_daily_volume

                    if volume_ratio >= self.intraday_volume_multiplier:
                        stats['volume_spikes'] += 1
                        volume_spikes.append({
                            'symbol': symbol,
                            'volume_ratio': volume_ratio,
                            'current_price': current_price,
                            'today_volume': today_volume,
                            'avg_volume': avg_daily_volume,
                        })
                        logger.info(
                            f"Volume spike detected: {symbol} @ ${current_price:.2f} "
                            f"(volume: {today_volume:,.0f} vs avg: {avg_daily_volume:,.0f}, "
                            f"ratio: {volume_ratio:.1f}x)"
                        )
                    else:
                        stats['rejected_no_spike'] += 1

            logger.info(
                f"Intraday screener stats: {stats['volume_spikes']} spikes found from "
                f"{stats['with_data']} stocks with data (sampled {stats['sampled']})"
            )

            # Sort by volume ratio (highest first)
            volume_spikes.sort(key=lambda x: x['volume_ratio'], reverse=True)

            return [s['symbol'] for s in volume_spikes]

        except Exception as e:
            logger.error(f"Error in intraday volume screener: {e}")
            return []

    def pre_filter_candidates(
        self,
        symbols: List[str],
        historical_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], Dict]:
        """
        Lightweight pre-filter to eliminate symbols before expensive strategy analysis

        Filters based on:
        - Price movement: Skip if abs(change) < 0.5% in last hour
        - Volume: Skip if current volume < 50% of 20-day average
        - ATR: Skip if ATR < $0.50 (too stable for day trading)
        - Spread: Skip if bid-ask spread > 0.5% (note: limited by IEX data)

        Args:
            symbols: List of symbols to filter
            historical_data: Dict of historical DataFrames (for ATR calculation)
            intraday_data: Dict of intraday DataFrames (for movement and volume checks)

        Returns:
            Tuple of (filtered_symbols, filter_stats)
        """
        if not self.pre_filter_enabled:
            logger.info("Pre-filter disabled, passing all symbols through")
            return symbols, {'total': len(symbols), 'passed': len(symbols), 'filtered': 0}

        logger.info(f"Pre-filtering {len(symbols)} candidates...")

        filtered_symbols = []
        stats = {
            'total': len(symbols),
            'passed': 0,
            'filtered_movement': 0,
            'filtered_volume': 0,
            'filtered_atr': 0,
            'filtered_no_data': 0
        }

        for symbol in symbols:
            # Get historical and intraday data
            hist_df = historical_data.get(symbol)
            intraday_df = intraday_data.get(symbol)

            # Skip if no data
            if hist_df is None or hist_df.empty:
                stats['filtered_no_data'] += 1
                continue

            # Check 1: Price movement in last hour (requires intraday data)
            if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 12:
                last_hour_start = intraday_df['close'].iloc[-12]
                last_price = intraday_df['close'].iloc[-1]
                movement_pct = abs((last_price - last_hour_start) / last_hour_start)

                if movement_pct < (self.pre_filter_min_movement_percent / 100):
                    stats['filtered_movement'] += 1
                    logger.debug(f"{symbol}: Filtered by movement ({movement_pct*100:.2f}% < {self.pre_filter_min_movement_percent}%)")
                    continue

            # Check 2: Volume ratio (current vs 20-day average)
            if intraday_df is not None and not intraday_df.empty and len(hist_df) >= 20:
                avg_volume_20d = hist_df['volume'].tail(20).mean()
                current_volume = intraday_df['volume'].sum()  # Total volume so far today

                if avg_volume_20d > 0:
                    volume_ratio = current_volume / avg_volume_20d

                    if volume_ratio < self.pre_filter_min_volume_ratio:
                        stats['filtered_volume'] += 1
                        logger.debug(f"{symbol}: Filtered by volume (ratio {volume_ratio:.2f} < {self.pre_filter_min_volume_ratio})")
                        continue

            # Check 2.5: Explosive Mover Bypass (10%+ intraday movement bypasses ATR filter)
            # This catches stocks like SEMR, SGBX, KZIA that gap up big then consolidate
            if intraday_df is not None and not intraday_df.empty:
                open_price = intraday_df['open'].iloc[0]
                current_price = intraday_df['close'].iloc[-1]
                intraday_movement_pct = abs((current_price - open_price) / open_price) * 100

                if intraday_movement_pct >= 10.0:
                    # Explosive mover - bypass ATR filter
                    filtered_symbols.append(symbol)
                    stats['passed'] += 1
                    logger.info(f"✨ {symbol}: EXPLOSIVE MOVER bypass (+{intraday_movement_pct:.1f}%) - skipping ATR filter")
                    continue

            # Check 3: ATR (Average True Range) - volatility filter
            if len(hist_df) >= 14:
                # Calculate ATR (14-day)
                high_low = hist_df['high'] - hist_df['low']
                high_close = abs(hist_df['high'] - hist_df['close'].shift())
                low_close = abs(hist_df['low'] - hist_df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.tail(14).mean()

                if atr < self.pre_filter_min_atr:
                    stats['filtered_atr'] += 1
                    logger.debug(f"{symbol}: Filtered by ATR (${atr:.2f} < ${self.pre_filter_min_atr})")
                    continue

            # Check 4: Spread filter (would need real-time quote data, skip for now with IEX feed)
            # IEX feed doesn't provide bid-ask spread, so we skip this check

            # Passed all filters
            filtered_symbols.append(symbol)
            stats['passed'] += 1

        stats['filtered'] = stats['total'] - stats['passed']
        filter_rate = (stats['filtered'] / stats['total'] * 100) if stats['total'] > 0 else 0

        logger.info(
            f"Pre-filter complete: {stats['passed']}/{stats['total']} passed ({filter_rate:.1f}% filtered out) - "
            f"movement: {stats['filtered_movement']}, volume: {stats['filtered_volume']}, "
            f"atr: {stats['filtered_atr']}, no_data: {stats['filtered_no_data']}"
        )

        return filtered_symbols, stats

    def scan_for_opportunities(
        self,
        strategy_manager,
        symbols: List[str],
        benchmark_data: pd.DataFrame,
        min_consensus: float = 0.30,
    ) -> List[Dict]:
        """
        Scan symbols and return those with strong signals

        Args:
            strategy_manager: StrategyManager instance
            symbols: List of symbols to scan
            benchmark_data: Benchmark data (e.g., SPY)
            min_consensus: Minimum consensus score to include

        Returns:
            List of opportunities with symbol and consensus data
        """
        logger.info(f"Scanning {len(symbols)} symbols for opportunities...")

        opportunities = []

        # Get historical data
        end = datetime.utcnow()
        start = end - timedelta(days=250)

        historical_data = self.alpaca_client.get_historical_bars(symbols, start, end)

        # Scan each symbol
        for symbol, data in historical_data.items():
            if data.empty or len(data) < 50:
                continue

            try:
                # Generate consensus signal
                action, consensus_score, signals = strategy_manager.generate_consensus_signal(
                    symbol, data, benchmark_data=benchmark_data
                )

                # Only include strong signals
                if action != 'hold' and abs(consensus_score) >= min_consensus:
                    opportunities.append({
                        'symbol': symbol,
                        'action': action,
                        'consensus_score': consensus_score,
                        'strategy_signals': signals,
                        'current_price': float(data['close'].iloc[-1]),
                        'volume': float(data['volume'].iloc[-1]),
                    })

            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue

        # Sort by consensus score
        opportunities.sort(key=lambda x: abs(x['consensus_score']), reverse=True)

        logger.info(f"Found {len(opportunities)} opportunities")
        return opportunities

    def get_sector_screener(
        self, sectors: List[str], max_per_sector: int = 5
    ) -> List[str]:
        """
        Get stocks from specific sectors

        Args:
            sectors: List of sector names
            max_per_sector: Max stocks per sector

        Returns:
            List of symbols
        """
        # This would require additional data source for sector classification
        # For now, return default universe
        logger.warning("Sector screening not yet implemented, using default universe")
        return self._get_default_universe()
