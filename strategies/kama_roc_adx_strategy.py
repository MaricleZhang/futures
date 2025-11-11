"""
KAMA-ROC-ADX Strategy for 15-minute timeframe
Uses KAMA (Kaufman's Adaptive Moving Average), ROC (Rate of Change), and ADX to predict price trends

File: strategies/kama_roc_adx_strategy.py
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy

class KAMARocAdxStrategy(BaseStrategy):
    """
    KAMA-ROC-ADX trend prediction strategy using 15-minute timeframe

    Strategy Logic:
    1. KAMA (Kaufman's Adaptive Moving Average) - Adaptive trend line that adjusts to market volatility
    2. ROC (Rate of Change) - Measures price momentum
    3. ADX (Average Directional Index) - Measures trend strength

    Trading Signals:
    - Buy: Price > KAMA AND ROC > 0 AND ADX > threshold (strong uptrend with momentum)
    - Sell: Price < KAMA AND ROC < 0 AND ADX > threshold (strong downtrend with momentum)
    - Hold: ADX below threshold or conflicting signals (weak trend or consolidation)
    """

    def __init__(self, trader):
        """Initialize the KAMA-ROC-ADX strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()

        # Timeframe configuration
        self.kline_interval = '15m'  # 15-minute timeframe
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
        self.lookback_period = 50  # Number of candles for analysis
        self.training_lookback = 100  # For compatibility with TradingManager

        # KAMA Parameters
        self.kama_period = 30  # KAMA period (default: 30)

        # ROC Parameters
        self.roc_period = 10  # ROC period (default: 10)
        self.roc_threshold = 0.5  # Minimum ROC for signal confirmation (0.5%)

        # ADX Parameters
        self.adx_period = 14  # Standard ADX period
        self.adx_min_threshold = 20  # Minimum ADX for trend confirmation
        self.adx_strong_threshold = 35  # Strong trend threshold

        # Additional Filters
        self.use_kama_slope = True  # Use KAMA slope as additional filter
        self.kama_slope_threshold = 0.001  # Minimum KAMA slope for trend confirmation

        # Position Management
        self.max_position_hold_time = 720  # 12 hours maximum hold time
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit

        # Trailing Stop Configuration
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.03  # Activate at 3% profit
        self.trailing_stop_distance = 0.015  # 1.5% trailing distance

        # Position Tracking
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_signal = 0
        self.last_signal_time = None
        self.trailing_stop_price = None

        self.logger.info("KAMA-ROC-ADX Strategy initialized for 15m timeframe")

    def calculate_indicators(self, df):
        """
        Calculate KAMA, ROC, and ADX indicators

        Args:
            df (DataFrame): OHLCV data

        Returns:
            dict: Dictionary containing all indicator values
        """
        try:
            # Extract price data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Calculate KAMA - Kaufman's Adaptive Moving Average
            kama = talib.KAMA(close, timeperiod=self.kama_period)

            # Calculate ROC - Rate of Change
            roc = talib.ROC(close, timeperiod=self.roc_period)

            # Calculate ADX - Average Directional Index
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)

            # Calculate +DI and -DI for additional analysis
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

            # Calculate KAMA slope (rate of change of KAMA)
            kama_slope = np.zeros_like(kama)
            for i in range(1, len(kama)):
                if not np.isnan(kama[i]) and not np.isnan(kama[i-1]) and kama[i-1] != 0:
                    kama_slope[i] = (kama[i] - kama[i-1]) / kama[i-1]

            return {
                'kama': kama,
                'kama_slope': kama_slope,
                'roc': roc,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'close': close
            }

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def analyze_trend(self, indicators):
        """
        Analyze trend using KAMA, ROC, and ADX

        Args:
            indicators (dict): Dictionary of indicator values

        Returns:
            dict: Trend analysis results
        """
        try:
            # Get latest values (using -2 for confirmed candle, not current forming candle)
            kama = indicators['kama'][-2]
            kama_slope = indicators['kama_slope'][-2]
            roc = indicators['roc'][-2]
            adx = indicators['adx'][-2]
            plus_di = indicators['plus_di'][-2]
            minus_di = indicators['minus_di'][-2]
            close = indicators['close'][-2]

            # Check for NaN values
            if np.isnan([kama, roc, adx, close]).any():
                return {
                    'signal': 0,
                    'strength': 0,
                    'reason': 'Insufficient data (NaN values)'
                }

            # Calculate price position relative to KAMA
            price_above_kama = close > kama
            price_below_kama = close < kama
            price_kama_distance = abs(close - kama) / kama * 100  # Distance in percentage

            # Trend strength based on ADX
            trend_strength = 'weak'
            if adx >= self.adx_strong_threshold:
                trend_strength = 'strong'
            elif adx >= self.adx_min_threshold:
                trend_strength = 'moderate'

            # Check KAMA slope condition
            kama_slope_bullish = kama_slope > self.kama_slope_threshold
            kama_slope_bearish = kama_slope < -self.kama_slope_threshold

            # ROC momentum check
            roc_bullish = roc > self.roc_threshold
            roc_bearish = roc < -self.roc_threshold

            # DI confirmation
            di_bullish = plus_di > minus_di
            di_bearish = minus_di > plus_di
            di_diff = abs(plus_di - minus_di)

            # Generate signal
            signal = 0
            reason = []
            confidence = 0

            # BULLISH SIGNAL CONDITIONS
            if (price_above_kama and roc_bullish and adx >= self.adx_min_threshold):
                signal = 1
                confidence = min(100, 40 + (adx - self.adx_min_threshold) * 1.5 + di_diff * 0.5)

                reason.append(f"Price above KAMA by {price_kama_distance:.2f}%")
                reason.append(f"ROC positive: {roc:.2f}%")
                reason.append(f"ADX trending: {adx:.2f} ({trend_strength})")

                if self.use_kama_slope and kama_slope_bullish:
                    reason.append(f"KAMA slope bullish: {kama_slope*100:.3f}%")
                    confidence += 10

                if di_bullish:
                    reason.append(f"+DI > -DI ({plus_di:.2f} > {minus_di:.2f})")
                    confidence += 10

            # BEARISH SIGNAL CONDITIONS
            elif (price_below_kama and roc_bearish and adx >= self.adx_min_threshold):
                signal = -1
                confidence = min(100, 40 + (adx - self.adx_min_threshold) * 1.5 + di_diff * 0.5)

                reason.append(f"Price below KAMA by {price_kama_distance:.2f}%")
                reason.append(f"ROC negative: {roc:.2f}%")
                reason.append(f"ADX trending: {adx:.2f} ({trend_strength})")

                if self.use_kama_slope and kama_slope_bearish:
                    reason.append(f"KAMA slope bearish: {kama_slope*100:.3f}%")
                    confidence += 10

                if di_bearish:
                    reason.append(f"-DI > +DI ({minus_di:.2f} > {plus_di:.2f})")
                    confidence += 10

            # NO CLEAR SIGNAL
            else:
                reason.append("No clear trend signal")
                if adx < self.adx_min_threshold:
                    reason.append(f"ADX too low: {adx:.2f}")
                if abs(roc) < self.roc_threshold:
                    reason.append(f"ROC too weak: {roc:.2f}%")
                if (price_above_kama and roc_bearish) or (price_below_kama and roc_bullish):
                    reason.append("Conflicting signals between price/KAMA and ROC")

            return {
                'signal': signal,
                'confidence': confidence,
                'strength': trend_strength,
                'adx': adx,
                'roc': roc,
                'kama': kama,
                'kama_slope': kama_slope,
                'close': close,
                'reason': ' | '.join(reason)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return {
                'signal': 0,
                'strength': 'error',
                'reason': str(e)
            }

    def generate_signal(self, klines=None):
        """
        Generate trading signal based on KAMA-ROC-ADX analysis

        Args:
            klines: K-line data (if None, will fetch from trader)

        Returns:
            int: Signal value (1=buy, -1=sell, 0=hold, 2=close position)
        """
        try:
            # Fetch k-line data if not provided
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )

            if not klines or len(klines) < self.lookback_period:
                self.logger.warning(f"Insufficient k-line data: {len(klines) if klines else 0}")
                return 0

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype(float)

            # Calculate indicators
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return 0

            # Analyze trend
            analysis = self.analyze_trend(indicators)

            # Log analysis
            self.logger.info(f"KAMA-ROC-ADX Analysis:")
            self.logger.info(f"  Signal: {analysis['signal']} | Confidence: {analysis.get('confidence', 0):.1f}%")
            self.logger.info(f"  Close: {analysis.get('close', 0):.4f} | KAMA: {analysis.get('kama', 0):.4f}")
            self.logger.info(f"  ROC: {analysis.get('roc', 0):.2f}% | ADX: {analysis.get('adx', 0):.2f}")
            self.logger.info(f"  Reason: {analysis['reason']}")

            # Update last signal
            if analysis['signal'] != 0:
                self.last_signal = analysis['signal']
                self.last_signal_time = datetime.now()

            return analysis['signal']

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return 0

    def should_close_position(self, position, current_price):
        """
        Check if position should be closed based on stop loss, take profit, or time

        Args:
            position: Current position info
            current_price: Current market price

        Returns:
            tuple: (should_close, reason)
        """
        try:
            if not position or position.get('positionAmt', 0) == 0:
                return False, "No position"

            position_amt = float(position.get('positionAmt', 0))
            entry_price = float(position.get('entryPrice', 0))

            if entry_price == 0:
                return False, "Invalid entry price"

            # Calculate profit/loss percentage
            if position_amt > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price

            # Update max profit for trailing stop
            if pnl_pct > self.max_profit_reached:
                self.max_profit_reached = pnl_pct

                # Activate trailing stop
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_stop_activation:
                    if position_amt > 0:  # Long position
                        self.trailing_stop_price = current_price * (1 - self.trailing_stop_distance)
                    else:  # Short position
                        self.trailing_stop_price = current_price * (1 + self.trailing_stop_distance)
                    self.logger.info(f"Trailing stop activated at {self.trailing_stop_price:.4f}")

            # Check trailing stop
            if self.trailing_stop_price:
                if position_amt > 0 and current_price <= self.trailing_stop_price:
                    return True, f"Trailing stop hit (price: {current_price:.4f}, stop: {self.trailing_stop_price:.4f})"
                elif position_amt < 0 and current_price >= self.trailing_stop_price:
                    return True, f"Trailing stop hit (price: {current_price:.4f}, stop: {self.trailing_stop_price:.4f})"

            # Check stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return True, f"Stop loss hit ({pnl_pct*100:.2f}%)"

            # Check take profit
            if pnl_pct >= self.take_profit_pct:
                return True, f"Take profit hit ({pnl_pct*100:.2f}%)"

            # Check holding time
            if self.position_entry_time:
                hold_time = (datetime.now() - self.position_entry_time).total_seconds() / 60
                if hold_time >= self.max_position_hold_time:
                    return True, f"Max hold time reached ({hold_time:.1f} minutes)"

            return False, f"Position OK (PnL: {pnl_pct*100:.2f}%)"

        except Exception as e:
            self.logger.error(f"Error checking position close: {str(e)}")
            return False, str(e)

    def monitor_position(self):
        """
        Monitor position and execute trading logic
        """
        try:
            # Get current position
            position = self.trader.get_position(self.trader.symbol)
            current_price = self.trader.get_market_price(self.trader.symbol)

            if not current_price:
                self.logger.warning("Failed to get current price")
                return

            position_amt = float(position.get('positionAmt', 0)) if position else 0

            # If we have a position
            if position_amt != 0:
                # Check if should close position
                should_close, reason = self.should_close_position(position, current_price)

                if should_close:
                    self.logger.info(f"Closing position: {reason}")
                    self.trader.close_position(self.trader.symbol)

                    # Reset position tracking
                    self.position_entry_time = None
                    self.position_entry_price = None
                    self.max_profit_reached = 0
                    self.trailing_stop_price = None
                else:
                    self.logger.info(f"Position status: {reason}")

            # If no position, check for entry signals
            else:
                signal = self.generate_signal()

                if signal == 1:  # Buy signal
                    self.logger.info("Opening LONG position")
                    trade_amount = self.trader.calculate_trade_amount(self.trader.symbol)
                    if trade_amount > 0:
                        self.trader.open_long(self.trader.symbol, trade_amount)
                        self.position_entry_time = datetime.now()
                        self.position_entry_price = current_price
                        self.max_profit_reached = 0
                        self.trailing_stop_price = None

                elif signal == -1:  # Sell signal
                    self.logger.info("Opening SHORT position")
                    trade_amount = self.trader.calculate_trade_amount(self.trader.symbol)
                    if trade_amount > 0:
                        self.trader.open_short(self.trader.symbol, trade_amount)
                        self.position_entry_time = datetime.now()
                        self.position_entry_price = current_price
                        self.max_profit_reached = 0
                        self.trailing_stop_price = None

        except Exception as e:
            self.logger.error(f"Error in monitor_position: {str(e)}")
