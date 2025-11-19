"""
Advanced Short-Term Trend Quantitative Strategy for 15-minute timeframe
Combines 8 technical indicators: EMA, MACD, ROC, ADX, +DI, -DI, KAMA, and MOM

File: strategies/advanced_short_term_strategy.py
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy

class AdvancedShortTermStrategy(BaseStrategy):
    """
    Advanced Short-Term Trend Quantitative Strategy using 15-minute timeframe
    
    Strategy combines 8 technical indicators for high-confidence signals:
    1. EMA (Exponential Moving Average) - Multiple periods for trend direction
    2. MACD (Moving Average Convergence Divergence) - Momentum and trend strength
    3. ROC (Rate of Change) - Price momentum
    4. ADX (Average Directional Index) - Trend strength measurement
    5. +DI (Positive Directional Indicator) - Bullish pressure
    6. -DI (Negative Directional Indicator) - Bearish pressure
    7. KAMA (Kaufman Adaptive Moving Average) - Adaptive trend line
    8. MOM (Momentum) - Price momentum confirmation
    
    Trading Signals:
    - Strong Buy: 7-8 indicators align bullish, ADX > 25, confidence 80-100%
    - Buy: 5-6 indicators align bullish, ADX > 20, confidence 60-80%
    - Strong Sell: 7-8 indicators align bearish, ADX > 25, confidence 80-100%
    - Sell: 5-6 indicators align bearish, ADX > 20, confidence 60-80%
    - Hold: Less than 5 indicators align or ADX < 20 (weak trend)
    """
    
    def __init__(self, trader):
        """Initialize the Advanced Short-Term Trend Strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # Timeframe configuration
        self.kline_interval = '15m'  # 15-minute timeframe
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
        self.lookback_period = 100  # Number of candles for analysis
        self.training_lookback = 100  # For compatibility with TradingManager
        
        # EMA Parameters
        self.ema_fast = 5
        self.ema_medium = 26
        self.ema_slow = 50
        
        # MACD Parameters
        self.macd_fast = 5
        self.macd_slow = 26
        self.macd_signal = 9
        
        # ROC Parameters
        self.roc_period = 10
        self.roc_threshold = 0.5  # Minimum ROC for signal confirmation (0.5%)
        
        # ADX Parameters
        self.adx_period = 14
        self.adx_min_threshold = 16  # Minimum ADX for trend confirmation
        self.adx_strong_threshold = 25  # Strong trend threshold
        
        # KAMA Parameters
        self.kama_period = 30
        self.kama_slope_threshold = 0.001  # Minimum KAMA slope for trend confirmation
        
        # MOM Parameters
        self.mom_period = 10
        self.mom_threshold = 0  # Momentum threshold
        
        # Position Management
        self.max_position_hold_time = 720  # 12 hours maximum hold time (in minutes)
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
        
        self.logger.info("Advanced Short-Term Trend Strategy initialized for 15m timeframe")
        self.logger.info(f"Indicators: EMA({self.ema_fast},{self.ema_medium},{self.ema_slow}), "
                        f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), "
                        f"ROC({self.roc_period}), ADX({self.adx_period}), "
                        f"KAMA({self.kama_period}), MOM({self.mom_period})")
    
    def calculate_indicators(self, df):
        """
        Calculate all 8 technical indicators
        
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
            
            # 1. EMA - Exponential Moving Averages
            ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
            ema_medium = talib.EMA(close, timeperiod=self.ema_medium)
            ema_slow = talib.EMA(close, timeperiod=self.ema_slow)
            
            # 2. MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            
            # 3. ROC - Rate of Change
            roc = talib.ROC(close, timeperiod=self.roc_period)
            
            # 4. ADX - Average Directional Index
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            
            # 5. +DI - Positive Directional Indicator
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            
            # 6. -DI - Negative Directional Indicator
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            
            # 7. KAMA - Kaufman Adaptive Moving Average
            kama = talib.KAMA(close, timeperiod=self.kama_period)
            
            # Calculate KAMA slope
            kama_slope = np.zeros_like(kama)
            for i in range(1, len(kama)):
                if not np.isnan(kama[i]) and not np.isnan(kama[i-1]) and kama[i-1] != 0:
                    kama_slope[i] = (kama[i] - kama[i-1]) / kama[i-1]
            
            # 8. MOM - Momentum
            mom = talib.MOM(close, timeperiod=self.mom_period)
            
            return {
                'ema_fast': ema_fast,
                'ema_medium': ema_medium,
                'ema_slow': ema_slow,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'roc': roc,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'kama': kama,
                'kama_slope': kama_slope,
                'mom': mom,
                'close': close
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def analyze_trend(self, indicators):
        """
        Analyze trend using all 8 technical indicators
        
        Args:
            indicators (dict): Dictionary of indicator values
            
        Returns:
            dict: Trend analysis results including signal, confidence, and reasoning
        """
        try:
            # Get latest values (using -2 for confirmed candle, not current forming candle)
            close = indicators['close'][-2]
            ema_fast = indicators['ema_fast'][-2]
            ema_medium = indicators['ema_medium'][-2]
            ema_slow = indicators['ema_slow'][-2]
            macd = indicators['macd'][-2]
            macd_signal = indicators['macd_signal'][-2]
            macd_hist = indicators['macd_hist'][-2]
            roc = indicators['roc'][-2]
            adx = indicators['adx'][-2]
            plus_di = indicators['plus_di'][-2]
            minus_di = indicators['minus_di'][-2]
            kama = indicators['kama'][-2]
            kama_slope = indicators['kama_slope'][-2]
            mom = indicators['mom'][-2]
            
            # Check for recent MACD histogram change
            macd_hist_prev = indicators['macd_hist'][-3] if len(indicators['macd_hist']) > 2 else 0
            
            # Check for NaN values
            required_values = [close, ema_fast, ema_medium, ema_slow, macd, macd_signal, 
                             macd_hist, roc, adx, plus_di, minus_di, kama, mom]
            if np.isnan(required_values).any():
                return {
                    'signal': 0,
                    'confidence': 0,
                    'strength': 'insufficient_data',
                    'reason': 'Insufficient data (NaN values)'
                }
            
            # Initialize scoring system
            bullish_score = 0
            bearish_score = 0
            reasons = []
            
            # 1. EMA Analysis (3 points possible)
            if close > ema_slow and ema_fast > ema_medium > ema_slow:
                bullish_score += 3
                reasons.append(f"✓ EMA aligned bullish: Price({close:.2f}) > Fast({ema_fast:.2f}) > Med({ema_medium:.2f}) > Slow({ema_slow:.2f})")
            elif close < ema_slow and ema_fast < ema_medium < ema_slow:
                bearish_score += 3
                reasons.append(f"✓ EMA aligned bearish: Price({close:.2f}) < Fast({ema_fast:.2f}) < Med({ema_medium:.2f}) < Slow({ema_slow:.2f})")
            elif close > ema_slow:
                bullish_score += 1
                reasons.append(f"• Price above EMA slow: {close:.2f} > {ema_slow:.2f}")
            elif close < ema_slow:
                bearish_score += 1
                reasons.append(f"• Price below EMA slow: {close:.2f} < {ema_slow:.2f}")
            
            # 2. MACD Analysis (2 points possible)
            if macd > macd_signal and macd_hist > 0 and macd_hist > macd_hist_prev:
                bullish_score += 2
                reasons.append(f"✓ MACD bullish: MACD({macd:.4f}) > Signal({macd_signal:.4f}), increasing histogram")
            elif macd < macd_signal and macd_hist < 0 and macd_hist < macd_hist_prev:
                bearish_score += 2
                reasons.append(f"✓ MACD bearish: MACD({macd:.4f}) < Signal({macd_signal:.4f}), decreasing histogram")
            elif macd > 0:
                bullish_score += 1
                reasons.append(f"• MACD positive: {macd:.4f}")
            elif macd < 0:
                bearish_score += 1
                reasons.append(f"• MACD negative: {macd:.4f}")
            
            # 3. ROC Analysis (1 point)
            if roc > self.roc_threshold:
                bullish_score += 1
                reasons.append(f"✓ ROC bullish: {roc:.2f}%")
            elif roc < -self.roc_threshold:
                bearish_score += 1
                reasons.append(f"✓ ROC bearish: {roc:.2f}%")
            
            # 4. ADX + DI Analysis (2 points possible)
            di_diff = abs(plus_di - minus_di)
            if adx >= self.adx_min_threshold:
                if plus_di > minus_di and di_diff > 5:
                    bullish_score += 2
                    reasons.append(f"✓ ADX strong bullish: ADX({adx:.2f}), +DI({plus_di:.2f}) > -DI({minus_di:.2f})")
                elif minus_di > plus_di and di_diff > 5:
                    bearish_score += 2
                    reasons.append(f"✓ ADX strong bearish: ADX({adx:.2f}), -DI({minus_di:.2f}) > +DI({plus_di:.2f})")
                elif plus_di > minus_di:
                    bullish_score += 1
                    reasons.append(f"• ADX trending bullish: ADX({adx:.2f}), +DI > -DI")
                elif minus_di > plus_di:
                    bearish_score += 1
                    reasons.append(f"• ADX trending bearish: ADX({adx:.2f}), -DI > +DI")
            else:
                reasons.append(f"⚠ ADX weak: {adx:.2f} < {self.adx_min_threshold}")
            
            # 5. KAMA Analysis (1 point)
            if close > kama and kama_slope > self.kama_slope_threshold:
                bullish_score += 1
                reasons.append(f"✓ KAMA bullish: Price({close:.2f}) > KAMA({kama:.2f}), slope {kama_slope*100:.3f}%")
            elif close < kama and kama_slope < -self.kama_slope_threshold:
                bearish_score += 1
                reasons.append(f"✓ KAMA bearish: Price({close:.2f}) < KAMA({kama:.2f}), slope {kama_slope*100:.3f}%")
            
            # 6. MOM Analysis (1 point)
            if mom > self.mom_threshold:
                bullish_score += 1
                reasons.append(f"✓ Momentum bullish: {mom:.2f}")
            elif mom < -self.mom_threshold:
                bearish_score += 1
                reasons.append(f"✓ Momentum bearish: {mom:.2f}")
            
            # Total possible score: 10 points
            # Determine signal based on alignment
            signal = 0
            confidence = 0
            trend_strength = 'weak'
            
            # Strong bullish: 7+ indicators align
            if bullish_score >= 7 and adx >= self.adx_strong_threshold:
                signal = 1
                confidence = min(100, 70 + bullish_score * 3 + (adx - self.adx_strong_threshold) * 2)
                trend_strength = 'strong_bullish'
            # Moderate bullish: 5-6 indicators align
            elif bullish_score >= 5 and adx >= self.adx_min_threshold:
                signal = 1
                confidence = min(100, 50 + bullish_score * 5 + (adx - self.adx_min_threshold))
                trend_strength = 'moderate_bullish'
            # Strong bearish: 7+ indicators align
            elif bearish_score >= 7 and adx >= self.adx_strong_threshold:
                signal = -1
                confidence = min(100, 70 + bearish_score * 3 + (adx - self.adx_strong_threshold) * 2)
                trend_strength = 'strong_bearish'
            # Moderate bearish: 5-6 indicators align
            elif bearish_score >= 5 and adx >= self.adx_min_threshold:
                signal = -1
                confidence = min(100, 50 + bearish_score * 5 + (adx - self.adx_min_threshold))
                trend_strength = 'moderate_bearish'
            else:
                trend_strength = 'neutral'
                reasons.append(f"⚠ Insufficient alignment: Bullish={bullish_score}, Bearish={bearish_score}")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'strength': trend_strength,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'adx': adx,
                'close': close,
                'reason': ' | '.join(reasons)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return {
                'signal': 0,
                'confidence': 0,
                'strength': 'error',
                'reason': str(e)
            }
    
    def generate_signal(self, klines=None):
        """
        Generate trading signal based on multi-indicator analysis
        
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
            self.logger.info("="*80)
            self.logger.info("Advanced Short-Term Trend Strategy Analysis")
            self.logger.info("="*80)
            self.logger.info(f"Signal: {analysis['signal']} | Confidence: {analysis.get('confidence', 0):.1f}% | Strength: {analysis['strength']}")
            self.logger.info(f"Bullish Score: {analysis.get('bullish_score', 0)}/10 | Bearish Score: {analysis.get('bearish_score', 0)}/10")
            self.logger.info(f"Current Price: {analysis.get('close', 0):.4f} | ADX: {analysis.get('adx', 0):.2f}")
            self.logger.info("-"*80)
            self.logger.info(f"Reasoning:\n{analysis['reason']}")
            self.logger.info("="*80)
            
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
            
            return False, f"Position OK (PnL: {pnl_pct*100:.2f}%, Max: {self.max_profit_reached*100:.2f}%)"
            
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
