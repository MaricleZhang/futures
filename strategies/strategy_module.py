"""
Directional Index Strategy - 15 minute timeframe
Trading strategy based solely on Directional Movement indicators (ADX, +DI, -DI)
"""
import numpy as np
import pandas as pd
import talib
import time
import logging
from datetime import datetime
from strategies.base_strategy import BaseStrategy

class DirectionalIndexStrategy15m(BaseStrategy):
    """DirectionalIndexStrategy15m - Trading strategy based on Directional Movement indicators
    
    A simplified trading strategy that relies entirely on ADX, +DI, -DI and their rate of change
    to identify trend direction and strength for generating trading signals.
    
    Features:
    1. Uses only ADX and DI indicators for decision making
    2. Tracks rate of change in these indicators to identify early trend changes
    3. Adaptive thresholds based on recent market behavior
    4. Multiple confirmation requirements for signal generation
    """
    
    def __init__(self, trader):
        """Initialize the Directional Index strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K-line settings
        self.kline_interval = '15m'      # 15-minute timeframe
        self.check_interval = 300        # Check signal interval (seconds)
        self.lookback_period = 100       # Number of k-lines for indicators
        self.training_lookback = 200     # For compatibility with TradingManager
        
        # Indicator parameters
        self.adx_period = 14             # ADX calculation period
        self.di_period = 14              # DI calculation period
        self.roc_period = 5              # Rate of change period
        
        # Signal thresholds
        self.adx_strong_trend = 25       # ADX above this indicates strong trend
        self.di_crossover_threshold = 2  # Minimum difference between +DI and -DI
        self.adx_rising_threshold = 0.5  # Minimum ADX rising rate for trend confirmation
        self.di_change_threshold = 1.0   # Minimum DI change rate for trend confirmation
        
        # Position control parameters
        self.max_position_hold_time = 480  # Maximum position hold time (minutes)
        self.stop_loss_pct = 0.02        # Stop loss percentage (2%)
        self.take_profit_pct = 0.04      # Take profit percentage (4%)
        self.trailing_stop = True        # Enable trailing stop
        self.trailing_stop_activation = 0.02  # Activate trailing stop at this profit
        self.trailing_stop_distance = 0.01    # Trailing stop distance
        
        # Internal state
        self.position_entry_time = None  # Position entry time
        self.position_entry_price = None # Position entry price
        self.max_profit_reached = 0      # Maximum profit reached
        self.last_signal_time = None     # Last signal time
        self.last_signal = 0             # Last signal: 1(buy), -1(sell), 0(neutral)
    
    def _prepare_dataframe(self, klines):
        """
        Convert k-line data to DataFrame format
        
        Args:
            klines (list): K-line data list [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: Converted DataFrame
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K-line data is empty or insufficient")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Add datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to prepare DataFrame: {str(e)}")
            return None
    
    def _calculate_indicators(self, df):
        """
        Calculate Directional Movement indicators
        
        Args:
            df (pandas.DataFrame): K-line data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with calculated indicators
        """
        try:
            # Calculate ADX, +DI, -DI
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            df['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.di_period)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.di_period)
            
            # Calculate rate of change for ADX and DI
            df['adx_roc'] = (df['adx'] - df['adx'].shift(self.roc_period)) / self.roc_period
            df['plus_di_roc'] = (df['plus_di'] - df['plus_di'].shift(self.roc_period)) / self.roc_period
            df['minus_di_roc'] = (df['minus_di'] - df['minus_di'].shift(self.roc_period)) / self.roc_period
            
            # Calculate DI difference
            df['di_diff'] = df['plus_di'] - df['minus_di']
            df['di_diff_roc'] = (df['di_diff'] - df['di_diff'].shift(self.roc_period)) / self.roc_period
            
            # Clean NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def analyze_indicators(self, klines):
        """
        Analyze the directional movement indicators to generate trading signals
        
        Args:
            klines (list): K-line data
            
        Returns:
            dict: Analysis results including signal and trend information
        """
        try:
            # Prepare data
            df = self._prepare_dataframe(klines)
            if df is None or len(df) < 50:
                return {'signal': 0, 'trend': 0, 'strength': 0, 'message': 'Insufficient data'}
                
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Get the most recent indicator values
            current_adx = df['adx'].iloc[-1]
            current_plus_di = df['plus_di'].iloc[-1]
            current_minus_di = df['minus_di'].iloc[-1]
            current_di_diff = df['di_diff'].iloc[-1]
            
            # Get rate of change values
            adx_roc = df['adx_roc'].iloc[-1]
            plus_di_roc = df['plus_di_roc'].iloc[-1]
            minus_di_roc = df['minus_di_roc'].iloc[-1]
            di_diff_roc = df['di_diff_roc'].iloc[-1]
            
            # Identify trend status
            is_strong_trend = current_adx >= self.adx_strong_trend
            adx_rising = adx_roc > self.adx_rising_threshold
            
            # Determine trend direction
            trend_direction = 0
            if current_plus_di > current_minus_di + self.di_crossover_threshold:
                trend_direction = 1  # Up trend
            elif current_minus_di > current_plus_di + self.di_crossover_threshold:
                trend_direction = -1  # Down trend
            
            # Generate trading signal
            signal = 0  # Default to neutral
            message = ""
            
            # Up trend signals
            if trend_direction == 1:
                if is_strong_trend and adx_rising:
                    # Strong and strengthening uptrend
                    signal = 1
                    message = "Strong rising uptrend detected"
                elif is_strong_trend and plus_di_roc > self.di_change_threshold:
                    # Strong uptrend with increasing +DI
                    signal = 1
                    message = "Strong uptrend with increasing +DI"
                elif di_diff_roc > self.di_change_threshold:
                    # Increasing trend strength
                    signal = 1
                    message = "Increasing uptrend strength"
            
            # Down trend signals
            elif trend_direction == -1:
                if is_strong_trend and adx_rising:
                    # Strong and strengthening downtrend
                    signal = -1
                    message = "Strong rising downtrend detected"
                elif is_strong_trend and minus_di_roc > self.di_change_threshold:
                    # Strong downtrend with increasing -DI
                    signal = -1
                    message = "Strong downtrend with increasing -DI"
                elif di_diff_roc < -self.di_change_threshold:
                    # Increasing trend strength (negative direction)
                    signal = -1
                    message = "Increasing downtrend strength"
            
            # Early reversal signals - look for trend exhaustion or reversal
            if trend_direction == 1 and minus_di_roc > plus_di_roc * 2:
                # -DI rising faster than +DI in an uptrend - potential reversal
                if signal == 1:
                    # Just neutralize the signal if we detected an up signal
                    signal = 0
                    message = "Potential uptrend exhaustion, -DI rising faster than +DI"
                else:
                    # Otherwise, consider this a sell signal
                    signal = -1
                    message = "Uptrend reversal signal, -DI rising faster than +DI"
                    
            elif trend_direction == -1 and plus_di_roc > minus_di_roc * 2:
                # +DI rising faster than -DI in a downtrend - potential reversal
                if signal == -1:
                    # Just neutralize the signal if we detected a down signal
                    signal = 0
                    message = "Potential downtrend exhaustion, +DI rising faster than -DI"
                else:
                    # Otherwise, consider this a buy signal
                    signal = 1
                    message = "Downtrend reversal signal, +DI rising faster than -DI"
            
            # Log analysis results
            self.logger.info(f"ADX: {current_adx:.2f} (change: {adx_roc:.2f}), +DI: {current_plus_di:.2f} (change: {plus_di_roc:.2f}), " +
                          f"-DI: {current_minus_di:.2f} (change: {minus_di_roc:.2f}), Trend: {trend_direction}")
            if message:
                self.logger.info(f"Signal analysis: {message}")
            
            # Return analysis results
            result = {
                'signal': signal,
                'trend': trend_direction,
                'strength': float(current_adx),
                'adx': float(current_adx),
                'plus_di': float(current_plus_di),
                'minus_di': float(current_minus_di),
                'adx_roc': float(adx_roc),
                'plus_di_roc': float(plus_di_roc),
                'minus_di_roc': float(minus_di_roc),
                'di_diff': float(current_di_diff),
                'di_diff_roc': float(di_diff_roc),
                'message': message
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Indicator analysis failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'signal': 0, 'trend': 0, 'strength': 0, 'message': f'Error: {str(e)}'}
    
    def generate_signal(self, klines):
        """
        Generate trading signal
        
        Args:
            klines (list): K-line data
            
        Returns:
            int: Trading signal, 1(buy), -1(sell), 0(neutral)
        """
        try:
            # Get indicator analysis
            analysis = self.analyze_indicators(klines)
            signal = analysis['signal']
            
            # Check if we have a position already
            position = self.trader.get_position()
            position_side = None
            
            if position and 'info' in position:
                position_amount = float(position['info'].get('positionAmt', 0))
                if position_amount > 0:
                    position_side = "long"
                elif position_amount < 0:
                    position_side = "short"
            
            # Don't send opposite signals too quickly
            current_time = datetime.now()
            min_signal_interval = 60 * 15  # 15 minutes (one candle)
            
            if self.last_signal_time and self.last_signal != 0:
                time_since_last_signal = (current_time - self.last_signal_time).total_seconds()
                
                if time_since_last_signal < min_signal_interval:
                    # If last signal was opposite of current signal, be cautious
                    if self.last_signal * signal < 0:
                        self.logger.info(f"Signal reversed too quickly ({time_since_last_signal:.0f}s), staying neutral")
                        signal = 0
            
            # Update last signal information
            if signal != 0:
                self.last_signal_time = current_time
                self.last_signal = signal
            
            # Log final signal
            if signal == 1:
                self.logger.info("Final signal: BUY")
            elif signal == -1:
                self.logger.info("Final signal: SELL")
            else:
                self.logger.info("Final signal: NEUTRAL")
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to generate signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """Monitor current position and decide whether to close it based on strategy"""
        try:
            # Get current position
            position = self.trader.get_position()
            
            # If no position, check for new trading signal
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # Get latest k-line data
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
                # Generate trading signal
                signal = self.generate_signal(klines)
                
                # Get current market price
                current_price = self.trader.get_market_price()
                
                # Execute trade based on signal
                if signal == 1:  # Buy signal
                    # Calculate trade amount
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # Get trade amount percentage from config
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # Calculate trade amount
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # Open long position
                    self.trader.open_long(amount=trade_amount)
                    self.logger.info(f"Open long position - Amount: {trade_amount:.6f}, Price: {current_price}")
                    
                    # Record entry information
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    
                elif signal == -1:  # Sell signal
                    # Calculate trade amount
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # Get trade amount percentage from config
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # Calculate trade amount
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # Open short position
                    self.trader.open_short(amount=trade_amount)
                    self.logger.info(f"Open short position - Amount: {trade_amount:.6f}, Price: {current_price}")
                    
                    # Record entry information
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # If position exists, check if it should be closed
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                current_price = self.trader.get_market_price()
                position_side = "long" if position_amount > 0 else "short"
                
                # Calculate holding time
                current_time = time.time()
                if self.position_entry_time is not None:
                    holding_time_minutes = (current_time - self.position_entry_time) / 60
                    
                    # Check maximum position hold time
                    if holding_time_minutes >= self.max_position_hold_time:
                        self.logger.info(f"Position held for over {self.max_position_hold_time} minutes, closing")
                        self.trader.close_position()
                        return
                
                # Calculate profit rate
                if position_side == "long":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # Update maximum profit reached
                if profit_rate > self.max_profit_reached:
                    self.max_profit_reached = profit_rate
                
                # Check take profit
                if profit_rate >= self.take_profit_pct:
                    self.logger.info(f"Take profit condition met, profit rate: {profit_rate:.4%}, closing position")
                    self.trader.close_position()
                    return
                
                # Check stop loss
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"Stop loss condition met, loss rate: {profit_rate:.4%}, closing position")
                    self.trader.close_position()
                    return
                
                # Check trailing stop
                if self.trailing_stop and profit_rate >= self.trailing_stop_activation:
                    # Calculate drawdown percentage
                    drawdown = self.max_profit_reached - profit_rate
                    
                    # If drawdown exceeds trailing stop distance, close position for profit
                    if drawdown >= self.trailing_stop_distance:
                        self.logger.info(f"Trailing stop triggered, max profit: {self.max_profit_reached:.4%}, " +
                                      f"current profit: {profit_rate:.4%}, drawdown: {drawdown:.4%}")
                        self.trader.close_position()
                        return
                
                # Get latest trend signal to check if trend reversed
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                analysis = self.analyze_indicators(klines)
                signal = analysis['signal']
                
                # Close position if trend reversed significantly
                if position_side == "long" and signal == -1:
                    self.logger.info("Trend reversed to down, closing long position")
                    self.trader.close_position()
                    return
                elif position_side == "short" and signal == 1:
                    self.logger.info("Trend reversed to up, closing short position")
                    self.trader.close_position()
                    return
                
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
