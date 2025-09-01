"""
Simple ADX-DI Strategy for 15-minute timeframe
Uses only ADX and DI indicators to predict price trends

File: strategies/simple_adx_di_strategy_15m.py
"""
import re
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy

class SimpleADXDIStrategy15m(BaseStrategy):
    """
    Simple ADX-DI trend prediction strategy using 15-minute timeframe
    
    Strategy Logic:
    1. ADX (Average Directional Index) measures trend strength
    2. +DI (Positive Directional Indicator) measures upward price movement
    3. -DI (Negative Directional Indicator) measures downward price movement
    
    Trading Signals:
    - Buy: +DI > -DI with ADX above threshold (strong uptrend)
    - Sell: -DI > +DI with ADX above threshold (strong downtrend)
    - Hold: ADX below threshold (weak or no trend)
    """
    
    def __init__(self, trader):
        """Initialize the Simple ADX-DI strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # Timeframe configuration
        self.kline_interval = '15m'  # 15-minute timeframe
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
        self.lookback_period = 50  # Number of candles for analysis
        self.training_lookback = 50  # For compatibility with TradingManager
        
        # ADX and DI Parameters
        self.adx_period = 14  # Standard ADX period
        self.di_period = 14   # Standard DI period
        
        # Trend Thresholds
        self.adx_min_threshold = 18  # Minimum ADX for trend confirmation
        self.adx_strong_threshold = 40  # Strong trend threshold
        self.di_diff_threshold = 12  # Minimum difference between +DI and -DI
        
        # Position Management
        self.max_position_hold_time = 720  # 12 hours maximum hold time
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        # Trailing Stop Configuration
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.03  # Activate at 3% profit
        self.trailing_stop_distance = 0.01  # 1% trailing distance
        
        # Position Tracking
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_signal = 0
        self.last_signal_time = None
        
        self.logger.info("Simple ADX-DI Strategy initialized for 15m timeframe")
        
    def calculate_adx_di_indicators(self, df):
        """
        Calculate ADX and DI indicators
        
        Args:
            df (DataFrame): OHLCV data
            
        Returns:
            dict: Dictionary containing ADX, +DI, and -DI values
        """
        try:
            # Extract price data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate indicators
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.di_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.di_period)
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX/DI indicators: {str(e)}")
            return None
    
    def analyze_trend(self, indicators):
        """
        Analyze trend based on ADX and DI indicators
        
        Args:
            indicators (dict): ADX and DI indicator values
            
        Returns:
            dict: Trend analysis results
        """
        try:
            # Get current values (last candle)
            current_adx = indicators['adx'][-1]
            current_plus_di = indicators['plus_di'][-1]
            current_minus_di = indicators['minus_di'][-1]
            
            # Get previous values for trend momentum
            prev_adx = indicators['adx'][-2]
            prev_plus_di = indicators['plus_di'][-2]
            prev_minus_di = indicators['minus_di'][-2]
            
            # Calculate DI difference
            di_diff = current_plus_di - current_minus_di
            
            # Determine trend direction
            if current_plus_di > current_minus_di:
                trend_direction = 1  # Uptrend
            elif current_minus_di > current_plus_di:
                trend_direction = -1  # Downtrend
            else:
                trend_direction = 0  # Neutral
            
            # Determine trend strength
            if current_adx >= self.adx_strong_threshold:
                trend_strength = "strong"
            elif current_adx >= self.adx_min_threshold:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            # Check for DI crossovers (potential trend reversal)
            bullish_crossover = (prev_plus_di <= prev_minus_di and 
                                current_plus_di > current_minus_di)
            bearish_crossover = (prev_plus_di >= prev_minus_di and 
                                current_plus_di < current_minus_di)
            
            # ADX trend momentum (increasing or decreasing)
            adx_momentum = "increasing" if current_adx > prev_adx else "decreasing"
            
            # Log the analysis
            self.logger.info(f"Trend Analysis - ADX: {current_adx:.2f} ({adx_momentum}), "
                           f"+DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}, "
                           f"Direction: {trend_direction}, Strength: {trend_strength}")
            
            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'adx_value': current_adx,
                'plus_di_value': current_plus_di,
                'minus_di_value': current_minus_di,
                'di_difference': di_diff,
                'bullish_crossover': bullish_crossover,
                'bearish_crossover': bearish_crossover,
                'adx_momentum': adx_momentum
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return None
    
    def generate_signal(self, klines=None):
        """
        Generate trading signal based on ADX-DI analysis
        
        Args:
            klines (list): K-line data (optional, will fetch if not provided)
            
        Returns:
            int: Trading signal (1=buy, -1=sell, 0=hold)
        """
        try:
            # Fetch k-lines if not provided
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            # Check if we have enough data
            if not klines or len(klines) < self.lookback_period:
                self.logger.warning("Insufficient k-line data for analysis")
                return 0
            
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0
            
            # Calculate indicators
            indicators = self.calculate_adx_di_indicators(df)
            if indicators is None:
                return 0
            
            # Analyze trend
            trend_analysis = self.analyze_trend(indicators)
            if trend_analysis is None:
                return 0
            
            # Generate signal based on trend analysis
            signal = 0
            if (abs(trend_analysis['di_difference']) >= self.di_diff_threshold and
                trend_analysis['adx_value'] >= self.adx_min_threshold):
                    signal = trend_analysis['direction']
            else:
                signal = 0

            return signal

            # Strong uptrend signal
            if (trend_analysis['direction'] == 1 and 
                trend_analysis['adx_value'] >= self.adx_min_threshold and
                abs(trend_analysis['di_difference']) >= self.di_diff_threshold):
                
                # Extra confirmation for strong signals
                if (trend_analysis['strength'] == "strong" or 
                    trend_analysis['bullish_crossover'] or
                    trend_analysis['adx_momentum'] == "increasing"):
                    signal = 1
                    self.logger.info("BUY signal generated - Strong uptrend detected")
            
            # Strong downtrend signal
            elif (trend_analysis['direction'] == -1 and 
                  trend_analysis['adx_value'] >= self.adx_min_threshold and
                  abs(trend_analysis['di_difference']) >= self.di_diff_threshold):
                
                # Extra confirmation for strong signals
                if (trend_analysis['strength'] == "strong" or 
                    trend_analysis['bearish_crossover'] or
                    trend_analysis['adx_momentum'] == "increasing"):
                    signal = -1
                    self.logger.info("SELL signal generated - Strong downtrend detected")
            
            # No clear trend or weak trend
            else:
                signal = 0
                if trend_analysis['strength'] == "weak":
                    self.logger.info("HOLD signal - Weak trend, waiting for stronger signals")
                else:
                    self.logger.info("HOLD signal - No clear trend direction")
            
            # Update signal tracking
            if signal != 0:
                self.last_signal = signal
                self.last_signal_time = time.time()
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _prepare_dataframe(self, klines):
        """
        Convert k-line data to DataFrame
        
        Args:
            klines (list): K-line data
            
        Returns:
            DataFrame: Processed data
        """
        try:
            if not klines or len(klines) < 30:
                self.logger.error("Insufficient k-line data")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def monitor_position(self):
        """Monitor current position and execute trading logic"""
        try:
            # Get current position
            position = self.trader.get_position()
            
            # No position - check for entry signals
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # Generate trading signal
                signal = self.generate_signal()
                
                if signal != 0:
                    # Get account balance
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # Get current price
                    current_price = self.trader.get_market_price()
                    
                    # Calculate trade amount (using config percentage)
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # Execute trade
                    if signal == 1:  # Buy signal
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"LONG position opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    elif signal == -1:  # Sell signal
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"SHORT position opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    
                    # Record entry details
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # Position exists - manage it
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"Error monitoring position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """
        Manage existing position with risk controls
        
        Args:
            position: Current position information
        """
        try:
            # Extract position details
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_side = "long" if position_amount > 0 else "short"
            
            # Calculate profit/loss percentage
            if position_side == "long":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price
            
            # Update maximum profit reached
            if profit_rate > self.max_profit_reached:
                self.max_profit_reached = profit_rate
                self.logger.debug(f"New max profit: {self.max_profit_reached:.3%}")
            
            # Check holding time
            if self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60  # minutes
                if holding_time >= self.max_position_hold_time:
                    self.logger.info(f"Maximum holding time reached ({holding_time:.1f} min), closing position")
                    self.trader.close_position()
                    return
            
            # Check stop loss
            if profit_rate <= -self.stop_loss_pct:
                self.logger.info(f"STOP LOSS triggered at {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check take profit
            if profit_rate >= self.take_profit_pct:
                self.logger.info(f"TAKE PROFIT triggered at {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check trailing stop
            if self.trailing_stop_enabled and profit_rate >= self.trailing_stop_activation:
                drawdown = self.max_profit_reached - profit_rate
                if drawdown >= self.trailing_stop_distance:
                    self.logger.info(f"TRAILING STOP triggered - Max profit: {self.max_profit_reached:.3%}, "
                                   f"Current: {profit_rate:.3%}, Drawdown: {drawdown:.3%}")
                    self.trader.close_position()
                    return
            
            # Check for trend reversal
            signal = self.generate_signal()
            if (position_side == "long" and signal == -1) or (position_side == "short" and signal == 1):
                self.logger.info(f"TREND REVERSAL detected, closing {position_side} position")
                self.trader.close_position()
                return
            
            # Log current position status
            self.logger.debug(f"Position status - Side: {position_side}, P/L: {profit_rate:.3%}, "
                            f"Price: {current_price}, Entry: {entry_price}")
                
        except Exception as e:
            self.logger.error(f"Error managing position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
