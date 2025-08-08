"""
ADX DI Multi-Timeframe Trend Strategy - 5m & 15m Combined
Advanced trend prediction strategy using ADX and DI technical indicators across multiple timeframes
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
from strategies.base_strategy import BaseStrategy

class ADXDIMultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe ADX/DI trend prediction strategy combining 5m and 15m signals
    
    Features:
    1. Dual timeframe analysis (5m and 15m) with weighted signals
    2. Enhanced trend strength validation using ADX
    3. DI crossover detection for entry signals
    4. Trend momentum confirmation across timeframes
    5. Adaptive position sizing based on combined trend strength
    6. Multiple exit strategies (profit taking, stop loss, trend reversal)
    """
    
    def __init__(self, trader):
        """Initialize the multi-timeframe ADX/DI trend strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # Multi-timeframe configuration
        self.timeframes = {
            '5m': {
                'interval': '5m',
                'weight': 0.5,
                'check_interval': 120,  # Check every 2 minutes
                'lookback_period': 50
            },
            '15m': {
                'interval': '15m', 
                'weight': 0.5,
                'check_interval': 300,  # Check every 5 minutes
                'lookback_period': 50
            }
        }
        
        # Primary check interval (use the shorter one)
        self.check_interval = min([tf['check_interval'] for tf in self.timeframes.values()])
        self.training_lookback = 50
        
        # ADX/DI Parameters
        self.adx_period = 14
        self.di_period = 14
        self.adx_threshold = 20  # Minimum ADX for trend validity
        self.adx_strong_threshold = 30  # Strong trend threshold
        self.adx_very_strong_threshold = 45  # Very strong trend
        
        # DI Parameters
        self.di_diff_threshold = 2.0  # Minimum DI difference for signal
        self.di_crossover_confirmation = 2  # Bars to confirm crossover
        
        # Trend Prediction Parameters
        self.trend_momentum_periods = 3  # Periods to check trend momentum
        self.price_momentum_periods = 5  # Periods for price momentum
        self.volume_confirmation = True  # Use volume for confirmation
        
        # Position Management
        self.max_position_hold_time = 480  # 8 hours max hold
        self.position_size_base = 0.02  # Base position size (2% of balance)
        self.position_size_multiplier = 2.0  # Multiplier for strong trends
        
        # Risk Management
        self.stop_loss_pct = 0.015  # 1.5% stop loss
        self.take_profit_pct = 0.045  # 4.5% take profit
        self.trailing_stop = True
        self.trailing_stop_activation = 0.02  # Activate at 2% profit
        self.trailing_stop_distance = 0.008  # 0.8% trailing distance
        
        # Dynamic Risk Adjustment
        self.volatility_adjustment = True
        self.atr_period = 14
        self.volatility_multiplier = 1.5
        
        # Internal State
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trend_direction = 0  # 1=up, -1=down, 0=neutral
        self.trend_strength = 0
        self.last_signal = 0
        self.last_signal_time = None
        self.consecutive_signals = 0
        
        # Multi-timeframe state
        self.timeframe_signals = {}
        self.timeframe_analysis = {}
        
    def calculate_indicators(self, df):
        """
        Calculate all technical indicators
        
        Args:
            df (DataFrame): OHLCV data
            
        Returns:
            dict: Dictionary containing all indicators
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # ADX and DI indicators
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.di_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.di_period)
            
            # Additional trend indicators
            ema_short = talib.EMA(close, timeperiod=9)
            ema_long = talib.EMA(close, timeperiod=21)
            atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
            
            # Price momentum
            roc = talib.ROC(close, timeperiod=self.price_momentum_periods)
            
            # Volume indicators
            volume_sma = talib.SMA(volume, timeperiod=10)
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'ema_short': ema_short,
                'ema_long': ema_long,
                'atr': atr,
                'roc': roc,
                'volume_sma': volume_sma,
                'close': close,
                'volume': volume
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    def analyze_trend_strength(self, indicators):
        """
        Analyze trend strength and direction
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            dict: Trend analysis results
        """
        try:
            adx = indicators['adx'][-1]
            plus_di = indicators['plus_di'][-1]
            minus_di = indicators['minus_di'][-1]
            
            # Calculate DI difference
            di_diff = plus_di - minus_di
            
            # Determine trend direction
            if plus_di > minus_di:
                trend_direction = 1  # Uptrend
            elif minus_di > plus_di:
                trend_direction = -1  # Downtrend
            else:
                trend_direction = 0  # Neutral
            
            # Classify trend strength
            if adx >= self.adx_very_strong_threshold:
                strength_level = "very_strong"
                strength_score = 1.0
            elif adx >= self.adx_strong_threshold:
                strength_level = "strong"
                strength_score = 0.75
            elif adx >= self.adx_threshold:
                strength_level = "moderate"
                strength_score = 0.5
            else:
                strength_level = "weak"
                strength_score = 0.25
            
            # Check for DI crossovers
            prev_plus_di = indicators['plus_di'][-2]
            prev_minus_di = indicators['minus_di'][-2]
            
            crossover_up = (prev_plus_di <= prev_minus_di and plus_di > minus_di)
            crossover_down = (prev_plus_di >= prev_minus_di and plus_di < minus_di)
            
            return {
                'direction': trend_direction,
                'strength_level': strength_level,
                'strength_score': strength_score,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'di_diff': di_diff,
                'crossover_up': crossover_up,
                'crossover_down': crossover_down
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend strength: {str(e)}")
            return None
    
    def predict_trend_continuation(self, indicators):
        """
        Predict trend continuation using multiple factors
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            dict: Trend prediction results
        """
        try:
            # EMA trend confirmation
            ema_short = indicators['ema_short'][-1]
            ema_long = indicators['ema_long'][-1]
            ema_trend = 1 if ema_short > ema_long else -1
            
            # Price momentum
            roc = indicators['roc'][-1]
            momentum_strength = abs(roc) / 100  # Normalize
            
            # Volume confirmation
            current_volume = indicators['volume'][-1]
            avg_volume = indicators['volume_sma'][-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # ADX momentum (trend strengthening/weakening)
            adx_current = indicators['adx'][-1]
            adx_prev = indicators['adx'][-2]
            adx_momentum = 1 if adx_current > adx_prev else -1
            
            # Calculate prediction confidence
            confidence_factors = []
            
            # Factor 1: DI difference strength
            di_diff = abs(indicators['plus_di'][-1] - indicators['minus_di'][-1])
            confidence_factors.append(min(di_diff / 10, 1.0))
            
            # Factor 2: ADX strength
            confidence_factors.append(min(adx_current / 50, 1.0))
            
            # Factor 3: EMA alignment
            confidence_factors.append(0.8 if ema_trend != 0 else 0.2)
            
            # Factor 4: Volume confirmation
            if self.volume_confirmation:
                volume_factor = min(volume_ratio / 1.5, 1.0) if volume_ratio > 1 else 0.5
                confidence_factors.append(volume_factor)
            
            # Factor 5: Momentum consistency
            momentum_factor = min(momentum_strength * 10, 1.0)
            confidence_factors.append(momentum_factor)
            
            # Calculate overall confidence
            confidence = sum(confidence_factors) / len(confidence_factors)
            
            return {
                'ema_trend': ema_trend,
                'momentum_strength': momentum_strength,
                'volume_ratio': volume_ratio,
                'adx_momentum': adx_momentum,
                'confidence': confidence,
                'prediction_factors': {
                    'di_strength': confidence_factors[0],
                    'adx_strength': confidence_factors[1],
                    'ema_alignment': confidence_factors[2],
                    'volume_confirmation': confidence_factors[3] if self.volume_confirmation else None,
                    'momentum_consistency': confidence_factors[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting trend continuation: {str(e)}")
            return None
    
    def calculate_position_size(self, trend_analysis, prediction):
        """
        Calculate dynamic position size based on combined timeframe analysis
        
        Returns:
            float: Position size multiplier
        """
        try:
            base_size = self.position_size_base
            
            if not self.timeframe_analysis:
                return base_size
            
            # Calculate average strength and confidence across timeframes
            total_strength = 0
            total_confidence = 0
            count = 0
            
            for tf_name, analysis in self.timeframe_analysis.items():
                weight = self.timeframes[tf_name]['weight']
                total_strength += analysis['trend_analysis']['strength_score'] * weight
                total_confidence += analysis['prediction']['confidence'] * weight
                count += weight
            
            if count > 0:
                avg_strength = total_strength / count
                avg_confidence = total_confidence / count
            else:
                avg_strength = 0.5
                avg_confidence = 0.5
            
            # Calculate final position size
            final_size = base_size * avg_strength * avg_confidence
            
            # Apply maximum multiplier limit
            final_size = min(final_size, self.position_size_base * self.position_size_multiplier)
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.position_size_base
    
    def generate_signal(self, klines=None):
        """
        Generate trading signal based on multi-timeframe ADX/DI analysis
        
        Args:
            klines (dict): K-line data for different timeframes
            
        Returns:
            int: Trading signal (1=buy, -1=sell, 0=hold)
        """
        try:
            combined_signal = 0
            total_weight = 0
            weighted_signal_sum = 0
            
            # Analyze each timeframe
            for tf_name, tf_config in self.timeframes.items():
                # Get k-lines for this timeframe
                if klines is None or tf_name not in klines:
                    tf_klines = self.trader.get_klines(
                        symbol=self.trader.symbol,
                        interval=tf_config['interval'],
                        limit=tf_config['lookback_period']
                    )
                else:
                    tf_klines = klines[tf_name]
                
                if not tf_klines or len(tf_klines) < tf_config['lookback_period']:
                    self.logger.warning(f"Insufficient k-line data for {tf_name} timeframe")
                    continue
                
                # Prepare DataFrame
                df = self._prepare_dataframe(tf_klines)
                if df is None:
                    continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(df)
                if indicators is None:
                    continue
                
                # Analyze trend
                trend_analysis = self.analyze_trend_strength(indicators)
                if trend_analysis is None:
                    continue
                
                # Predict trend continuation
                prediction = self.predict_trend_continuation(indicators)
                if prediction is None:
                    continue
                
                # Store analysis for this timeframe
                self.timeframe_analysis[tf_name] = {
                    'trend_analysis': trend_analysis,
                    'prediction': prediction
                }
                
                # Generate signal for this timeframe
                tf_signal = self._generate_timeframe_signal(trend_analysis, prediction)
                self.timeframe_signals[tf_name] = tf_signal
                
                # Apply weight to signal
                weighted_signal = tf_signal * tf_config['weight']
                weighted_signal_sum += weighted_signal
                total_weight += tf_config['weight']
                
                # Log timeframe analysis
                self.logger.info(f"{tf_name} Analysis - ADX: {trend_analysis['adx']:.2f}, "
                               f"+DI: {trend_analysis['plus_di']:.2f}, "
                               f"-DI: {trend_analysis['minus_di']:.2f}, "
                               f"Signal: {tf_signal}, Weight: {tf_config['weight']}, "
                               f"Confidence: {prediction['confidence']:.3f}")
            
            # Calculate combined signal
            if total_weight > 0:
                weighted_average = weighted_signal_sum / total_weight
                
                # Convert weighted average to discrete signal
                if weighted_average >= 0.5:
                    combined_signal = 1
                elif weighted_average <= -0.5:
                    combined_signal = -1
                else:
                    combined_signal = 0
                
                self.logger.info(f"Combined Signal Analysis - Weighted Average: {weighted_average:.3f}, "
                               f"Final Signal: {combined_signal}")
            
            # Update signal tracking
            if combined_signal != 0:
                if combined_signal == self.last_signal:
                    self.consecutive_signals += 1
                else:
                    self.consecutive_signals = 1
                self.last_signal = combined_signal
                self.last_signal_time = time.time()
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error generating multi-timeframe signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _generate_timeframe_signal(self, trend_analysis, prediction):
        """
        Generate signal for a specific timeframe
        
        Args:
            trend_analysis (dict): Trend analysis results
            prediction (dict): Trend prediction results
            
        Returns:
            int: Signal for this timeframe (1=buy, -1=sell, 0=hold)
        """
        signal = 0
        
        # Check for strong trend with high confidence
        if (trend_analysis['strength_score'] >= 0.5 and 
            prediction['confidence'] >= 0.6 and
            abs(trend_analysis['di_diff']) >= self.di_diff_threshold):
            
            # Buy signal conditions
            if (trend_analysis['direction'] == 1 and
                (trend_analysis['crossover_up'] or 
                 (trend_analysis['plus_di'] > trend_analysis['minus_di'] and
                  prediction['ema_trend'] == 1))):
                signal = 1
                
            # Sell signal conditions
            elif (trend_analysis['direction'] == -1 and
                  (trend_analysis['crossover_down'] or
                   (trend_analysis['minus_di'] > trend_analysis['plus_di'] and
                    prediction['ema_trend'] == -1))):
                signal = -1
        
        return signal
    
    def calculate_position_size(self, trend_analysis=None, prediction=None):
        """
        Calculate dynamic position size based on combined timeframe analysis
        
        Returns:
            float: Position size multiplier
        """
        try:
            base_size = self.position_size_base
            
            if not self.timeframe_analysis:
                return base_size
            
            # Calculate average strength and confidence across timeframes
            total_strength = 0
            total_confidence = 0
            count = 0
            
            for tf_name, analysis in self.timeframe_analysis.items():
                weight = self.timeframes[tf_name]['weight']
                total_strength += analysis['trend_analysis']['strength_score'] * weight
                total_confidence += analysis['prediction']['confidence'] * weight
                count += weight
            
            if count > 0:
                avg_strength = total_strength / count
                avg_confidence = total_confidence / count
            else:
                avg_strength = 0.5
                avg_confidence = 0.5
            
            # Calculate final position size
            final_size = base_size * avg_strength * avg_confidence
            
            # Apply maximum multiplier limit
            final_size = min(final_size, self.position_size_base * self.position_size_multiplier)
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.position_size_base
    
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
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def monitor_position(self):
        """Monitor current position and manage trades"""
        try:
            position = self.trader.get_position()
            
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # No position - check for new signals
                signal = self.generate_signal()
                
                if signal != 0:
                    current_price = self.trader.get_market_price()
                    
                    # Get indicators for position sizing
                    klines = self.trader.get_klines(
                        symbol=self.trader.symbol,
                        interval=self.kline_interval,
                        limit=self.lookback_period
                    )
                    
                    if klines:
                        df = self._prepare_dataframe(klines)
                        if df is not None:
                            indicators = self.calculate_indicators(df)
                            if indicators is not None:
                                trend_analysis = self.analyze_trend_strength(indicators)
                                prediction = self.predict_trend_continuation(indicators)
                                
                                # Calculate position size
                                if trend_analysis and prediction:
                                    size_multiplier = self.calculate_position_size(trend_analysis, prediction)
                                else:
                                    size_multiplier = self.position_size_base
                            else:
                                size_multiplier = self.position_size_base
                        else:
                            size_multiplier = self.position_size_base
                    else:
                        size_multiplier = self.position_size_base
                    
                    # Calculate trade amount
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    trade_amount = (available_balance * size_multiplier) / current_price
                    
                    # Execute trade
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"Opened long position - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}, Size multiplier: {size_multiplier:.3f}")
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"Opened short position - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}, Size multiplier: {size_multiplier:.3f}")
                    
                    # Record entry
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            else:
                # Position exists - manage it
                self._manage_existing_position(position)
                
        except Exception as e:
            self.logger.error(f"Error monitoring position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_existing_position(self, position):
        """Manage existing position with risk controls"""
        try:
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_side = "long" if position_amount > 0 else "short"
            
            # Calculate profit/loss
            if position_side == "long":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price
            
            # Update max profit
            if profit_rate > self.max_profit_reached:
                self.max_profit_reached = profit_rate
            
            # Check time-based exit
            current_time = time.time()
            if self.position_entry_time is not None:
                holding_time_minutes = (current_time - self.position_entry_time) / 60
                if holding_time_minutes >= self.max_position_hold_time:
                    self.logger.info(f"Max hold time reached ({holding_time_minutes:.1f} min), closing position")
                    self.trader.close_position()
                    return
            
            # Check stop loss
            if profit_rate <= -self.stop_loss_pct:
                self.logger.info(f"Stop loss triggered: {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check take profit
            if profit_rate >= self.take_profit_pct:
                self.logger.info(f"Take profit triggered: {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check trailing stop
            if (self.trailing_stop and 
                profit_rate >= self.trailing_stop_activation and
                self.max_profit_reached > 0):
                
                drawdown = self.max_profit_reached - profit_rate
                if drawdown >= self.trailing_stop_distance:
                    self.logger.info(f"Trailing stop triggered - Max: {self.max_profit_reached:.3%}, "
                                   f"Current: {profit_rate:.3%}, Drawdown: {drawdown:.3%}")
                    self.trader.close_position()
                    return
            
            # Check for trend reversal exit
            signal = self.generate_signal()
            if ((position_side == "long" and signal == -1) or 
                (position_side == "short" and signal == 1)):
                self.logger.info(f"Trend reversal signal detected, closing {position_side} position")
                self.trader.close_position()
                return
                
        except Exception as e:
            self.logger.error(f"Error managing position: {str(e)}")