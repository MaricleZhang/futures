"""
Multi-Timeframe DI-ADX Strategy
Uses Directional Movement Index (DI) and Average Directional Index (ADX)
across multiple timeframes (1m, 3m, 5m, 15m) to predict trend direction and strength.
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
from strategies.base_strategy import BaseStrategy

class MultiTimeframeDIADXStrategy(BaseStrategy):
    """MultiTimeframeDIADXStrategy - Multi-timeframe trend prediction strategy using DI and ADX
    
    Analyzes market trends using DI+ (positive directional indicator), DI- (negative directional indicator), 
    and ADX (average directional index) across multiple timeframes (1m, 3m, 5m, 15m).
    
    Features:
    1. Multi-timeframe analysis to filter out noise and confirm trends
    2. Trend strength confirmation using ADX
    3. Weighing system that gives more importance to longer timeframes
    4. Cross-timeframe trend confirmation to reduce false signals
    5. Adaptive parameters based on market volatility
    """
    
    def __init__(self, trader):
        """Initialize the Multi-timeframe DI-ADX strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # Timeframes to analyze
        self.timeframes = ['1m', '3m', '5m', '15m']
        self.timeframe_weights = {
            '1m': 0.1,    # Lowest weight for shortest timeframe
            '3m': 0.2,
            '5m': 0.3,
            '15m': 0.4    # Highest weight for longest timeframe
        }
        
        # Primary timeframe for monitoring
        self.primary_timeframe = '5m'
        self.check_interval = 60  # Check signal interval (seconds)
        self.lookback_period = 100  # Number of k-lines for indicators
        self.training_lookback = 100  # For compatibility with TradingManager
        
        # Indicator parameters
        self.adx_period = 14  # ADX period
        self.di_period = 14   # DI period
        self.adx_threshold = 25  # ADX threshold for trend strength
        self.adx_strong_threshold = 40  # Strong trend threshold
        
        # Trend consensus parameters
        self.min_consensus_threshold = 0.7  # Minimum consensus threshold for valid signal
        self.min_signal_strength = 0.2      # Minimum signal strength required
        
        # Position management parameters
        self.max_position_hold_time = 240  # Maximum position hold time (minutes)
        self.stop_loss_pct = 0.02  # Stop loss percentage (2%)
        self.take_profit_pct = 0.04  # Take profit percentage (4%)
        self.trailing_stop = True  # Enable trailing stop
        self.trailing_stop_activation = 0.015  # Activate trailing stop at 1.5% profit
        self.trailing_stop_distance = 0.008  # Trailing stop distance (0.8%)
        
        # Market state tracking
        self.market_state = "unknown"  # Market state (trend, range, unknown)
        self.current_trend = 0  # Current trend: 1(up), -1(down), 0(neutral)
        self.trend_strength = 0  # Trend strength (0-100)
        
        # Internal state
        self.position_entry_time = None  # Position entry time
        self.position_entry_price = None  # Position entry price
        self.max_profit_reached = 0  # Maximum profit reached
        self.last_signal = 0  # Last signal: 1(buy), -1(sell), 0(neutral)
        self.last_signal_time = None  # Last signal time
        
    def analyze_timeframe(self, klines, tf):
        """
        Analyze a specific timeframe
        
        Args:
            klines (list): K-line data for the timeframe
            tf (str): Timeframe being analyzed
            
        Returns:
            dict: Analysis results for the timeframe
        """
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None or len(df) < 30:  # Need enough data for reliable analysis
                return {"trend": 0, "strength": 0, "di_diff": 0, "valid": False}
            
            # Calculate DI and ADX indicators
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.di_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.di_period)
            
            # Get latest values
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # Calculate DI differential
            di_diff = current_plus_di - current_minus_di
            
            # Determine trend direction
            trend = 0
            if current_plus_di > current_minus_di:
                trend = 1  # Uptrend
            elif current_minus_di > current_plus_di:
                trend = -1  # Downtrend
                
            # Calculate signal strength (normalized)
            di_strength = abs(di_diff) / (current_plus_di + current_minus_di) if (current_plus_di + current_minus_di) > 0 else 0
            
            # Determine if trend is strong enough
            is_strong_trend = current_adx > self.adx_threshold
            
            # Get previous values to detect crossovers
            prev_plus_di = plus_di[-2]
            prev_minus_di = minus_di[-2]
            
            # Detect crossovers
            crossover_up = prev_plus_di < prev_minus_di and current_plus_di >= current_minus_di
            crossover_down = prev_plus_di > prev_minus_di and current_plus_di <= current_minus_di
            
            # Log the results
            self.logger.info(f"{tf} Analysis: ADX={current_adx:.2f}, +DI={current_plus_di:.2f}, -DI={current_minus_di:.2f}, " +
                          f"Trend={trend}, Strength={di_strength:.2f}, Strong={is_strong_trend}")
            
            return {
                "timeframe": tf,
                "trend": trend,
                "adx": float(current_adx),
                "plus_di": float(current_plus_di),
                "minus_di": float(current_minus_di),
                "di_diff": float(di_diff),
                "strength": float(di_strength),
                "is_strong": is_strong_trend,
                "crossover_up": crossover_up,
                "crossover_down": crossover_down,
                "valid": True
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {tf} timeframe: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"trend": 0, "strength": 0, "di_diff": 0, "valid": False}
            
    def _prepare_dataframe(self, klines):
        """
        Convert k-line data to DataFrame format
        
        Args:
            klines (list): K-line data list [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: Converted DataFrame
        """
        try:
            if not isinstance(klines, list) or len(klines) < 30:
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
            
    def analyze_multi_timeframe(self):
        """
        Analyze all timeframes and generate a combined signal
        
        Returns:
            dict: Combined analysis results
        """
        try:
            # Store all timeframe results
            tf_results = {}
            valid_results = 0
            
            # Analyze each timeframe
            for tf in self.timeframes:
                # Get k-lines for this timeframe
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=tf,
                    limit=self.lookback_period
                )
                
                if klines and len(klines) > 0:
                    # Analyze this timeframe
                    result = self.analyze_timeframe(klines, tf)
                    tf_results[tf] = result
                    
                    if result["valid"]:
                        valid_results += 1
                else:
                    self.logger.error(f"Failed to get k-lines for {tf} timeframe")
            
            # Check if we have enough valid results
            if valid_results < len(self.timeframes) / 2:  # At least half of timeframes must be valid
                self.logger.warning(f"Not enough valid timeframe analyses: {valid_results}/{len(self.timeframes)}")
                return {"trend": 0, "strength": 0, "confidence": 0}
            
            # Calculate weighted trend score
            trend_score = 0
            strength_score = 0
            total_weight = 0
            
            for tf, result in tf_results.items():
                if result["valid"]:
                    weight = self.timeframe_weights.get(tf, 0.25)  # Default weight if not specified
                    trend_score += result["trend"] * weight
                    strength_score += (result["strength"] * result["adx"] / 100) * weight
                    total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                trend_score /= total_weight
                strength_score /= total_weight
            
            # Calculate trend consensus
            uptrend_consensus = 0
            downtrend_consensus = 0
            
            for tf, result in tf_results.items():
                if result["valid"]:
                    if result["trend"] > 0:
                        uptrend_consensus += self.timeframe_weights.get(tf, 0.25)
                    elif result["trend"] < 0:
                        downtrend_consensus += self.timeframe_weights.get(tf, 0.25)
            
            # Normalize consensus scores
            uptrend_consensus /= total_weight if total_weight > 0 else 1
            downtrend_consensus /= total_weight if total_weight > 0 else 1
            
            # Determine final trend based on consensus
            final_trend = 0
            confidence = 0
            
            if uptrend_consensus > self.min_consensus_threshold:
                final_trend = 1
                confidence = uptrend_consensus
            elif downtrend_consensus > self.min_consensus_threshold:
                final_trend = -1
                confidence = downtrend_consensus
            else:
                # If no clear consensus, use the weighted trend score
                if abs(trend_score) > self.min_signal_strength:
                    final_trend = 1 if trend_score > 0 else -1
                    confidence = abs(trend_score)
            
            # Check for trend crossovers (higher priority)
            crossover_up_count = sum(1 for result in tf_results.values() if result.get("valid") and result.get("crossover_up"))
            crossover_down_count = sum(1 for result in tf_results.values() if result.get("valid") and result.get("crossover_down"))
            
            # Stronger signal if multiple timeframes show crossover
            if crossover_up_count >= 2:
                final_trend = 1
                confidence = max(confidence, 0.7)  # Minimum 70% confidence for multiple crossovers
            elif crossover_down_count >= 2:
                final_trend = -1
                confidence = max(confidence, 0.7)
            
            # Calculate average ADX for market state determination
            avg_adx = sum(result["adx"] for tf, result in tf_results.items() if result["valid"]) / valid_results
            
            # Determine market state
            if avg_adx > self.adx_strong_threshold:
                market_state = "strong_trend"
            elif avg_adx > self.adx_threshold:
                market_state = "trend"
            else:
                market_state = "range"
            
            # Update internal state
            self.market_state = market_state
            self.current_trend = final_trend
            self.trend_strength = avg_adx
            
            # Log the results
            self.logger.info(f"Multi-Timeframe Analysis: Trend={final_trend}, Strength={strength_score:.4f}, " +
                          f"Confidence={confidence:.4f}, Market State={market_state}, ADX={avg_adx:.2f}")
            
            return {
                "trend": final_trend,
                "strength": float(strength_score),
                "confidence": float(confidence),
                "market_state": market_state,
                "adx": float(avg_adx),
                "timeframes": tf_results
            }
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"trend": 0, "strength": 0, "confidence": 0}
    
    def generate_signal(self, klines=None):
        """
        Generate trading signal
        
        Args:
            klines (list, optional): K-line data (not used in this strategy as we use multiple timeframes)
            
        Returns:
            int: Trading signal, 1(buy), -1(sell), 0(neutral)
        """
        try:
            # Analyze all timeframes
            analysis = self.analyze_multi_timeframe()
            
            # Get current position
            position = self.trader.get_position()
            position_side = None
            
            if position and 'info' in position:
                position_amount = float(position['info'].get('positionAmt', 0))
                if position_amount > 0:
                    position_side = "long"
                elif position_amount < 0:
                    position_side = "short"
            
            # Determine signal based on analysis
            signal = 0
            
            # Check if trend and confidence are strong enough
            if analysis["trend"] != 0 and analysis["confidence"] >= self.min_consensus_threshold:
                signal = analysis["trend"]
                
                # Record the time of signal
                self.last_signal_time = time.time()
                self.last_signal = signal
            
            # Special case for market exit
            # If there's a strong counter-trend signal and we're in a position
            if position_side == "long" and analysis["trend"] == -1 and analysis["confidence"] > 0.8:
                signal = -1  # Exit long position
                self.logger.info("Strong counter-trend signal, recommending exit of long position")
            elif position_side == "short" and analysis["trend"] == 1 and analysis["confidence"] > 0.8:
                signal = 1  # Exit short position
                self.logger.info("Strong counter-trend signal, recommending exit of short position")
            
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
                # Generate trading signal (which will analyze all timeframes)
                signal = self.generate_signal()
                
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
                
                # Check for exit signal based on multi-timeframe analysis
                analysis = self.analyze_multi_timeframe()
                
                # Exit if trend has clearly reversed with strong confidence
                if position_side == "long" and analysis["trend"] == -1 and analysis["confidence"] > 0.8:
                    self.logger.info(f"Strong trend reversal detected, closing long position")
                    self.trader.close_position()
                    return
                elif position_side == "short" and analysis["trend"] == 1 and analysis["confidence"] > 0.8:
                    self.logger.info(f"Strong trend reversal detected, closing short position")
                    self.trader.close_position()
                    return
                
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
