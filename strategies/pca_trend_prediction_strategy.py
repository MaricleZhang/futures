"""
PCA Trend Prediction Strategy for 15-minute timeframe
Uses Principal Component Analysis (PCA) to fuse multiple trend indicators for price prediction

File: strategies/pca_trend_prediction_strategy.py
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCATrendPredictionStrategy(BaseStrategy):
    """
    PCA Trend Prediction Strategy using 15-minute timeframe

    Strategy Logic:
    1. Calculates multiple trend indicators:
       - EMA (Fast & Slow) - Trend direction
       - MACD - Momentum and trend changes
       - ROC - Rate of change
       - ADX, +DI, -DI - Trend strength and direction
       - KAMA - Adaptive moving average
       - MOM - Momentum
       - RSI - Relative strength

    2. Uses PCA to reduce dimensionality and extract principal components
    3. Predicts price direction based on PC1 (first principal component)
    4. Uses PC2 for trend strength confirmation

    Trading Signals:
    - Buy: PC1 > threshold AND PC2 confirms strength
    - Sell: PC1 < -threshold AND PC2 confirms strength
    - Hold: Weak signal or conflicting components
    """

    def __init__(self, trader):
        """Initialize the PCA Trend Prediction strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()

        # Timeframe configuration
        self.kline_interval = '15m'  # 15-minute timeframe
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
        self.lookback_period = 200  # Number of candles for analysis (need more for PCA)
        self.training_lookback = 200  # For compatibility with TradingManager

        # PCA Configuration
        self.n_components = 3  # Number of principal components to use
        self.pca_training_window = 100  # Rolling window for PCA training
        self.pc1_threshold = 0.5  # Threshold for PC1 signal
        self.pc2_strength_threshold = 0.3  # Threshold for PC2 confirmation

        # Indicator Parameters
        # EMA Parameters
        self.ema_fast_period = 12
        self.ema_slow_period = 26

        # MACD Parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # ROC Parameters
        self.roc_period = 10

        # ADX Parameters
        self.adx_period = 14

        # KAMA Parameters
        self.kama_period = 30

        # MOM Parameters
        self.mom_period = 10

        # RSI Parameters
        self.rsi_period = 14

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

        # PCA model storage
        self.pca_model = None
        self.scaler = None

        self.logger.info("PCA Trend Prediction Strategy initialized for 15m timeframe")

    def calculate_indicators(self, df):
        """
        Calculate all trend indicators for PCA analysis

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
            volume = df['volume'].values

            # 1. EMA - Fast and Slow
            ema_fast = talib.EMA(close, timeperiod=self.ema_fast_period)
            ema_slow = talib.EMA(close, timeperiod=self.ema_slow_period)
            ema_diff = ema_fast - ema_slow  # EMA difference (trend direction)
            ema_diff_pct = (ema_diff / close) * 100  # Percentage difference

            # 2. MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            macd_pct = (macd / close) * 100  # Normalize MACD

            # 3. ROC - Rate of Change
            roc = talib.ROC(close, timeperiod=self.roc_period)

            # 4. ADX and DI - Trend Strength and Direction
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            di_diff = plus_di - minus_di  # DI difference (directional bias)

            # 5. KAMA - Kaufman's Adaptive Moving Average
            kama = talib.KAMA(close, timeperiod=self.kama_period)
            kama_diff = close - kama  # Distance from KAMA
            kama_diff_pct = (kama_diff / close) * 100

            # 6. MOM - Momentum
            mom = talib.MOM(close, timeperiod=self.mom_period)
            mom_pct = (mom / close) * 100  # Normalize momentum

            # 7. RSI - Relative Strength Index
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            rsi_normalized = (rsi - 50) / 50  # Normalize RSI to [-1, 1]

            # 8. Price momentum (simple change)
            price_change = np.zeros_like(close)
            price_change[1:] = (close[1:] - close[:-1]) / close[:-1] * 100

            return {
                'close': close,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'ema_diff_pct': ema_diff_pct,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'macd_pct': macd_pct,
                'roc': roc,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'di_diff': di_diff,
                'kama': kama,
                'kama_diff_pct': kama_diff_pct,
                'mom_pct': mom_pct,
                'rsi': rsi,
                'rsi_normalized': rsi_normalized,
                'price_change': price_change
            }

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def prepare_pca_features(self, indicators):
        """
        Prepare feature matrix for PCA analysis

        Args:
            indicators (dict): Dictionary of indicator values

        Returns:
            DataFrame: Feature matrix ready for PCA
        """
        try:
            # Select features for PCA
            # We choose normalized/percentage-based features to ensure proper scaling
            feature_names = [
                'ema_diff_pct',    # EMA trend direction
                'macd_pct',        # MACD momentum
                'roc',             # Rate of change
                'di_diff',         # Directional movement
                'kama_diff_pct',   # Distance from KAMA
                'mom_pct',         # Price momentum
                'rsi_normalized',  # RSI strength
                'price_change',    # Simple price change
                'adx'              # Trend strength
            ]

            # Create feature matrix
            feature_matrix = pd.DataFrame()
            for feature in feature_names:
                if feature in indicators:
                    feature_matrix[feature] = indicators[feature]

            # Remove rows with NaN values
            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
            feature_matrix = feature_matrix.ffill().fillna(0)

            return feature_matrix

        except Exception as e:
            self.logger.error(f"Error preparing PCA features: {str(e)}")
            return None

    def train_pca_model(self, feature_matrix):
        """
        Train PCA model on feature matrix

        Args:
            feature_matrix (DataFrame): Features for PCA

        Returns:
            tuple: (pca_model, scaler, transformed_data)
        """
        try:
            # Use rolling window for training
            train_window = min(self.pca_training_window, len(feature_matrix) - 20)
            train_data = feature_matrix.iloc[-train_window:].values

            # Standardize features (important for PCA)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(train_data)

            # Fit PCA model
            pca = PCA(n_components=self.n_components)
            transformed_data = pca.fit_transform(scaled_data)

            # Log PCA information
            explained_variance = pca.explained_variance_ratio_
            self.logger.info(f"PCA Explained Variance Ratio: {explained_variance}")
            self.logger.info(f"PC1: {explained_variance[0]*100:.2f}%, PC2: {explained_variance[1]*100:.2f}%, PC3: {explained_variance[2]*100:.2f}%")

            return pca, scaler, transformed_data

        except Exception as e:
            self.logger.error(f"Error training PCA model: {str(e)}")
            return None, None, None

    def analyze_pca_signals(self, pca_components, indicators):
        """
        Analyze PCA components to generate trading signals

        Args:
            pca_components (array): PCA transformed data
            indicators (dict): Original indicator values

        Returns:
            dict: Signal analysis results
        """
        try:
            # Get latest PCA component values
            pc1 = pca_components[-1, 0]  # First principal component (main trend)
            pc2 = pca_components[-1, 1]  # Second principal component (strength)
            pc3 = pca_components[-1, 2]  # Third principal component (additional info)

            # Calculate PC momentum (change in PC1)
            pc1_momentum = 0
            if len(pca_components) > 5:
                pc1_momentum = pc1 - pca_components[-5, 0]

            # Get current price and ADX for reference
            close = indicators['close'][-1]
            adx = indicators['adx'][-1]

            # Signal generation logic
            signal = 0
            confidence = 0
            reasons = []

            # BULLISH SIGNAL: PC1 positive and above threshold
            if pc1 > self.pc1_threshold:
                signal = 1
                confidence = min(100, abs(pc1) * 30 + abs(pc2) * 15)

                reasons.append(f"PC1 bullish: {pc1:.3f}")

                # PC2 confirmation (if PC2 is also positive, stronger signal)
                if pc2 > self.pc2_strength_threshold:
                    confidence += 15
                    reasons.append(f"PC2 confirms strength: {pc2:.3f}")

                # PC1 momentum confirmation
                if pc1_momentum > 0:
                    confidence += 10
                    reasons.append(f"PC1 momentum positive: {pc1_momentum:.3f}")

                # ADX confirmation
                if adx > 20:
                    confidence += 10
                    reasons.append(f"ADX confirms trend: {adx:.2f}")

            # BEARISH SIGNAL: PC1 negative and below threshold
            elif pc1 < -self.pc1_threshold:
                signal = -1
                confidence = min(100, abs(pc1) * 30 + abs(pc2) * 15)

                reasons.append(f"PC1 bearish: {pc1:.3f}")

                # PC2 confirmation
                if pc2 < -self.pc2_strength_threshold:
                    confidence += 15
                    reasons.append(f"PC2 confirms strength: {pc2:.3f}")

                # PC1 momentum confirmation
                if pc1_momentum < 0:
                    confidence += 10
                    reasons.append(f"PC1 momentum negative: {pc1_momentum:.3f}")

                # ADX confirmation
                if adx > 20:
                    confidence += 10
                    reasons.append(f"ADX confirms trend: {adx:.2f}")

            # NO CLEAR SIGNAL
            else:
                reasons.append(f"PC1 weak signal: {pc1:.3f}")
                if abs(pc1) < self.pc1_threshold:
                    reasons.append(f"PC1 below threshold ({self.pc1_threshold})")
                if adx < 20:
                    reasons.append(f"ADX too low: {adx:.2f}")

            return {
                'signal': signal,
                'confidence': confidence,
                'pc1': pc1,
                'pc2': pc2,
                'pc3': pc3,
                'pc1_momentum': pc1_momentum,
                'adx': adx,
                'close': close,
                'reasons': ' | '.join(reasons)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing PCA signals: {str(e)}")
            return {
                'signal': 0,
                'confidence': 0,
                'reasons': f"Error: {str(e)}"
            }

    def generate_signal(self, klines=None):
        """
        Generate trading signal based on PCA analysis

        Args:
            klines: K-line data (if None, will fetch from trader)

        Returns:
            int: Signal value (1=buy, -1=sell, 0=hold)
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

            # Calculate all indicators
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return 0

            # Prepare features for PCA
            feature_matrix = self.prepare_pca_features(indicators)
            if feature_matrix is None or len(feature_matrix) < 50:
                self.logger.warning("Insufficient data for PCA analysis")
                return 0

            # Train PCA model
            pca_model, scaler, pca_components = self.train_pca_model(feature_matrix)
            if pca_model is None:
                return 0

            # Store models for future use
            self.pca_model = pca_model
            self.scaler = scaler

            # Analyze PCA signals
            analysis = self.analyze_pca_signals(pca_components, indicators)

            # Log analysis
            self.logger.info(f"PCA Trend Prediction Analysis:")
            self.logger.info(f"  Signal: {analysis['signal']} | Confidence: {analysis['confidence']:.1f}%")
            self.logger.info(f"  PC1: {analysis['pc1']:.3f} | PC2: {analysis['pc2']:.3f} | PC3: {analysis['pc3']:.3f}")
            self.logger.info(f"  PC1 Momentum: {analysis['pc1_momentum']:.3f}")
            self.logger.info(f"  Close: {analysis['close']:.4f} | ADX: {analysis['adx']:.2f}")
            self.logger.info(f"  Reasons: {analysis['reasons']}")

            # Update last signal
            if analysis['signal'] != 0:
                self.last_signal = analysis['signal']
                self.last_signal_time = datetime.now()

            return analysis['signal']

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
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
