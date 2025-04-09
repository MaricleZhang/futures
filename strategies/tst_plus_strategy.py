"""
TST+ (Time Series Transformer Plus) Strategy - 30 minute timeframe
Advanced transformer-based model for cryptocurrency price prediction
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from strategies.base_strategy import BaseStrategy

class TSTPlus30mStrategy(BaseStrategy):
    """TSTPlus30mStrategy - Time Series Transformer Plus Strategy
    
    A transformer-based deep learning strategy for 30-minute timeframe trading.
    Leverages advanced attention mechanisms to capture complex temporal patterns
    and dependencies in cryptocurrency price movements.
    
    Features:
    1. Transformer architecture with self-attention for sequence modeling
    2. Multi-head attention to capture different aspects of time series data
    3. Feature-wise attention to focus on the most relevant technical indicators
    4. Adaptive learning rate and regularization techniques
    5. Advanced risk management with dynamic stop-loss and take-profit
    """
    
    def __init__(self, trader):
        """Initialize the TST+ strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K-line settings
        self.kline_interval = '15m'      # 30-minute timeframe
        self.check_interval = 300        # Check signal interval (seconds)
        self.lookback_period = 500       # Number of k-lines for indicators
        self.training_lookback = 1500    # Number of k-lines for model training
        self.retraining_interval = 86400 # Retraining interval (seconds)
        
        # Model parameters
        self.sequence_length = 60        # Input sequence length
        self.feature_count = 20          # Number of features
        self.d_model = 64                # Transformer model dimension
        self.num_heads = 4               # Number of attention heads
        self.dropout_rate = 0.2          # Dropout rate
        self.num_layers = 3              # Number of transformer layers
        self.model = None                # The model instance
        self.model_path = self._get_model_path()  # Model save path
        self.scaler = None               # Feature scaler
        self.last_training_time = 0      # Last training time
        self.min_confidence = 0.6        # Minimum prediction confidence
        
        # Trend parameters
        self.trend_confirmation_window = 3  # Trend confirmation window
        self.market_state = "unknown"    # Market state
        self.trend_strength = 0          # Trend strength
        self.current_trend = 0           # Current trend: 1(up), -1(down), 0(neutral)
        
        # Position control parameters
        self.max_position_hold_time = 720  # Maximum position hold time (minutes)
        self.stop_loss_pct = 0.03        # Stop loss percentage (3%)
        self.take_profit_pct = 0.05      # Take profit percentage (5%)
        self.trailing_stop = True        # Enable trailing stop
        self.trailing_stop_activation = 0.02  # Activate trailing stop at this profit
        self.trailing_stop_distance = 0.01    # Trailing stop distance
        
        # Signal history
        self.signal_history = []         # Recent signal history
        self.max_signal_history = 20     # Maximum number of signals to store
        self.last_signal_time = None     # Last signal time
        self.last_signal = 0             # Last signal: 1(buy), -1(sell), 0(neutral)
        
        # Internal state
        self.position_entry_time = None  # Position entry time
        self.position_entry_price = None # Position entry price
        self.max_profit_reached = 0      # Maximum profit reached
        
        # Initialize
        self._initialize_model()
        
    def _get_model_path(self):
        """Get model save path"""
        # Create model directory
        model_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Create specific model path for this trading pair
        model_path = os.path.join(model_dir, f'tst_plus_model_{self.trader.symbol}.h5')
        scaler_path = os.path.join(model_dir, f'tst_plus_scaler_{self.trader.symbol}.pkl')
        
        return {
            'model': model_path,
            'scaler': scaler_path
        }
        
    def _initialize_model(self):
        """Initialize or load the transformer model"""
        try:
            # Check if a pre-trained model exists
            if os.path.exists(self.model_path['model']):
                self.logger.info(f"Loading existing model: {self.model_path['model']}")
                self.model = tf.keras.models.load_model(self.model_path['model'])
                
                # Load feature scaler
                if os.path.exists(self.model_path['scaler']):
                    with open(self.model_path['scaler'], 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                # Create new model
                self.logger.info("No pre-trained model found, creating new model")
                self._build_model()
                
                # Immediately train the model
                self._initial_training()
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Create new model as backup
            self._build_model()
    
    def _build_model(self):
        """Build the TST+ model using TensorFlow's transformer components"""
        try:
            # Set random seed for reproducibility
            tf.random.set_seed(42)
            
            # Input layer
            inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.feature_count))
            
            # Dense projection to d_model
            x = tf.keras.layers.Dense(self.d_model)(inputs)
            
            # Transformer Encoder blocks
            for _ in range(self.num_layers):
                # Multi-head attention
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.d_model // self.num_heads
                )(x, x)
                
                # Add & Norm
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
                
                # Feed Forward
                ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
                    tf.keras.layers.Dense(self.d_model)
                ])
                
                # Add & Norm
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
            
            # Global average pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Final layers
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
            # Output: 3 classes (up, neutral, down)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.logger.info("TST+ model built successfully")
            
            # Create feature scaler
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _initial_training(self):
        """Initial model training"""
        try:
            self.logger.info("Starting initial model training...")
            
            # Get historical k-line data
            klines = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if klines and len(klines) > 0:
                self.logger.info(f"Successfully got {len(klines)} k-lines for training")
                
                # Prepare training data
                X, y = self._prepare_training_data(klines)
                
                if X is not None and y is not None and len(X) > 0:
                    # Train model
                    self.logger.info(f"Starting model training, sample size: {len(X)}")
                    
                    # Batch training to avoid memory issues
                    history = self.model.fit(
                        X, y,
                        epochs=20,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True
                            )
                        ],
                        verbose=1
                    )
                    
                    # Save model
                    self.model.save(self.model_path['model'])
                    
                    # Save scaler
                    with open(self.model_path['scaler'], 'wb') as f:
                        pickle.dump(self.scaler, f)
                    
                    self.logger.info("Model training completed and saved")
                    
                    # Record training time
                    self.last_training_time = time.time()
                else:
                    self.logger.error("Failed to prepare training data")
            else:
                self.logger.error("Failed to get training data")
                
        except Exception as e:
            self.logger.error(f"Initial model training failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
    
    def _engineer_features(self, df):
        """
        Feature engineering - Create and calculate model input features
        
        Args:
            df (pandas.DataFrame): K-line data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with calculated features
        """
        try:
            # Copy original data
            features_df = df.copy()
            
            # 1. Price features
            # Normalize price (using past N k-lines' high and low)
            window = 20
            features_df['price_norm'] = (features_df['close'] - features_df['low'].rolling(window=window).min()) / \
                              (features_df['high'].rolling(window=window).max() - features_df['low'].rolling(window=window).min())
            
            # Close-Open ratio
            features_df['close_open_ratio'] = features_df['close'] / features_df['open'] - 1
            
            # High-Close ratio
            features_df['high_close_ratio'] = features_df['high'] / features_df['close'] - 1
            
            # Low-Close ratio
            features_df['low_close_ratio'] = features_df['low'] / features_df['close'] - 1
            
            # 2. Price changes
            # Calculate price change rates for different periods
            for period in [1, 3, 5, 10, 20]:
                features_df[f'return_{period}'] = features_df['close'].pct_change(periods=period)
            
            # 3. Technical indicators
            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                features_df[f'ma_{period}'] = talib.SMA(features_df['close'].values, timeperiod=period)
                features_df[f'ma_ratio_{period}'] = features_df['close'] / features_df[f'ma_{period}'] - 1
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(
                features_df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features_df['macd'] = macd
            features_df['macdsignal'] = macdsignal
            features_df['macdhist'] = macdhist
            
            # RSI
            features_df['rsi_14'] = talib.RSI(features_df['close'].values, timeperiod=14)
            features_df['rsi_7'] = talib.RSI(features_df['close'].values, timeperiod=7)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                features_df['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            features_df['bb_upper'] = upper
            features_df['bb_middle'] = middle
            features_df['bb_lower'] = lower
            features_df['bb_width'] = (upper - lower) / middle
            features_df['bb_position'] = (features_df['close'] - lower) / (upper - lower)
            
            # ADX - Trend strength indicator
            features_df['adx'] = talib.ADX(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values,
                timeperiod=14
            )
            
            # ATR - Volatility indicator
            features_df['atr'] = talib.ATR(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values,
                timeperiod=14
            )
            features_df['atr_ratio'] = features_df['atr'] / features_df['close']
            
            # 4. Volume features
            # Normalize volume
            features_df['volume_norm'] = features_df['volume'] / features_df['volume'].rolling(window=20).max()
            
            # Volume change
            features_df['volume_change'] = features_df['volume'].pct_change()
            
            # OBV - On-Balance Volume
            features_df['obv'] = talib.OBV(features_df['close'].values, features_df['volume'].values)
            features_df['obv_change'] = features_df['obv'].pct_change(periods=5)
            
            # 5. Derived features
            # Price acceleration (second derivative)
            features_df['price_accel'] = features_df['close'].pct_change().pct_change()
            
            # Volatility feature
            features_df['volatility'] = features_df['close'].pct_change().rolling(window=10).std()
            
            # Trend consistency (price direction vs MA consistency)
            for period in [5, 20]:
                ma_col = f'ma_{period}'
                features_df[f'trend_consistency_{period}'] = (
                    (features_df['close'] > features_df[ma_col]) & 
                    (features_df['close'].shift(1) > features_df[ma_col].shift(1))
                ).astype(int) - (
                    (features_df['close'] < features_df[ma_col]) & 
                    (features_df['close'].shift(1) < features_df[ma_col].shift(1))
                ).astype(int)
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _prepare_training_data(self, klines):
        """
        Prepare model training data
        
        Args:
            klines (list): K-line data
            
        Returns:
            tuple: (X, y) feature matrix and labels
        """
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None, None
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Generate labels - future N k-lines' price change
            future_periods = 3  # Predict future 3 k-lines
            # 1: up (>0.5%), 0: neutral (±0.5%), -1: down (<-0.5%)
            future_returns = df['close'].pct_change(periods=future_periods).shift(-future_periods) * 100
            
            # Generate 3-class labels: down(-1), neutral(0), up(1)
            labels = np.zeros(len(future_returns))
            labels[future_returns > 0.5] = 1    # Up
            labels[future_returns < -0.5] = 2   # Down (use 2 for down to match model output)
            
            # Remove rows without future labels
            df = df.iloc[:-future_periods].copy()
            labels = labels[:-future_periods]
            
            # Select feature columns
            feature_columns = [
                'price_norm', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
                'return_1', 'return_5', 'return_10',
                'ma_ratio_5', 'ma_ratio_20', 'ma_ratio_50',
                'rsi_14', 'rsi_7',
                'macd', 'macdsignal', 'macdhist',
                'bb_width', 'bb_position',
                'adx', 'atr_ratio',
                'volume_norm'
            ]
            
            # Check if feature columns exist
            available_features = []
            for col in feature_columns:
                if col in df.columns:
                    available_features.append(col)
                else:
                    self.logger.error(f"Feature column '{col}' does not exist")
            
            if len(available_features) < 10:  # Require at least 10 features
                self.logger.error(f"Not enough feature columns available: {len(available_features)}")
                return None, None
            
            # Extract features
            X_raw = df[available_features].values
            
            # Feature scaling
            X_scaled = self.scaler.fit_transform(X_raw)
            
            # Create sequence data (for the transformer)
            X_sequences = []
            y_sequences = []
            
            for i in range(len(X_scaled) - self.sequence_length + 1):
                X_sequences.append(X_scaled[i:i+self.sequence_length])
                y_sequences.append(labels[i+self.sequence_length-1])
            
            # Convert to numpy arrays
            X = np.array(X_sequences, dtype=np.float32)
            y = np.array(y_sequences, dtype=np.int32)
            
            # Convert labels to one-hot encoding
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)
            
            # Check sample count
            if len(X) < 100:
                self.logger.warning(f"Too few samples: {len(X)}")
                return None, None
            
            self.logger.info(f"Training data preparation complete, feature shape: {X.shape}, label shape: {y_onehot.shape}")
            self.feature_count = X.shape[2]  # Update feature count
            
            return X, y_onehot
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def _prepare_prediction_data(self, klines):
        """
        Prepare prediction data
        
        Args:
            klines (list): K-line data
            
        Returns:
            numpy.ndarray: Feature matrix for prediction
        """
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Select feature columns
            feature_columns = [
                'price_norm', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
                'return_1', 'return_5', 'return_10',
                'ma_ratio_5', 'ma_ratio_20', 'ma_ratio_50',
                'rsi_14', 'rsi_7',
                'macd', 'macdsignal', 'macdhist',
                'bb_width', 'bb_position',
                'adx', 'atr_ratio',
                'volume_norm'
            ]
            
            # Check if feature columns exist
            available_features = []
            for col in feature_columns:
                if col in df.columns:
                    available_features.append(col)
                else:
                    self.logger.warning(f"Feature column '{col}' does not exist")
            
            if len(available_features) < 10:  # Require at least 10 features
                self.logger.error(f"Not enough feature columns available: {len(available_features)}")
                return None
            
            # Extract the most recent data
            X_raw = df[available_features].tail(self.sequence_length).values
            
            # Feature scaling (using trained scaler)
            X_scaled = self.scaler.transform(X_raw)
            
            # Check if data length is sufficient
            if len(X_scaled) < self.sequence_length:
                self.logger.error(f"Insufficient data length, need {self.sequence_length}, got {len(X_scaled)}")
                return None
            
            # Create single sample (for model input)
            X = np.array([X_scaled[-self.sequence_length:]])
            
            return X
            
        except Exception as e:
            self.logger.error(f"Prediction data preparation failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def should_retrain(self):
        """Check if model retraining is needed"""
        current_time = time.time()
        
        # If model never trained or time exceeds retraining interval, retrain
        if self.last_training_time == 0 or (current_time - self.last_training_time) > self.retraining_interval:
            return True
        
        return False
    
    def analyze_market_state(self, klines):
        """
        Analyze market state
        
        Args:
            klines (list): K-line data
            
        Returns:
            dict: Market state information
        """
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return {"state": "unknown", "trend": 0, "strength": 0}
            
            # Calculate ADX indicator - Trend strength
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            adx = talib.ADX(high, low, close, timeperiod=14)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Get recent values
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # Determine trend direction
            trend_direction = 0
            if current_plus_di > current_minus_di and current_plus_di > 20:
                trend_direction = 1  # Up trend
            elif current_minus_di > current_plus_di and current_minus_di > 20:
                trend_direction = -1  # Down trend
            
            # Determine market state
            state = "unknown"
            if current_adx >= 30:
                state = "trend"  # Strong trend
            elif current_adx >= 20:
                state = "trend"  # Trend
            else:
                state = "range"  # Range-bound market
            
            # Calculate volatility
            atr = talib.ATR(high, low, close, timeperiod=14)
            current_price = close[-1]
            volatility = (atr[-1] / current_price) * 100  # Volatility percentage
            
            # Calculate volume change
            volume = df['volume'].values
            volume_change = (volume[-1] / volume[-21:-1].mean() - 1) * 100  # Volume change percentage relative to 20-bar average
            
            # Update internal state
            self.market_state = state
            self.current_trend = trend_direction
            self.trend_strength = min(100, float(current_adx))
            
            # Return market state info
            market_state = {
                "state": state,
                "trend": trend_direction,
                "strength": min(100, float(current_adx)),
                "adx": float(current_adx),
                "plus_di": float(current_plus_di),
                "minus_di": float(current_minus_di),
                "volatility": float(volatility),
                "volume_change": float(volume_change)
            }
            
            self.logger.info(f"Market state: {state}, trend: {trend_direction}, "
                          f"strength: {current_adx:.2f}, volatility: {volatility:.2f}%, "
                          f"volume change: {volume_change:.2f}%")
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"Market state analysis failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"state": "unknown", "trend": 0, "strength": 0}
    
    def generate_signal(self, klines):
        """
        Generate trading signal
        
        Args:
            klines (list): K-line data
            
        Returns:
            int: Trading signal, 1(buy), -1(sell), 0(neutral)
        """
        try:
            # Check if model is initialized
            if self.model is None:
                self.logger.error("Model not initialized, cannot generate signal")
                return 0
            
            # Check if retraining is needed
            if self.should_retrain():
                self.logger.info("Model needs retraining")
                # Get more historical data for training
                training_klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                
                if training_klines and len(training_klines) > 0:
                    X, y = self._prepare_training_data(training_klines)
                    if X is not None and y is not None:
                        # Continue training on existing model
                        self.model.fit(
                            X, y,
                            epochs=5,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=3,
                                    restore_best_weights=True
                                )
                            ],
                            verbose=1
                        )
                        
                        # Save updated model
                        self.model.save(self.model_path['model'])
                        
                        # Update training time
                        self.last_training_time = time.time()
                        self.logger.info("Model retraining completed")
            
            # Prepare prediction data
            X = self._prepare_prediction_data(klines)
            if X is None:
                self.logger.error("Failed to prepare prediction data")
                return 0
            
            # Get model prediction
            predictions = self.model.predict(X, verbose=0)
            probabilities = predictions[0]  # Get first sample's prediction probabilities
            
            # Extract class probabilities
            up_prob = probabilities[0]    # Up probability (index 0)
            neutral_prob = probabilities[1]  # Neutral probability (index 1)
            down_prob = probabilities[2]    # Down probability (index 2)
            
            self.logger.info(f"Prediction probabilities: up={up_prob:.4f}, neutral={neutral_prob:.4f}, down={down_prob:.4f}")
            
            # Analyze market state
            market_state = self.analyze_market_state(klines)
            
            # Adjust prediction confidence threshold
            confidence_threshold = self.min_confidence
            
            # Lower confidence requirement in strong trend markets
            if market_state['state'] == "trend" and market_state['strength'] > 40:
                confidence_threshold = max(0.55, self.min_confidence - 0.1)
            
            # Increase confidence requirement in high volatility markets
            if market_state.get('volatility', 0) > 2.0:
                confidence_threshold = min(0.75, self.min_confidence + 0.1)
            
            # Determine trading signal
            signal = 0  # Default to neutral
            
            # Check for strong up signal
            if up_prob > down_prob:
                signal = 1  # Buy signal
                
            # Check for strong down signal
            elif down_prob > up_prob:
                signal = -1  # Sell signal
            
            # Signal verification and trend confirmation
            if signal != 0:
                # Check if prediction is against market trend
                if signal * market_state['trend'] < 0 and market_state['strength'] > 30:
                    self.logger.info(f"Prediction signal is against current trend direction, be cautious")
                    
                    # If trend is very strong, don't go against it
                    if market_state['strength'] > 50:
                        self.logger.info("Current trend is very strong, not going against it, changing signal to neutral")
                        signal = 0
            
            # Record signal history
            if signal != 0:
                self.signal_history.append({
                    'time': datetime.now(),
                    'signal': signal,
                    'up_prob': float(up_prob),
                    'down_prob': float(down_prob),
                    'market_state': market_state['state']
                })
                
                # Limit history size
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                self.last_signal_time = datetime.now()
                self.last_signal = signal
            
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
                
                # Get latest trend prediction, check if trend reversed
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                signal = self.generate_signal(klines)
                
                # If trend clearly reversed, consider closing the position
                if position_side == "long" and signal == -1:
                    self.logger.info("Trend prediction reversed to down, closing long position")
                    self.trader.close_position()
                    return
                elif position_side == "short" and signal == 1:
                    self.logger.info("Trend prediction reversed to up, closing short position")
                    self.trader.close_position()
                    return
                
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def run(self):
        """Run strategy"""
        self.logger.info("Starting TST+ strategy")
        
        # Initialize or load model
        if self.model is None:
            self._initialize_model()
        
        # Run parent class's run method
        super().run()
