"""
Deep Learning Strategy - 15 minute timeframe
Advanced LSTM-based model with attention mechanism for cryptocurrency price prediction
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
from strategies.base_strategy import BaseStrategy

class DeepLearningStrategy15m(BaseStrategy):
    """DeepLearningStrategy15m - Deep Learning Strategy for 15-minute timeframe
    
    An advanced LSTM-based deep learning strategy with attention mechanism designed
    specifically for 15-minute timeframe trading. This strategy captures both
    short-term patterns and medium-term trends in cryptocurrency markets.
    
    Features:
    1. Bidirectional LSTM with attention mechanism
    2. Multi-scale feature extraction (different timeframes)
    3. Advanced feature engineering including market microstructure
    4. Ensemble prediction with confidence scoring
    5. Dynamic risk management based on model confidence
    6. Adaptive retraining based on market regime changes
    """
    
    def __init__(self, trader):
        """Initialize the Deep Learning 15m strategy"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K-line settings
        self.kline_interval = '15m'      # 15-minute timeframe
        self.check_interval = 300        # Check signal interval (5 minutes)
        self.lookback_period = 500       # Number of k-lines for indicators
        self.training_lookback = 2000    # Number of k-lines for model training
        self.retraining_interval = 43200 # Retraining interval (12 hours)
        
        # Model parameters
        self.sequence_length = 48        # Input sequence length (12 hours of 15m candles)
        self.prediction_horizon = 4      # Predict 4 candles ahead (1 hour)
        self.lstm_units = 128            # LSTM hidden units
        self.attention_units = 64        # Attention layer units
        self.dropout_rate = 0.3          # Dropout rate
        self.model = None                # The model instance
        self.ensemble_models = []        # Ensemble of models
        self.model_path = self._get_model_path()  # Model save path
        self.scaler = None               # Feature scaler
        self.last_training_time = 0      # Last training time
        self._feature_count = None       # Will be set dynamically
        
        # Prediction thresholds
        self.min_confidence = 0.5       # Minimum prediction confidence
        self.strong_signal_threshold = 0.75  # Strong signal threshold
        self.ensemble_agreement = 0.7    # Minimum ensemble agreement
        
        # Feature engineering parameters
        self.feature_windows = [5, 10, 20, 40]  # Multiple timeframe windows
        self.volume_windows = [5, 10, 20]       # Volume analysis windows
        
        # Market regime detection
        self.market_regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']
        self.current_regime = None
        self.regime_history = []
        self.regime_change_threshold = 0.7
        
        # Position control parameters
        self.max_position_hold_time = 360   # Maximum position hold time (6 hours)
        self.base_stop_loss_pct = 0.02      # Base stop loss (2%)
        self.base_take_profit_pct = 0.04    # Base take profit (4%)
        self.dynamic_sl_tp = True           # Enable dynamic SL/TP based on volatility
        self.trailing_stop = True           # Enable trailing stop
        self.trailing_stop_activation = 0.015  # Activate at 1.5% profit
        self.trailing_stop_distance = 0.008    # Trailing stop distance (0.8%)
        
        # Risk management
        self.max_daily_trades = 8           # Maximum trades per day
        self.daily_trade_count = 0          # Current day trade count
        self.last_trade_date = None         # Last trade date
        self.consecutive_losses = 0         # Consecutive loss counter
        self.max_consecutive_losses = 3     # Maximum allowed consecutive losses
        
        # Signal history and validation
        self.signal_history = []            # Recent signal history
        self.prediction_history = []        # Model prediction history
        self.max_history_size = 100         # Maximum history size
        self.signal_consistency_window = 3  # Signal consistency check window
        
        # Performance tracking
        self.model_performance = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'total_trades': 0
        }
        
        # Internal state
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.current_volatility = 0
        self.current_trend_strength = 0
        
        # Initialize model
        self._initialize_models()
        
    def _get_model_path(self):
        """Get model save paths"""
        # Create model directory
        model_dir = os.path.join(os.getcwd(), 'models', 'dl_15m')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Create paths for different model components
        base_path = os.path.join(model_dir, f'dl_15m_{self.trader.symbol}')
        
        return {
            'main_model': f'{base_path}_main.h5',
            'ensemble': f'{base_path}_ensemble',
            'scaler': f'{base_path}_scaler.pkl',
            'performance': f'{base_path}_performance.json',
            'regime_model': f'{base_path}_regime.h5'
        }
        
    def _initialize_models(self):
        """Initialize or load the deep learning models"""
        try:
            # Try to load existing models
            if os.path.exists(self.model_path['main_model']):
                self.logger.info(f"Loading existing model: {self.model_path['main_model']}")
                
                # First, get feature count from a small sample of data
                sample_klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=200
                )
                
                if sample_klines:
                    # Prepare sample data to get feature count
                    df = self._prepare_dataframe(sample_klines)
                    if df is not None:
                        df = self._engineer_features(df)
                        feature_columns = [col for col in df.columns if col not in 
                                         ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
                        self._feature_count = len(feature_columns)
                        self.logger.info(f"Detected {self._feature_count} features")
                
                try:
                    self.model = tf.keras.models.load_model(
                        self.model_path['main_model'],
                        custom_objects={'AttentionLayer': AttentionLayer}
                    )
                    
                    # Load ensemble models
                    self._load_ensemble_models()
                    
                    # Load feature scaler
                    if os.path.exists(self.model_path['scaler']):
                        with open(self.model_path['scaler'], 'rb') as f:
                            self.scaler = pickle.load(f)
                    else:
                        self.scaler = StandardScaler()
                        
                    # Load performance metrics
                    if os.path.exists(self.model_path['performance']):
                        with open(self.model_path['performance'], 'r') as f:
                            self.model_performance = json.load(f)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to load model, will create new: {str(e)}")
                    # Create new models
                    self._build_models()
                    self._initial_training()
                    
            else:
                # Create new models
                self.logger.info("No pre-trained model found, creating new models")
                
                # Get feature count first
                sample_klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=200
                )
                
                if sample_klines:
                    df = self._prepare_dataframe(sample_klines)
                    if df is not None:
                        df = self._engineer_features(df)
                        feature_columns = [col for col in df.columns if col not in 
                                         ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
                        self._feature_count = len(feature_columns)
                        self.logger.info(f"Detected {self._feature_count} features")
                
                self._build_models()
                # Immediately train the models
                self._initial_training()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Create new models as backup
            self._build_models()
    
    def _build_models(self):
        """Build the deep learning models"""
        try:
            # Set random seed for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)
            
            # Build main LSTM model with attention
            self.model = self._build_lstm_attention_model()
            
            # Build ensemble models (3 different architectures)
            self.ensemble_models = [
                self._build_lstm_attention_model(),
                self._build_cnn_lstm_model(),
                self._build_gru_model()
            ]
            
            # Create feature scaler
            self.scaler = StandardScaler()
            
            self.logger.info("Deep learning models built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build models: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _build_lstm_attention_model(self):
        """Build LSTM model with attention mechanism"""
        # Calculate number of features dynamically
        num_features = self._calculate_feature_count()
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, num_features))
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate))(inputs)
        lstm2 = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_rate))(lstm1)
        
        # Attention mechanism
        attention = AttentionLayer(self.attention_units)(lstm2)
        
        # Concatenate LSTM output with attention
        concat = Concatenate()([lstm2, attention])
        
        # Global max pooling
        pooled = GlobalMaxPooling1D()(concat)
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(pooled)
        dropout1 = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(self.dropout_rate)(dense2)
        
        # Output layer: 3 classes (up, neutral, down)
        outputs = Dense(3, activation='softmax')(dropout2)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid model"""
        num_features = self._calculate_feature_count()
        
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, num_features)),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation='relu'),
            
            # LSTM layer
            LSTM(64, return_sequences=False, dropout=self.dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_gru_model(self):
        """Build GRU model"""
        num_features = self._calculate_feature_count()
        
        model = Sequential([
            GRU(128, return_sequences=True, dropout=self.dropout_rate, 
                input_shape=(self.sequence_length, num_features)),
            GRU(64, return_sequences=False, dropout=self.dropout_rate),
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_feature_count(self):
        """Calculate the number of features based on feature engineering"""
        # This will be set dynamically during feature engineering
        if hasattr(self, '_feature_count'):
            return self._feature_count
        # Default estimate
        return 60
    
    def _prepare_dataframe(self, klines):
        """Convert k-line data to DataFrame format"""
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
        """Advanced feature engineering for deep learning model"""
        try:
            features_df = df.copy()
            
            # 1. Basic price features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['hl_ratio'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['co_ratio'] = (features_df['close'] - features_df['open']) / features_df['open']
            features_df['upper_shadow'] = (features_df['high'] - features_df[['open', 'close']].max(axis=1)) / features_df['close']
            features_df['lower_shadow'] = (features_df[['open', 'close']].min(axis=1) - features_df['low']) / features_df['close']
            
            # 2. Multi-timeframe features
            for window in self.feature_windows:
                # Moving averages
                features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
                features_df[f'sma_ratio_{window}'] = features_df['close'] / features_df[f'sma_{window}']
                
                # Volatility
                features_df[f'volatility_{window}'] = features_df['returns'].rolling(window=window).std()
                
                # Price position in range
                rolling_high = features_df['high'].rolling(window=window).max()
                rolling_low = features_df['low'].rolling(window=window).min()
                features_df[f'price_position_{window}'] = (features_df['close'] - rolling_low) / (rolling_high - rolling_low)
                
                # Momentum
                features_df[f'momentum_{window}'] = features_df['close'].pct_change(periods=window)
            
            # 3. Technical indicators
            # RSI variations
            features_df['rsi_14'] = talib.RSI(features_df['close'].values, timeperiod=14)
            features_df['rsi_7'] = talib.RSI(features_df['close'].values, timeperiod=7)
            features_df['rsi_21'] = talib.RSI(features_df['close'].values, timeperiod=21)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(features_df['close'].values)
            features_df['macd'] = macd
            features_df['macd_signal'] = macdsignal
            features_df['macd_hist'] = macdhist
            features_df['macd_hist_diff'] = features_df['macd_hist'].diff()
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(features_df['close'].values)
            features_df['bb_upper'] = upper
            features_df['bb_middle'] = middle
            features_df['bb_lower'] = lower
            features_df['bb_width'] = (upper - lower) / middle
            features_df['bb_position'] = (features_df['close'] - lower) / (upper - lower)
            
            # Stochastic
            features_df['stoch_k'], features_df['stoch_d'] = talib.STOCH(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            )
            
            # ADX for trend strength
            features_df['adx'] = talib.ADX(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            )
            features_df['plus_di'] = talib.PLUS_DI(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            )
            features_df['minus_di'] = talib.MINUS_DI(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            )
            
            # 4. Volume features
            features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # Volume-weighted features
            features_df['vwap'] = (features_df['close'] * features_df['volume']).cumsum() / features_df['volume'].cumsum()
            features_df['vwap_ratio'] = features_df['close'] / features_df['vwap']
            
            # On-Balance Volume
            features_df['obv'] = talib.OBV(features_df['close'].values, features_df['volume'].values)
            features_df['obv_sma'] = features_df['obv'].rolling(window=10).mean()
            features_df['obv_ratio'] = features_df['obv'] / features_df['obv_sma']
            
            # 5. Market microstructure features
            # Spread
            features_df['spread'] = features_df['high'] - features_df['low']
            features_df['spread_ratio'] = features_df['spread'] / features_df['close']
            
            # Order flow imbalance proxy
            features_df['buy_pressure'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])
            features_df['sell_pressure'] = (features_df['high'] - features_df['close']) / (features_df['high'] - features_df['low'])
            features_df['order_imbalance'] = features_df['buy_pressure'] - features_df['sell_pressure']
            
            # 6. Pattern recognition features
            # Candlestick patterns
            features_df['doji'] = talib.CDLDOJI(
                features_df['open'].values,
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            ) / 100  # Normalize to [-1, 1]
            
            features_df['hammer'] = talib.CDLHAMMER(
                features_df['open'].values,
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            ) / 100
            
            features_df['engulfing'] = talib.CDLENGULFING(
                features_df['open'].values,
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values
            ) / 100
            
            # 7. Trend features
            # Linear regression slope
            for window in [10, 20, 50]:
                features_df[f'trend_slope_{window}'] = features_df['close'].rolling(window=window).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _detect_market_regime(self, df):
        """Detect current market regime"""
        try:
            # Calculate regime indicators
            adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
            volatility = df['volatility_20'].iloc[-1] if 'volatility_20' in df.columns else 0
            trend_slope = df['trend_slope_20'].iloc[-1] if 'trend_slope_20' in df.columns else 0
            bb_width = df['bb_width'].iloc[-1] if 'bb_width' in df.columns else 0
            
            # Handle NaN values
            adx = 0 if np.isnan(adx) else adx
            volatility = 0 if np.isnan(volatility) else volatility
            trend_slope = 0 if np.isnan(trend_slope) else trend_slope
            bb_width = 0 if np.isnan(bb_width) else bb_width
            
            # Determine regime
            if adx > 25 and trend_slope > 0.001:
                regime = 'trending_up'
            elif adx > 25 and trend_slope < -0.001:
                regime = 'trending_down'
            elif volatility > 0 and 'volatility_20' in df.columns:
                vol_threshold = df['volatility_20'].quantile(0.8)
                if volatility > vol_threshold:
                    regime = 'volatile'
                else:
                    regime = 'ranging'
            else:
                regime = 'ranging'
                
            # Update regime history
            self.regime_history.append({
                'time': datetime.now(),
                'regime': regime,
                'adx': float(adx),
                'volatility': float(volatility),
                'trend_slope': float(trend_slope)
            })
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
                
            self.current_regime = regime
            return regime
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed: {str(e)}")
            return 'unknown'
    
    def _prepare_training_data(self, klines):
        """Prepare model training data with advanced labeling"""
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None, None, None
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Detect market regime
            self._detect_market_regime(df)
            
            # Generate labels based on future returns
            future_returns = []
            for i in range(self.prediction_horizon):
                future_returns.append(df['close'].pct_change(periods=i+1).shift(-(i+1)))
            
            # Average future returns
            avg_future_return = pd.concat(future_returns, axis=1).mean(axis=1) * 100
            
            # Dynamic thresholds based on recent volatility
            if 'volatility_20' in df.columns:
                recent_volatility = df['volatility_20'].rolling(window=50).mean()
                # Fill NaN values with a default volatility
                recent_volatility = recent_volatility.fillna(0.01)  # 1% default volatility
            else:
                # Use returns standard deviation as volatility proxy
                recent_volatility = df['returns'].rolling(window=20).std()
                recent_volatility = recent_volatility.fillna(0.01)
                
            up_threshold = recent_volatility * 0.5  # Half of volatility
            down_threshold = -recent_volatility * 0.5
            
            # Generate labels
            labels = np.zeros(len(avg_future_return))
            labels[avg_future_return > up_threshold] = 0    # Up (class 0)
            labels[avg_future_return < down_threshold] = 2  # Down (class 2)
            labels[(avg_future_return >= down_threshold) & (avg_future_return <= up_threshold)] = 1  # Neutral (class 1)
            
            # Remove rows without future labels
            df = df.iloc[:-self.prediction_horizon].copy()
            labels = labels[:-self.prediction_horizon]
            
            # Select features
            feature_columns = [col for col in df.columns if col not in 
                             ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            regime_sequences = []
            
            for i in range(len(df) - self.sequence_length):
                # Feature sequence
                seq = df[feature_columns].iloc[i:i+self.sequence_length].values
                
                # Only use sequences with valid data
                if not np.isnan(seq).any():
                    X_sequences.append(seq)
                    y_sequences.append(labels[i+self.sequence_length-1])
                    
                    # Get regime for this sequence
                    regime_idx = i + self.sequence_length - 1
                    if regime_idx < len(self.regime_history):
                        regime_sequences.append(self.regime_history[regime_idx]['regime'])
            
            # Convert to numpy arrays
            X = np.array(X_sequences, dtype=np.float32)
            y = np.array(y_sequences, dtype=np.int32)
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            # Convert labels to one-hot encoding
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)
            
            self.logger.info(f"Training data prepared: X shape {X.shape}, y shape {y_onehot.shape}")
            
            return X, y_onehot, regime_sequences
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None, None
    
    def _initial_training(self):
        """Initial model training with ensemble"""
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
                X, y, regimes = self._prepare_training_data(klines)
                
                if X is not None and y is not None and len(X) > 100:
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Training callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                    ]
                    
                    # Train main model
                    self.logger.info("Training main model...")
                    history = self.model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Train ensemble models
                    for i, model in enumerate(self.ensemble_models):
                        self.logger.info(f"Training ensemble model {i+1}/{len(self.ensemble_models)}...")
                        model.fit(
                            X_train, y_train,
                            epochs=30,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=0
                        )
                    
                    # Save models
                    self._save_models()
                    
                    # Update performance metrics
                    self._update_performance_metrics(X_val, y_val)
                    
                    self.logger.info("Model training completed")
                    self.last_training_time = time.time()
                else:
                    self.logger.error("Insufficient training data")
            else:
                self.logger.error("Failed to get training data")
                
        except Exception as e:
            self.logger.error(f"Initial model training failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _save_models(self):
        """Save all models and components"""
        try:
            # Save main model
            self.model.save(self.model_path['main_model'])
            
            # Save ensemble models
            for i, model in enumerate(self.ensemble_models):
                model.save(f"{self.model_path['ensemble']}_{i}.h5")
            
            # Save scaler
            with open(self.model_path['scaler'], 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save performance metrics
            with open(self.model_path['performance'], 'w') as f:
                json.dump(self.model_performance, f)
                
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
    
    def _load_ensemble_models(self):
        """Load ensemble models"""
        try:
            self.ensemble_models = []
            for i in range(3):
                model_path = f"{self.model_path['ensemble']}_{i}.h5"
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects={'AttentionLayer': AttentionLayer}
                    )
                    self.ensemble_models.append(model)
                    
            self.logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble models: {str(e)}")
    
    def _update_performance_metrics(self, X_val, y_val):
        """Update model performance metrics"""
        try:
            # Get predictions
            y_pred = self.model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_val, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_true_classes)
            
            # Update performance
            self.model_performance['accuracy'] = float(accuracy)
            
            self.logger.info(f"Model performance - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {str(e)}")
    
    def generate_signal(self, klines):
        """Generate trading signal using ensemble prediction"""
        try:
            # Check if models are initialized
            if self.model is None:
                self.logger.error("Model not initialized")
                return 0
            
            # Check daily trade limit
            current_date = datetime.now().date()
            if self.last_trade_date != current_date:
                self.daily_trade_count = 0
                self.last_trade_date = current_date
            
            # if self.daily_trade_count >= self.max_daily_trades:
            #     self.logger.info("Daily trade limit reached")
            #     return 0
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.warning("Maximum consecutive losses reached, skipping signal")
                return 0
            
            # Prepare prediction data
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0
            
            df = self._engineer_features(df)
            
            # Detect market regime
            current_regime = self._detect_market_regime(df)
            
            # Select features
            feature_columns = [col for col in df.columns if col not in 
                             ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Check if feature count matches model expectation
            if len(feature_columns) != self._feature_count:
                self.logger.warning(f"Feature count mismatch: expected {self._feature_count}, got {len(feature_columns)}")
                # Update feature count and possibly retrain
                self._feature_count = len(feature_columns)
                return 0
            
            # Get the most recent sequence
            if len(df) < self.sequence_length:
                self.logger.error("Insufficient data for prediction")
                return 0
            
            X_raw = df[feature_columns].tail(self.sequence_length).values
            
            # Check for NaN values
            if np.isnan(X_raw).any():
                self.logger.error("NaN values in features, skipping prediction")
                return 0
            
            # Scale features
            X_scaled = self.scaler.transform(X_raw)
            X = np.array([X_scaled])
            
            # Get ensemble predictions
            predictions = []
            confidences = []
            
            # Main model prediction
            main_pred = self.model.predict(X, verbose=0)[0]
            predictions.append(np.argmax(main_pred))
            confidences.append(np.max(main_pred))
            
            # Ensemble model predictions
            for model in self.ensemble_models:
                pred = model.predict(X, verbose=0)[0]
                predictions.append(np.argmax(pred))
                confidences.append(np.max(pred))
            
            # Calculate ensemble agreement
            unique, counts = np.unique(predictions, return_counts=True)
            max_agreement = np.max(counts) / len(predictions)
            
            # Get majority vote
            majority_class = unique[np.argmax(counts)]
            avg_confidence = np.mean(confidences)
            
            self.logger.info(f"Ensemble predictions: {predictions}, Agreement: {max_agreement:.2f}, "
                           f"Avg confidence: {avg_confidence:.4f}, Regime: {current_regime}")
            
            # Dynamic confidence adjustment based on regime
            adjusted_confidence = avg_confidence
            if current_regime in ['trending_up', 'trending_down']:
                adjusted_confidence *= 1.1  # Boost confidence in trending markets
            elif current_regime == 'volatile':
                adjusted_confidence *= 0.9  # Reduce confidence in volatile markets
            
            # Generate signal
            signal = 0
            
            # Check confidence and agreement thresholds
            if max_agreement >= self.ensemble_agreement and adjusted_confidence >= self.min_confidence:
                if majority_class == 0:  # Up
                    signal = 1
                elif majority_class == 2:  # Down
                    signal = -1
                    
                # Additional regime-based filtering
                if current_regime == 'trending_up' and signal == -1:
                    self.logger.info("Filtering out sell signal in uptrend")
                    signal = 0
                elif current_regime == 'trending_down' and signal == 1:
                    self.logger.info("Filtering out buy signal in downtrend")
                    signal = 0
            
            # Check signal consistency
            if signal != 0 and len(self.signal_history) >= self.signal_consistency_window:
                recent_signals = [s['signal'] for s in self.signal_history[-self.signal_consistency_window:]]
                if all(s == -signal for s in recent_signals):
                    self.logger.info("Signal conflicts with recent history, skipping")
                    signal = 0
            
            # Record signal
            if signal != 0:
                self.signal_history.append({
                    'time': datetime.now(),
                    'signal': signal,
                    'confidence': float(adjusted_confidence),
                    'regime': current_regime
                })
                
                # Limit history size
                if len(self.signal_history) > self.max_history_size:
                    self.signal_history = self.signal_history[-self.max_history_size:]
                
                self.daily_trade_count += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to generate signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """Monitor current position with advanced risk management"""
        try:
            # Get current position
            position = self.trader.get_position()
            
            # If no position, check for new signal
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # Get latest k-line data
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
                # Generate trading signal
                signal = self.generate_signal(klines)
                
                if signal != 0:
                    # Get current market price
                    current_price = self.trader.get_market_price()
                    
                    # Calculate position size based on Kelly Criterion or fixed percentage
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # Dynamic position sizing based on confidence
                    if len(self.signal_history) > 0:
                        recent_confidence = self.signal_history[-1]['confidence']
                        position_percent = min(50, 20 + (recent_confidence - 0.6) * 100)
                    else:
                        position_percent = 30
                    
                    trade_amount = (available_balance * position_percent / 100) / current_price
                    
                    if signal == 1:  # Buy signal
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"Opened long position - Amount: {trade_amount:.6f}, Price: {current_price}")
                    elif signal == -1:  # Sell signal
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"Opened short position - Amount: {trade_amount:.6f}, Price: {current_price}")
                    
                    # Record entry information
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    
                    # Calculate current volatility for dynamic SL/TP
                    df = self._prepare_dataframe(klines)
                    if df is not None:
                        df = self._engineer_features(df)
                        self.current_volatility = df['volatility_20'].iloc[-1] if 'volatility_20' in df.columns else 0.02
            
            # If position exists, manage it
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                current_price = self.trader.get_market_price()
                position_side = "long" if position_amount > 0 else "short"
                
                # Calculate profit/loss
                if position_side == "long":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # Update maximum profit
                if profit_rate > self.max_profit_reached:
                    self.max_profit_reached = profit_rate
                
                # Dynamic stop loss and take profit based on volatility
                if self.dynamic_sl_tp and self.current_volatility > 0:
                    stop_loss = self.base_stop_loss_pct * (1 + self.current_volatility * 2)
                    take_profit = self.base_take_profit_pct * (1 + self.current_volatility)
                else:
                    stop_loss = self.base_stop_loss_pct
                    take_profit = self.base_take_profit_pct
                
                # Check exit conditions
                exit_position = False
                exit_reason = ""
                
                # 1. Check holding time
                if self.position_entry_time:
                    holding_time = (time.time() - self.position_entry_time) / 60
                    if holding_time >= self.max_position_hold_time:
                        exit_position = True
                        exit_reason = "Maximum holding time reached"
                
                # 2. Check stop loss
                if profit_rate <= -stop_loss:
                    exit_position = True
                    exit_reason = f"Stop loss hit at {profit_rate:.4%}"
                    self.consecutive_losses += 1
                
                # 3. Check take profit
                if profit_rate >= take_profit:
                    exit_position = True
                    exit_reason = f"Take profit hit at {profit_rate:.4%}"
                    self.consecutive_losses = 0  # Reset on profitable trade
                
                # 4. Check trailing stop
                if self.trailing_stop and profit_rate >= self.trailing_stop_activation:
                    drawdown = self.max_profit_reached - profit_rate
                    if drawdown >= self.trailing_stop_distance:
                        exit_position = True
                        exit_reason = f"Trailing stop triggered, max: {self.max_profit_reached:.4%}, current: {profit_rate:.4%}"
                
                # 5. Check for signal reversal
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                current_signal = self.generate_signal(klines)
                
                if (position_side == "long" and current_signal == -1) or \
                   (position_side == "short" and current_signal == 1):
                    exit_position = True
                    exit_reason = "Signal reversal detected"
                
                # Execute exit if needed
                if exit_position:
                    self.logger.info(f"Closing position - Reason: {exit_reason}")
                    self.trader.close_position()
                    
                    # Update performance tracking
                    if profit_rate > 0:
                        self.model_performance['total_trades'] += 1
                        wins = self.model_performance.get('wins', 0) + 1
                        self.model_performance['wins'] = wins
                        self.model_performance['win_rate'] = wins / self.model_performance['total_trades']
                    else:
                        self.model_performance['total_trades'] += 1
                        self.model_performance['win_rate'] = self.model_performance.get('wins', 0) / self.model_performance['total_trades']
                
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def should_retrain(self):
        """Check if model needs retraining based on performance and regime changes"""
        current_time = time.time()
        
        # Time-based retraining
        if current_time - self.last_training_time > self.retraining_interval:
            return True
        
        # Performance-based retraining
        if self.model_performance.get('win_rate', 1.0) < 0.4 and self.model_performance.get('total_trades', 0) > 10:
            self.logger.info("Model performance degraded, triggering retraining")
            return True
        
        # Regime change detection
        if len(self.regime_history) >= 20:
            recent_regimes = [r['regime'] for r in self.regime_history[-20:]]
            regime_changes = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i-1])
            
            if regime_changes > 10:  # Too many regime changes
                self.logger.info("Frequent regime changes detected, triggering retraining")
                return True
        
        return False


# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer for sequence models"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_vector'
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Calculate attention scores
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights
        a = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        weighted_input = inputs * tf.expand_dims(a, -1)
        
        return weighted_input
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config