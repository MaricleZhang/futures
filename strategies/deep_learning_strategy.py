"""
深度学习交易策略 - 30分钟时间框架
基于深度学习模型的价格趋势预测
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
import os
from strategies.base_strategy import BaseStrategy
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DeepLearningStrategy30m(BaseStrategy):
    """DeepLearningStrategy30m - 深度学习交易策略
    
    基于30分钟K线的深度学习预测策略，通过LSTM神经网络模型
    学习市场模式和价格走势，提供更加准确的趋势预测。
    
    特点:
    1. 深度LSTM架构：捕捉时间序列的长期依赖关系
    2. 多特征融合：整合技术指标、价格数据和交易量数据
    3. 滚动预测：定期重新训练模型适应市场变化
    4. 概率输出：不仅预测方向，还给出置信度
    5. 市场状态感知：区分不同市场状态下的预测准确性
    6. 动态特征重要性：识别当前市场中最有效的指标
    """
    
    def __init__(self, trader):
        """初始化深度学习预测策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '30m'       # 30分钟K线
        self.check_interval = 300         # 检查信号间隔(秒)
        self.lookback_period = 400        # 计算指标所需的K线数量
        self.training_lookback = 766     # 训练模型所需的K线数量
        self.sequence_length = 20         # 序列长度，用于LSTM输入
        
        # 模型参数
        self.model_path = 'models'
        self.model_name = f'lstm_model_{trader.symbol.replace("/", "_")}.h5'
        self.batch_size = 32
        self.epochs = 50
        self.validation_split = 0.2
        self.learning_rate = 0.001
        self.loss_function = 'categorical_crossentropy'
        self.prediction_threshold = 0.55  # 预测概率阈值
        self.retraining_interval = 24 * 60 * 60  # 重新训练间隔(秒)
        
        # 特征工程参数
        self.use_ta_features = True       # 是否使用技术指标特征
        self.use_price_features = True    # 是否使用价格特征
        self.use_volume_features = True   # 是否使用交易量特征
        self.normalize_features = True    # 是否标准化特征
        
        # 训练参数
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        self.last_training_time = 0
        
        # 预测和信号参数
        self.signal_smoothing = True      # 是否平滑信号
        self.smoothing_window = 3         # 平滑窗口大小
        self.consensus_threshold = 0.7    # 信号一致性阈值
        self.min_signal_confidence = 0.6  # 最小信号置信度
        
        # 风险控制参数
        self.max_position_hold_time = 8 * 60  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.05     # 目标利润率
        self.stop_loss_pct = 0.02         # 止损比例
        self.trailing_stop = True         # 是否启用追踪止损
        self.trailing_stop_activation = 0.02  # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.01    # 追踪止损距离百分比
        
        # 内部状态
        self.model = None
        self.scalers = {}
        self.last_signals = []
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        
        # 创建模型目录
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        # 加载或初始化模型
        self._load_or_init_model()
        
    def _load_or_init_model(self):
        """加载现有模型或初始化新模型"""
        model_file = os.path.join(self.model_path, self.model_name)
        
        if os.path.exists(model_file):
            try:
                self.logger.info(f"加载现有模型: {model_file}")
                self.model = load_model(model_file)
                return
            except Exception as e:
                self.logger.error(f"加载模型失败: {str(e)}")
        
        self.logger.info("现有模型不存在，将在首次运行时训练新模型")
        
    def _build_model(self, input_shape):
        """构建LSTM深度学习模型"""
        self.logger.info(f"构建深度学习模型，输入形状: {input_shape}")
        
        try:
            # 使用TensorFlow混合精度
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                self.logger.info("启用混合精度训练")
            except:
                self.logger.info("无法启用混合精度训练，使用默认精度")
            
            # 两层LSTM模型，带Dropout和BatchNormalization
            model = Sequential()
            
            # 第一层LSTM - 使用较低的recurrent_dropout值，避免可能的兼容性问题
            model.add(LSTM(units=64, 
                          return_sequences=True, 
                          input_shape=input_shape,
                          recurrent_dropout=0.1))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 第二层LSTM
            model.add(LSTM(units=32, 
                          return_sequences=False, 
                          recurrent_dropout=0.1))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 全连接层
            model.add(Dense(units=16, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 输出层 - 使用softmax进行多分类：上涨、下跌、横盘
            model.add(Dense(units=3, activation='softmax'))
            
            # 编译模型 - 使用明确的TensorFlow数据类型
            # Adam优化器中显式设置数据类型
            optimizer = Adam(learning_rate=float(self.learning_rate))
            
            model.compile(
                optimizer=optimizer,
                loss=self.loss_function,
                metrics=['accuracy']
            )
            
            self.model = model
            self.logger.info("模型构建完成")
            self.logger.info(f"模型配置: {model.get_config()}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"构建模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def prepare_features(self, klines):
        """准备特征数据
        
        Args:
            klines (list): K线数据 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: 特征数据DataFrame
        """
        try:
            # 转换K线数据到DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算价格特征
            if self.use_price_features:
                # 价格变化率
                df['price_change'] = df['close'].pct_change()
                df['price_change_1'] = df['price_change'].shift(1)
                df['price_change_2'] = df['price_change'].shift(2)
                
                # 价格波动
                df['price_volatility'] = df['close'].rolling(window=10).std() / df['close']
                
                # 价格走势
                df['price_trend_5'] = df['close'].pct_change(periods=5)
                df['price_trend_10'] = df['close'].pct_change(periods=10)
                df['price_trend_20'] = df['close'].pct_change(periods=20)
                
                # 价格与移动平均线的关系
                df['ema_5'] = talib.EMA(df['close'].values, timeperiod=5)
                df['ema_10'] = talib.EMA(df['close'].values, timeperiod=10)
                df['ema_20'] = talib.EMA(df['close'].values, timeperiod=20)
                df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
                
                df['price_to_ema_5'] = df['close'] / df['ema_5'] - 1
                df['price_to_ema_10'] = df['close'] / df['ema_10'] - 1
                df['price_to_ema_20'] = df['close'] / df['ema_20'] - 1
                df['price_to_ema_50'] = df['close'] / df['ema_50'] - 1
                
                # 价格与前期高低点的关系
                df['high_10'] = df['high'].rolling(window=10).max()
                df['low_10'] = df['low'].rolling(window=10).min()
                df['high_20'] = df['high'].rolling(window=20).max()
                df['low_20'] = df['low'].rolling(window=20).min()
                
                df['price_to_high_10'] = df['close'] / df['high_10'] - 1
                df['price_to_low_10'] = df['close'] / df['low_10'] - 1
                df['price_to_high_20'] = df['close'] / df['high_20'] - 1
                df['price_to_low_20'] = df['close'] / df['low_20'] - 1
            
            # 计算技术指标特征
            if self.use_ta_features:
                # RSI
                df['rsi_6'] = talib.RSI(df['close'].values, timeperiod=6)
                df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(
                    df['close'].values, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_hist'] = macd_hist
                
                # 布林带
                upper, middle, lower = talib.BBANDS(
                    df['close'].values, 
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2
                )
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
                df['bb_width'] = (upper - lower) / middle
                df['bb_position'] = (df['close'] - lower) / (upper - lower)
                
                # ATR - 真实波动范围
                df['atr'] = talib.ATR(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=14
                )
                df['atr_percent'] = df['atr'] / df['close']
                
                # ADX - 平均趋向指数
                df['adx'] = talib.ADX(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=14
                )
                
                # CCI - 顺势指标
                df['cci'] = talib.CCI(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=20
                )
                
                # 随机指标
                df['slowk'], df['slowd'] = talib.STOCH(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0
                )
                
                # OBV - 能量潮
                df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
                df['obv_change'] = df['obv'].pct_change()
                
                # 动量指标
                df['momentum_10'] = talib.MOM(df['close'].values, timeperiod=10)
                df['momentum_20'] = talib.MOM(df['close'].values, timeperiod=20)
            
            # 计算交易量特征
            if self.use_volume_features:
                # 交易量变化率
                df['volume_change'] = df['volume'].pct_change()
                df['volume_change_1'] = df['volume_change'].shift(1)
                df['volume_change_2'] = df['volume_change'].shift(2)
                
                # 交易量移动平均
                df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
                df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                
                # 相对交易量
                df['rel_volume_5'] = df['volume'] / df['volume_ma_5']
                df['rel_volume_10'] = df['volume'] / df['volume_ma_10']
                df['rel_volume_20'] = df['volume'] / df['volume_ma_20']
                
                # 价格上涨/下跌的交易量比率
                df['up_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
                df['down_volume'] = df['volume'] * (df['close'] < df['open']).astype(int)
                df['up_volume_ma_5'] = df['up_volume'].rolling(window=5).mean()
                df['down_volume_ma_5'] = df['down_volume'].rolling(window=5).mean()
                df['volume_ratio'] = df['up_volume_ma_5'] / df['down_volume_ma_5'].replace(0, 1)
            
            # 删除NaN值
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def prepare_training_data(self, df, target_column='close', future_bars=3, threshold=0.003):
        """
        准备训练数据，创建序列输入和分类标签
        
        Args:
            df (pandas.DataFrame): 包含特征的DataFrame
            target_column (str): 用于生成标签的目标列
            future_bars (int): 预测未来多少根K线
            threshold (float): 价格变动阈值，大于该值视为上涨，小于负阈值视为下跌
            
        Returns:
            tuple: (X, y) 特征序列和标签
        """
        try:
            # 删除不需要的列
            drop_columns = ['timestamp', 'datetime']
            feature_df = df.drop(columns=drop_columns, errors='ignore')
            
            # 创建标签：1(上涨)、0(横盘)、-1(下跌)
            df['future_return'] = df[target_column].pct_change(periods=future_bars).shift(-future_bars)
            df['label'] = 0  # 默认为横盘
            df.loc[df['future_return'] > threshold, 'label'] = 1  # 上涨
            df.loc[df['future_return'] < -threshold, 'label'] = -1  # 下跌
            
            # 独热编码标签
            df['label_up'] = (df['label'] == 1).astype(int)
            df['label_neutral'] = (df['label'] == 0).astype(int)
            df['label_down'] = (df['label'] == -1).astype(int)
            
            # 删除有NaN的行
            feature_df = feature_df.dropna()
            
            # 选择特征列
            feature_columns = feature_df.columns
            
            # 创建特征矩阵和标签
            X_list = []
            y_list = []
            
            # 对每个特征进行标准化
            self.scalers = {}
            scaled_features = feature_df.copy()
            
            for column in feature_columns:
                scaler = StandardScaler()
                scaled_features[column] = scaler.fit_transform(feature_df[column].values.reshape(-1, 1)).flatten()
                self.scalers[column] = scaler
            
            # 创建序列数据
            for i in range(self.sequence_length, len(scaled_features) - future_bars):
                # 序列特征
                seq_features = scaled_features.iloc[i - self.sequence_length:i].values
                X_list.append(seq_features)
                
                # 标签
                label = df.iloc[i][['label_up', 'label_neutral', 'label_down']].values
                y_list.append(label)
            
            # 转换为numpy数组并显式指定数据类型为float32
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            
            # 检查无效值(NaN或Inf)
            if np.isnan(X).any() or np.isinf(X).any():
                self.logger.warning("特征数据中存在NaN或Inf值，将替换为0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
            if np.isnan(y).any() or np.isinf(y).any():
                self.logger.warning("标签数据中存在NaN或Inf值，将替换为0")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 确保标签是合法的one-hot编码
            y_sum = np.sum(y, axis=1)
            if not np.all(y_sum == 1):
                self.logger.warning("存在非标准one-hot编码标签，将进行修正")
                # 修正: 设置最大值为1，其他为0
                for i in range(len(y)):
                    if y_sum[i] != 1:
                        max_idx = np.argmax(y[i])
                        y[i] = np.zeros(3, dtype=np.float32)
                        y[i][max_idx] = 1.0
            
            self.logger.info(f"数据准备完成: X.shape={X.shape}, X.dtype={X.dtype}, y.shape={y.shape}, y.dtype={y.dtype}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def train_model(self, klines=None):
        """
        训练深度学习模型
        
        Args:
            klines (list, optional): K线数据。如果为None，则从交易所获取
            
        Returns:
            bool: 训练是否成功
        """
        try:
            # 设置TensorFlow内存增长
            try:
                import tensorflow as tf
                # 允许内存增长
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"已配置GPU内存增长，检测到{len(gpus)}个GPU")
            except Exception as e:
                self.logger.info(f"GPU配置失败: {str(e)}，使用CPU训练")
            
            # 获取K线数据
            if klines is None or len(klines) < self.training_lookback:
                self.logger.info(f"获取训练数据: {self.training_lookback}根K线")
                klines = self.trader.get_klines(
                    interval=self.kline_interval, 
                    limit=self.training_lookback
                )
                
            if klines is None or len(klines) < self.training_lookback:
                self.logger.error(f"训练数据不足: 需要{self.training_lookback}根K线，实际获取{len(klines) if klines else 0}根")
                return False
                
            self.logger.info(f"开始训练模型，使用{len(klines)}根K线数据")
            
            # 准备特征
            df = self.prepare_features(klines)
            if df is None or len(df) < self.sequence_length + 10:
                self.logger.error(f"特征准备失败或数据不足: DataFrame长度为{len(df) if df is not None else 0}")
                return False
            
            # 准备训练数据
            X, y = self.prepare_training_data(df)
            if X is None or y is None:
                self.logger.error("训练数据准备失败，X或y为None")
                return False
                
            self.logger.info(f"训练数据形状: X={X.shape}, y={y.shape}")
            
            # 降低批量大小，使用更小的模型以避免内存问题
            self.batch_size = min(16, self.batch_size)
            self.epochs = min(20, self.epochs)
            
            # 检查数据有效性
            if X.shape[0] < self.batch_size * 4:
                self.logger.error(f"训练样本数量太少: {X.shape[0]}，至少需要{self.batch_size * 4}个样本")
                return False
            
            # 构建模型
            if self.model is None:
                self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
                
            if self.model is None:
                self.logger.error("模型构建失败")
                return False
            
            # 设置回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=0.0001
                )
            ]
            
            # 设置检查点
            try:
                model_file = os.path.join(self.model_path, self.model_name)
                model_checkpoint = ModelCheckpoint(
                    filepath=model_file,
                    monitor='val_loss',
                    save_best_only=True
                )
                callbacks.append(model_checkpoint)
            except Exception as e:
                self.logger.warning(f"设置模型检查点失败: {str(e)}，将不保存中间模型")
            
            # 尝试使用较小的验证集以节省内存
            validation_split = min(0.15, self.validation_split)
            
            # 使用异常处理包装fit调用以捕获详细错误
            try:
                # 训练模型
                history = self.model.fit(
                    X, y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=2
                )
                
                # 保存模型
                try:
                    self.model.save(os.path.join(self.model_path, self.model_name))
                    self.logger.info(f"模型训练完成，保存至: {os.path.join(self.model_path, self.model_name)}")
                except Exception as e:
                    self.logger.error(f"保存模型失败: {str(e)}")
                
                # 更新训练时间
                self.last_training_time = time.time()
                
                # 记录训练结果
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history['accuracy'][-1]
                val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
                val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
                
                self.logger.info(f"训练结果: loss={final_loss:.4f}, accuracy={final_accuracy:.4f}")
                if 'val_loss' in history.history:
                    self.logger.info(f"验证结果: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
                
                return True
            except Exception as e:
                self.logger.error(f"模型训练过程中发生错误: {str(e)}")
                # 尝试更简单的模型
                self.logger.info("尝试使用更简单的模型架构...")
                
                # 使用更简单的模型架构
                simple_model = Sequential([
                    LSTM(32, input_shape=(X.shape[1], X.shape[2])),
                    Dense(16, activation='relu'),
                    Dense(3, activation='softmax')
                ])
                
                simple_model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.model = simple_model
                
                # 使用更小的批量和更少的周期
                simple_history = self.model.fit(
                    X, y,
                    epochs=10,
                    batch_size=8,
                    validation_split=0.1,
                    verbose=2
                )
                
                self.logger.info("简化模型训练完成")
                self.last_training_time = time.time()
                
                return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def should_retrain(self):
        """检查是否需要重新训练模型"""
        current_time = time.time()
        if self.model is None:
            return True
        if current_time - self.last_training_time > self.retraining_interval:
            return True
        return False
    
    def predict(self, klines):
        """
        使用模型进行预测
        
        Args:
            klines (list): K线数据
            
        Returns:
            tuple: (预测类别, 预测概率)
        """
        try:
            if self.model is None or self.should_retrain():
                success = self.train_model(klines)
                if not success:
                    self.logger.error("模型训练失败，无法进行预测")
                    return 0, [0.33, 0.34, 0.33]  # 返回均匀分布，表示不确定性
            
            # 准备特征
            df = self.prepare_features(klines)
            if df is None or len(df) < self.sequence_length:
                self.logger.error("特征准备失败或数据不足")
                return 0, [0.33, 0.34, 0.33]
            
            # 选择最近的序列长度作为输入
            recent_df = df.tail(self.sequence_length)
            
            # 应用特征缩放
            scaled_features = recent_df.copy()
            for column in scaled_features.columns:
                if column in self.scalers:
                    scaled_features[column] = self.scalers[column].transform(
                        scaled_features[column].values.reshape(-1, 1)
                    ).flatten()
            
            # 删除不需要的列
            drop_columns = ['timestamp', 'datetime']
            scaled_features = scaled_features.drop(columns=drop_columns, errors='ignore')
            
            # 创建模型输入
            X = scaled_features.values.reshape(1, self.sequence_length, -1)
            
            # 进行预测
            prediction = self.model.predict(X, verbose=0)[0]
            
            # 将预测概率转换为类别（上涨、横盘、下跌）
            predicted_class = np.argmax(prediction)
            
            # 将类别从[0,1,2]映射到[1,0,-1]
            class_mapping = {0: 1, 1: 0, 2: -1}
            signal = class_mapping[predicted_class]
            
            # 记录信号
            self.last_signals.append(signal)
            if len(self.last_signals) > self.smoothing_window:
                self.last_signals.pop(0)
            
            # 输出预测结果
            self.logger.info(f"预测结果: 上涨={prediction[0]:.4f}, 横盘={prediction[1]:.4f}, 下跌={prediction[2]:.4f}")
            self.logger.info(f"预测信号: {signal} (置信度: {prediction[predicted_class]:.4f})")
            
            return signal, prediction
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0, [0.33, 0.34, 0.33]
    
    def generate_signal(self, klines):
        """
        生成交易信号
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 进行预测
            signal, probabilities = self.predict(klines)
            
            # 获取最高概率
            max_prob = max(probabilities)
            
            # 如果信号平滑启用，使用最近几个信号的众数
            if self.signal_smoothing and len(self.last_signals) >= self.smoothing_window:
                # 计算众数
                from collections import Counter
                signal_counter = Counter(self.last_signals)
                most_common_signal, count = signal_counter.most_common(1)[0]
                
                # 检查信号一致性
                consistency = count / len(self.last_signals)
                
                if consistency >= self.consensus_threshold:
                    signal = most_common_signal
                    self.logger.info(f"平滑后的信号: {signal} (一致性: {consistency:.4f})")
                else:
                    # 如果一致性不够，考虑概率值
                    if max_prob >= self.min_signal_confidence:
                        # 使用原始预测
                        self.logger.info(f"信号一致性不足但置信度高: {signal} (置信度: {max_prob:.4f})")
                    else:
                        # 置信度不够，保持观望
                        signal = 0
                        self.logger.info(f"信号不稳定且置信度不足，保持观望")
            else:
                # 没有足够历史信号，检查置信度
                if max_prob < self.min_signal_confidence:
                    signal = 0
                    self.logger.info(f"置信度不足，保持观望")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
                # 生成交易信号
                signal = self.generate_signal(klines)
                
                # 获取当前市场价格
                current_price = self.trader.get_market_price()
                
                # 根据信号执行交易
                if signal == 1:  # 买入信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 开多仓
                    self.trader.open_long(amount=trade_amount)
                    self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    
                elif signal == -1:  # 卖出信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 开空仓
                    self.trader.open_short(amount=trade_amount)
                    self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # 如果有持仓，检查是否需要平仓
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                current_price = self.trader.get_market_price()
                position_side = "多" if position_amount > 0 else "空"
                
                # 计算持仓时间
                current_time = time.time()
                if self.position_entry_time is not None:
                    holding_time_minutes = (current_time - self.position_entry_time) / 60
                    
                    # 检查最大持仓时间
                    if holding_time_minutes >= self.max_position_hold_time:
                        self.logger.info(f"持仓时间超过{self.max_position_hold_time}分钟，平仓")
                        self.trader.close_position()
                        return
                
                # 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # 更新最大利润记录
                if profit_rate > self.max_profit_reached:
                    self.max_profit_reached = profit_rate
                
                # 检查止盈
                if profit_rate >= self.profit_target_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查追踪止损
                if self.trailing_stop and profit_rate >= self.trailing_stop_activation:
                    # 计算回撤比例
                    drawdown = self.max_profit_reached - profit_rate
                    
                    # 如果回撤超过追踪止损距离，平仓止盈
                    if drawdown >= self.trailing_stop_distance:
                        self.logger.info(f"触发追踪止损，最大利润: {self.max_profit_reached:.4%}, " +
                                      f"当前利润: {profit_rate:.4%}, 回撤: {drawdown:.4%}")
                        self.trader.close_position()
                        return
                
                # 获取最新信号，检查趋势是否反转
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                new_signal = self.generate_signal(klines)
                
                # 如果信号与当前持仓方向相反，平仓
                if (position_side == "多" and new_signal == -1) or (position_side == "空" and new_signal == 1):
                    self.logger.info(f"趋势反转，信号: {new_signal}，平仓")
                    self.trader.close_position()
                    return
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
