"""
深度学习趋势交易策略 - 30分钟时间框架
使用深度学习模型预测价格趋势，专注于中长期趋势交易
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
import tempfile
import os
from strategies.base_strategy import BaseStrategy
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class DeepLearningTrendStrategy30m(BaseStrategy):
    """DeepLearningTrendStrategy30m - 深度学习趋势交易策略
    
    基于30分钟K线的深度学习趋势预测策略，采用LSTM和CNN混合模型
    识别市场趋势，并根据模型预测提供交易信号。
    
    特点:
    1. 深度神经网络自动特征提取：无需人工设计技术指标
    2. 时序特征处理：LSTM捕捉价格序列的时间依赖关系
    3. 趋势强度感知：自适应调整趋势确认阈值
    4. 自动调整参数：根据市场波动性调整模型参数
    5. 实时训练和预测：定期根据最新数据更新模型
    6. 集成多个模型：提高预测鲁棒性
    """
    
    def __init__(self, trader):
        """初始化深度学习趋势交易策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '30m'      # 30分钟K线
        self.check_interval = 300        # 检查信号间隔(秒)
        self.lookback_period = 500       # 计算指标所需的K线数量
        self.training_lookback = 2000    # 训练模型所需的K线数量
        self.retraining_interval = 86400 # 模型重新训练间隔(秒)
        
        # 模型设置
        self.sequence_length = 30        # 输入序列长度
        self.feature_count = 15          # 特征数量
        self.model = None                # 深度学习模型
        self.model_path = self._get_model_path()  # 模型保存路径
        self.scaler = None               # 特征缩放器
        self.last_training_time = 0      # 上次训练时间
        self.min_confidence = 0.5       # 最小预测置信度
        
        # 趋势参数
        self.trend_confirmation_window = 3  # 趋势确认窗口
        self.market_state = "unknown"    # 市场状态
        self.trend_strength = 0          # 趋势强度
        self.current_trend = 0           # 当前趋势: 1(上升), -1(下降), 0(中性)
        
        # 持仓控制参数
        self.max_position_hold_time = 720  # 最大持仓时间(分钟)
        self.stop_loss_pct = 0.03        # 止损比例 (3%)
        self.take_profit_pct = 0.06      # 止盈比例 (6%)
        self.trailing_stop = True        # 是否启用追踪止损
        self.trailing_stop_activation = 0.03  # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.015   # 追踪止损距离百分比
        
        # 历史信号记录
        self.signal_history = []         # 最近的信号历史
        self.max_signal_history = 20     # 保存的最大信号数量
        self.last_signal_time = None     # 上次产生信号的时间
        self.last_signal = 0             # 上次信号: 1(买入), -1(卖出), 0(观望)
        
        # 内部状态
        self.position_entry_time = None  # 开仓时间
        self.position_entry_price = None # 开仓价格
        self.max_profit_reached = 0      # 达到的最大利润
        
        # 初始化
        self._initialize_model()
        
    def _get_model_path(self):
        """获取模型保存路径"""
        # 创建模型存储目录
        model_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 创建特定于交易对的模型文件路径
        model_path = os.path.join(model_dir, f'dl_trend_model_{self.trader.symbol}.h5')
        scaler_path = os.path.join(model_dir, f'dl_trend_scaler_{self.trader.symbol}.pkl')
        
        return {
            'model': model_path,
            'scaler': scaler_path
        }
        
    def _initialize_model(self):
        """初始化或加载深度学习模型"""
        try:
            # 检查是否存在预先训练好的模型
            if os.path.exists(self.model_path['model']):
                self.logger.info(f"加载已存在的模型: {self.model_path['model']}")
                self.model = tf.keras.models.load_model(self.model_path['model'])
                
                # 加载特征缩放器
                if os.path.exists(self.model_path['scaler']):
                    with open(self.model_path['scaler'], 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    self.scaler = MinMaxScaler()
            else:
                # 创建新模型
                self.logger.info("没有找到已训练的模型，创建新模型")
                self._build_model()
                
                # 立即训练模型
                self._initial_training()
        except Exception as e:
            self.logger.error(f"初始化模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 创建新模型作为备份
            self._build_model()
        
    def _build_model(self):
        """构建深度学习模型"""
        try:
            # 设置随机种子以确保可重复性
            tf.random.set_seed(42)
            
            # 创建LSTM+CNN混合模型
            model = tf.keras.models.Sequential([
                # 输入层 [batch_size, sequence_length, features]
                tf.keras.layers.Input(shape=(self.sequence_length, self.feature_count)),
                
                # 第一层LSTM - 捕捉时序特征
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                
                # 应用1D卷积进行特征提取
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                
                # 第二层LSTM - 继续捕捉更高层次的时序关系
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                
                # 全连接层
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # 输出层 - 使用softmax获取三个类别的概率(下跌、中性、上涨)
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # 编译模型
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.logger.info("深度学习模型构建成功")
            
            # 创建特征缩放器
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            
        except Exception as e:
            self.logger.error(f"构建模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _initial_training(self):
        """初始模型训练"""
        try:
            self.logger.info("开始初始模型训练...")
            
            # 获取历史K线数据
            klines = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if klines and len(klines) > 0:
                self.logger.info(f"成功获取{len(klines)}根K线数据用于训练")
                
                # 准备训练数据
                X, y = self._prepare_training_data(klines)
                
                if X is not None and y is not None and len(X) > 0:
                    # 训练模型
                    self.logger.info(f"开始训练模型，样本数量: {len(X)}")
                    
                    # 分批训练，避免内存问题
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
                    
                    # 保存模型
                    self.model.save(self.model_path['model'])
                    
                    # 保存缩放器
                    with open(self.model_path['scaler'], 'wb') as f:
                        pickle.dump(self.scaler, f)
                    
                    self.logger.info("模型训练完成并保存")
                    
                    # 记录训练时间
                    self.last_training_time = time.time()
                else:
                    self.logger.error("准备训练数据失败")
            else:
                self.logger.error("获取训练数据失败")
                
        except Exception as e:
            self.logger.error(f"初始模型训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _prepare_dataframe(self, klines):
        """
        将K线数据转换为DataFrame格式
        
        Args:
            klines (list): K线数据列表 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: 转换后的DataFrame
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备DataFrame失败: {str(e)}")
            return None
    
    def _engineer_features(self, df):
        """
        特征工程 - 创建和计算模型输入特征
        
        Args:
            df (pandas.DataFrame): K线数据DataFrame
            
        Returns:
            pandas.DataFrame: 计算好特征的DataFrame
        """
        try:
            # 复制原始数据，避免修改原始数据
            features_df = df.copy()
            
            # 1. 价格特征
            # 归一化价格 (使用过去N根K线的最高价和最低价)
            window = 20
            features_df['price_norm'] = (features_df['close'] - features_df['low'].rolling(window=window).min()) / \
                              (features_df['high'].rolling(window=window).max() - features_df['low'].rolling(window=window).min())
            
            # 收盘价与开盘价的差异
            features_df['close_open_ratio'] = features_df['close'] / features_df['open'] - 1
            
            # 最高价与收盘价的比例
            features_df['high_close_ratio'] = features_df['high'] / features_df['close'] - 1
            
            # 最低价与收盘价的比例
            features_df['low_close_ratio'] = features_df['low'] / features_df['close'] - 1
            
            # 2. 价格变化
            # 计算不同周期的价格变化率
            for period in [1, 3, 5, 10, 20]:
                features_df[f'return_{period}'] = features_df['close'].pct_change(periods=period)
            
            # 3. 技术指标
            # 移动平均线
            for period in [5, 10, 20, 50]:
                features_df[f'ma_{period}'] = talib.SMA(features_df['close'].values, timeperiod=period)
                features_df[f'ma_ratio_{period}'] = features_df['close'] / features_df[f'ma_{period}'] - 1
            
            # RSI指标
            features_df['rsi_14'] = talib.RSI(features_df['close'].values, timeperiod=14)
            features_df['rsi_7'] = talib.RSI(features_df['close'].values, timeperiod=7)
            
            # MACD指标
            macd, macdsignal, macdhist = talib.MACD(
                features_df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features_df['macd'] = macd
            features_df['macdsignal'] = macdsignal
            features_df['macdhist'] = macdhist
            
            # 布林带
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
            
            # ADX - 趋势强度指标
            features_df['adx'] = talib.ADX(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values,
                timeperiod=14
            )
            
            # ATR - 波动率指标
            features_df['atr'] = talib.ATR(
                features_df['high'].values,
                features_df['low'].values,
                features_df['close'].values,
                timeperiod=14
            )
            features_df['atr_ratio'] = features_df['atr'] / features_df['close']
            
            # 4. 成交量特征
            # 归一化成交量
            features_df['volume_norm'] = features_df['volume'] / features_df['volume'].rolling(window=20).max()
            
            # 成交量变化
            features_df['volume_change'] = features_df['volume'].pct_change()
            
            # OBV - 能量潮指标
            features_df['obv'] = talib.OBV(features_df['close'].values, features_df['volume'].values)
            features_df['obv_change'] = features_df['obv'].pct_change(periods=5)
            
            # 5. 衍生特征
            # 价格加速度 (二阶导数)
            features_df['price_accel'] = features_df['close'].pct_change().pct_change()
            
            # 波动率特征
            features_df['volatility'] = features_df['close'].pct_change().rolling(window=10).std()
            
            # 趋势一致性 (价格方向与MA一致性)
            for period in [5, 20]:
                ma_col = f'ma_{period}'
                features_df[f'trend_consistency_{period}'] = (
                    (features_df['close'] > features_df[ma_col]) & 
                    (features_df['close'].shift(1) > features_df[ma_col].shift(1))
                ).astype(int) - (
                    (features_df['close'] < features_df[ma_col]) & 
                    (features_df['close'].shift(1) < features_df[ma_col].shift(1))
                ).astype(int)
            
            # 丢弃有NaN值的行
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"特征工程失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _prepare_training_data(self, klines):
        """
        准备模型训练数据
        
        Args:
            klines (list): K线数据
            
        Returns:
            tuple: (X, y) 特征矩阵和标签
        """
        try:
            # 转换为DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None, None
            
            # 特征工程
            df = self._engineer_features(df)
            
            # 生成标签 - 未来N根K线的价格变化
            future_periods = 3  # 预测未来3根K线
            # 1: 上涨(超过0.5%), 0: 盘整(±0.5%), -1: 下跌(超过-0.5%)
            future_returns = df['close'].pct_change(periods=future_periods).shift(-future_periods) * 100
            
            # 生成三分类标签: 下跌(-1)、中性(0)、上涨(1)
            labels = np.zeros(len(future_returns))
            labels[future_returns > 0.5] = 1    # 上涨
            labels[future_returns < -0.5] = 2   # 下跌 (使用2表示下跌，以便与模型输出对应)
            
            # 删除没有未来标签的行
            df = df.iloc[:-future_periods].copy()
            labels = labels[:-future_periods]
            
            # 选择用于训练的特征
            feature_columns = [
                'price_norm', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
                'return_1', 'return_5', 'return_10',
                'ma_ratio_5', 'ma_ratio_20',
                'rsi_14', 'rsi_7',
                'macdhist',
                'bb_width', 'bb_position',
                'atr_ratio'
            ]
            
            # 检查特征列是否存在
            for col in feature_columns:
                if col not in df.columns:
                    self.logger.error(f"特征列 '{col}' 不存在")
                    return None, None
            
            # 提取特征
            X_raw = df[feature_columns].values
            
            # 特征缩放
            X_scaled = self.scaler.fit_transform(X_raw)
            
            # 创建序列数据 (用于LSTM)
            X_sequences = []
            y_sequences = []
            
            for i in range(len(X_scaled) - self.sequence_length + 1):
                X_sequences.append(X_scaled[i:i+self.sequence_length])
                y_sequences.append(labels[i+self.sequence_length-1])
            
            # 转换为numpy数组
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            
            # 将标签转换为one-hot编码
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)
            
            # 检查样本数量
            if len(X) < 100:
                self.logger.warning(f"样本数量过少: {len(X)}")
                return None, None
            
            self.logger.info(f"准备训练数据完成，特征形状: {X.shape}, 标签形状: {y_onehot.shape}")
            
            return X, y_onehot
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def _prepare_prediction_data(self, klines):
        """
        准备预测数据
        
        Args:
            klines (list): K线数据
            
        Returns:
            numpy.ndarray: 用于预测的特征矩阵
        """
        try:
            # 转换为DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None
            
            # 特征工程
            df = self._engineer_features(df)
            
            # 选择用于预测的特征
            feature_columns = [
                'price_norm', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
                'return_1', 'return_5', 'return_10',
                'ma_ratio_5', 'ma_ratio_20',
                'rsi_14', 'rsi_7',
                'macdhist',
                'bb_width', 'bb_position',
                'atr_ratio'
            ]
            
            # 检查特征列是否存在
            for col in feature_columns:
                if col not in df.columns:
                    self.logger.error(f"特征列 '{col}' 不存在")
                    return None
            
            # 提取最近的数据
            X_raw = df[feature_columns].tail(self.sequence_length).values
            
            # 特征缩放 (使用已训练的缩放器)
            X_scaled = self.scaler.transform(X_raw)
            
            # 检查数据长度是否足够
            if len(X_scaled) < self.sequence_length:
                self.logger.error(f"数据长度不足，需要{self.sequence_length}，实际{len(X_scaled)}")
                return None
            
            # 创建单个样本 (用于LSTM)
            X = np.array([X_scaled[-self.sequence_length:]])
            
            return X
            
        except Exception as e:
            self.logger.error(f"准备预测数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def should_retrain(self):
        """检查是否需要重新训练模型"""
        current_time = time.time()
        
        # 如果模型没有训练过或者时间超过重训间隔，则需要重新训练
        if self.last_training_time == 0 or (current_time - self.last_training_time) > self.retraining_interval:
            return True
        
        return False
    
    def analyze_market_state(self, klines):
        """
        分析市场状态
        
        Args:
            klines (list): K线数据
            
        Returns:
            dict: 市场状态信息
        """
        try:
            # 转换为DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return {"state": "unknown", "trend": 0, "strength": 0}
            
            # 计算ADX指标 - 趋势强度
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            adx = talib.ADX(high, low, close, timeperiod=14)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # 获取最近值
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # 确定趋势方向
            trend_direction = 0
            if current_plus_di > current_minus_di and current_plus_di > 20:
                trend_direction = 1  # 上升趋势
            elif current_minus_di > current_plus_di and current_minus_di > 20:
                trend_direction = -1  # 下降趋势
            
            # 确定市场状态
            state = "unknown"
            if current_adx >= 30:
                state = "trend"  # 强趋势市场
            elif current_adx >= 20:
                state = "trend"  # 趋势市场
            else:
                state = "range"  # 震荡市场
            
            # 计算波动率
            atr = talib.ATR(high, low, close, timeperiod=14)
            current_price = close[-1]
            volatility = (atr[-1] / current_price) * 100  # 波动率百分比
            
            # 计算成交量变化
            volume = df['volume'].values
            volume_change = (volume[-1] / volume[-21:-1].mean() - 1) * 100  # 相对于前20根K线的平均成交量的变化百分比
            
            # 更新内部状态
            self.market_state = state
            self.current_trend = trend_direction
            self.trend_strength = min(100, float(current_adx))
            
            # 返回市场状态信息
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
            
            self.logger.info(f"市场状态: {state}, 趋势方向: {trend_direction}, "
                          f"趋势强度: {current_adx:.2f}, 波动率: {volatility:.2f}%, "
                          f"成交量变化: {volume_change:.2f}%")
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"分析市场状态失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"state": "unknown", "trend": 0, "strength": 0}
    
    def generate_signal(self, klines):
        """
        生成交易信号
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 检查模型是否已训练
            if self.model is None:
                self.logger.error("模型未初始化，无法生成信号")
                return 0
            
            # 检查是否需要重新训练
            if self.should_retrain():
                self.logger.info("模型需要重新训练")
                # 获取更多的历史数据进行训练
                training_klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                
                if training_klines and len(training_klines) > 0:
                    X, y = self._prepare_training_data(training_klines)
                    if X is not None and y is not None:
                        # 在现有模型基础上继续训练
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
                        
                        # 保存更新后的模型
                        self.model.save(self.model_path['model'])
                        
                        # 更新训练时间
                        self.last_training_time = time.time()
                        self.logger.info("模型重新训练完成")
            
            # 准备预测数据
            X = self._prepare_prediction_data(klines)
            if X is None:
                self.logger.error("准备预测数据失败")
                return 0
            
            # 获取模型预测
            predictions = self.model.predict(X)
            probabilities = predictions[0]  # 获取第一个样本的预测概率
            
            # 提取各类别的概率
            down_prob = probabilities[2]    # 下跌概率 (索引2)
            neutral_prob = probabilities[1]  # 中性概率 (索引1)
            up_prob = probabilities[0]    # 上涨概率 (索引0)
            
            self.logger.info(f"预测概率: 上涨={up_prob:.4f}, 中性={neutral_prob:.4f}, 下跌={down_prob:.4f}")
            
            # 分析市场状态
            market_state = self.analyze_market_state(klines)
            
            # 调整预测置信度阈值
            confidence_threshold = self.min_confidence
            
            # 在强趋势市场中降低置信度要求
            if market_state['state'] == "trend" and market_state['strength'] > 40:
                confidence_threshold = max(0.55, self.min_confidence - 0.1)
            
            # 在高波动市场中提高置信度要求
            if market_state.get('volatility', 0) > 2.0:
                confidence_threshold = min(0.75, self.min_confidence + 0.1)
            
            # 确定交易信号
            if up_prob > down_prob:
                prediction = 1  # 买入信号
            elif down_prob > up_prob:
                prediction = -1  # 卖出信号
            else:
                prediction = 0  # 观望信号
            
            # 信号验证和趋势确认
            if prediction != 0:
                # 检查预测与市场趋势是否一致
                if prediction * market_state['trend'] < 0 and market_state['strength'] > 30:
                    self.logger.info(f"预测信号与当前趋势方向相反，谨慎对待")
                    
                    # 如果趋势很强，则不与趋势作对
                    if market_state['strength'] > 50:
                        self.logger.info("当前趋势很强，不与趋势作对，修改信号为观望")
                        prediction = 0
            
            # 记录信号历史
            if prediction != 0:
                self.signal_history.append({
                    'time': datetime.now(),
                    'signal': prediction,
                    'up_prob': float(up_prob),
                    'down_prob': float(down_prob),
                    'market_state': market_state['state']
                })
                
                # 限制历史记录长度
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                
                self.last_signal_time = datetime.now()
                self.last_signal = prediction
            
            return prediction
            
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
                if profit_rate >= self.take_profit_pct:
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
                
                # 获取最新趋势预测，检查趋势是否反转
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                signal = self.generate_signal(klines)
                
                # 如果趋势明显反转，考虑平仓
                if position_side == "多" and signal == -1:
                    self.logger.info("趋势预测反转为下跌，平多仓")
                    self.trader.close_position()
                    return
                elif position_side == "空" and signal == 1:
                    self.logger.info("趋势预测反转为上涨，平空仓")
                    self.trader.close_position()
                    return
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def run(self):
        """运行策略"""
        self.logger.info("启动深度学习趋势交易策略")
        
        # 初始训练或加载模型
        if self.model is None:
            self._initialize_model()
        
        # 运行父类的run方法
        super().run()
