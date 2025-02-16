import numpy as np
import pandas as pd
import tensorflow as tf
import os
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from .base_strategy import BaseStrategy
import talib

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.WARNING)

# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 配置线程数以优化CPU性能
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

class AIStrategy(BaseStrategy):
    def __init__(self, trader):
        """初始化AI策略"""
        super().__init__(trader)
        
        # 初始化日志记录器
        from utils.logger import Logger
        self.logger = Logger.get_logger()
        
        # 配置TensorFlow环境
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
        # 清理之前的模型和会话
        tf.keras.backend.clear_session()
        
        self.model = None
        self.scaler = StandardScaler()
        self.lookback_period = 20  # 进一步减少回看周期
        
        # 信号控制参数
        self.signal_threshold = 1.5  # 进一步降低信号阈值
        self.max_sustain = 3  # 进一步减少最大持仓时间
        self.volatility_window = 5  # 进一步缩短波动率计算窗口
        
        # 新增参数
        self.min_profit_target = 0.001  # 最小获利目标
        self.max_loss_target = 0.002  # 最大损失目标
        self.trend_window = 3  # 趋势窗口
        
        self.last_signal = 0
        self.signal_count = 0
        self.optimal_params = {
            'signal_threshold': [1, 2, 3],
            'max_sustain': [3, 5, 7]
        }
        
        self._verify_tf_config()
        self.build_model()
    
    def _verify_tf_config(self):
        """验证TensorFlow配置"""
        current_policy = tf.keras.mixed_precision.global_policy()
        if current_policy.name != 'mixed_float16':
            print(f"警告: 混合精度未启用，当前策略: {current_policy.name}")
        
        inter_threads = tf.config.threading.get_inter_op_parallelism_threads()
        intra_threads = tf.config.threading.get_intra_op_parallelism_threads()
        print(f"TensorFlow线程配置: inter={inter_threads}, intra={intra_threads}")
        
        log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '未设置')
        print(f"TensorFlow日志级别: {log_level}")
    
    def build_model(self):
        """构建改进的LSTM模型"""
        # 清理之前的模型和会话
        if self.model is not None:
            del self.model
        tf.keras.backend.clear_session()
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU配置错误: {e}")
        
        # 确保特征数量已初始化
        if not hasattr(self, 'feature_count'):
            # 创建一个临时DataFrame来获取特征数量
            temp_df = pd.DataFrame({'close': [0]*30, 'volume': [0]*30,
                                  'high': [0]*30, 'low': [0]*30})
            self.prepare_features(temp_df)
        
        # 优化后的模型结构
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.lookback_period, self.feature_count)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        # 使用Adam优化器，使用learning_rate_schedule代替decay
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def prepare_features(self, df):
        """准备增强的特征数据"""
        try:
            # 基础技术指标
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
            df['MACD'], df['MACD_SIGNAL'], _ = talib.MACD(df['close'])
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # 短线交易相关指标
            df['EMA3'] = talib.EMA(df['close'], timeperiod=3)
            df['EMA5'] = talib.EMA(df['close'], timeperiod=5)
            df['EMA8'] = talib.EMA(df['close'], timeperiod=8)
            df['EMA_CROSS_SHORT'] = (df['EMA3'] > df['EMA5']).astype(int)
            df['EMA_CROSS_MID'] = (df['EMA5'] > df['EMA8']).astype(int)
            
            # KDJ指标
            df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['K-D'] = df['slowk'] - df['slowd']
            
            # 布林带
            df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=5)
            df['BB_WIDTH'] = (df['upper'] - df['lower']) / df['middle']
            
            # 动量指标
            df['ROC'] = talib.ROC(df['close'], timeperiod=3)
            df['MOM'] = talib.MOM(df['close'], timeperiod=3)
            
            # 价格和成交量变化
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['price_acceleration'] = df['price_change'].diff()
            
            # 波动率指标
            df['volatility_short'] = df['close'].rolling(window=5).std() / df['close']
            df['volatility_mid'] = df['close'].rolling(window=10).std() / df['close']
            
            # 使用前向填充处理NaN值
            df = df.ffill()
            df = df.bfill()
            
            # 选择特征
            features = ['close', 'volume', 'RSI', 'MACD', 'ATR', 
                       'EMA_CROSS_SHORT', 'EMA_CROSS_MID', 'K-D', 
                       'BB_WIDTH', 'ROC', 'MOM', 'price_change', 
                       'volume_change', 'price_acceleration',
                       'volatility_short', 'volatility_mid']
            
            # 确保特征数量一致性
            self.feature_count = len(features)
            
            X = df[features].values
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            return X
            
        except Exception as e:
            self.logger.error(f"特征准备过程中发生错误: {str(e)}")
            raise

    def create_sequences(self, data):
        """创建序列数据"""
        X = []
        for i in range(len(data) - self.lookback_period):
            X.append(data[i:(i + self.lookback_period)])
        return np.array(X)

    def train_model(self, df):
        """训练模型"""
        try:
            # 准备特征
            X = self.prepare_features(df)
            
            # 创建更复杂的训练标签
            future_returns = df['close'].shift(-1) / df['close'] - 1
            y = np.zeros(len(future_returns))
            y[future_returns > 0.001] = 1  # 上涨信号阈值
            y[future_returns < -0.001] = 0  # 下跌信号阈值
            y = y[:-1]  # 移除最后一个无效数据
            
            # 创建序列数据
            X_seq = self.create_sequences(X[:-1])
            
            # 确保X和y的长度匹配
            y = y[self.lookback_period:]
            
            if len(X_seq) != len(y):
                raise ValueError(f"数据维度不匹配: X shape = {X_seq.shape}, y length = {len(y)}")
            
            # 添加早停机制
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            )
            
            # 添加学习率调整
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                cooldown=2
            )
            
            # 添加模型检查点
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=0
            )
            
            # 训练模型
            history = self.model.fit(
                X_seq, y,
                epochs=100,  # 增加训练轮数
                batch_size=64,  # 增加batch size
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=0
            )
            
            # 记录训练结果
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            
            self.logger.info(f"模型训练完成 - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # 验证模型性能
            if final_accuracy < 0.55 or val_accuracy < 0.53:
                self.logger.warning("模型准确率过低，考虑重新训练或调整参数")
                return False
            
            return True
                
        except Exception as e:
            self.logger.error(f"训练模型时发生错误: {str(e)}")
            return False

    @tf.function(experimental_compile=True)
    def async_predict(self, input_data):
        return self.model(input_data)
    
    def predict(self, df):
        """优化的预测函数"""
        try:
            # 准备特征
            X = self.prepare_features(df)
            
            # 创建序列数据
            X_seq = self.create_sequences(X)
            
            # 确保有足够的数据
            if len(X_seq) == 0:
                return 0.5
            
            # 获取最新的序列数据
            latest_seq = X_seq[-1:]
            
            # 进行预测
            prediction = self.model.predict(latest_seq, verbose=0)[0][0]
            
            # 添加预测置信度检查
            prediction_confidence = abs(prediction - 0.5)
            if prediction_confidence < 0.1:  # 如果预测置信度过低
                self.logger.info("预测置信度过低，建议观望")
                return 0.5
                
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {str(e)}")
            return 0.5

    def validate_signal(self, new_signal, df):
        """增强的多维信号验证"""
        try:
            # 成交量验证
            volume_ma = df['volume'].rolling(window=5).mean()
            if df['volume'].iloc[-1] < volume_ma.iloc[-1] * 0.8:
                self.logger.info("成交量不足，避免交易")
                return 0
            
            # 计算波动率和动态调整信号阈值
            volatility = df['close'].pct_change().std() * np.sqrt(252)
            self.signal_threshold = max(1.2, min(2.0, volatility * 10))
            
            # 趋势确认
            ema9 = talib.EMA(df['close'], timeperiod=9)
            ema21 = talib.EMA(df['close'], timeperiod=21)
            trend_signal = 1 if ema9.iloc[-1] > ema21.iloc[-1] else -1
            
            if new_signal * trend_signal < 0:
                self.logger.info("信号与趋势相反，避免交易")
                return 0
            
            # 计算超短期趋势
            ultra_short_trend = df['close'].iloc[-3:].pct_change().mean()
            short_trend = df['close'].iloc[-5:].pct_change().mean()
            
            # RSI过滤
            rsi = talib.RSI(df['close']).iloc[-1]
            if new_signal > 0 and rsi > 70:
                self.logger.info("RSI超买，避免做多")
                return 0
            if new_signal < 0 and rsi < 30:
                self.logger.info("RSI超卖，避免做空")
                return 0
            
            # MACD验证
            macd, signal, _ = talib.MACD(df['close'])
            macd_cross = macd.iloc[-1] - signal.iloc[-1]
            
            if new_signal > 0 and macd_cross < 0:
                self.logger.info("MACD死叉，避免做多")
                return 0
            if new_signal < 0 and macd_cross > 0:
                self.logger.info("MACD金叉，避免做空")
                return 0
            
            return new_signal
            
        except Exception as e:
            self.logger.error(f"信号验证过程中发生错误: {str(e)}")
            return 0

    def calculate_volatility(self, df):
        """计算市场波动率"""
        try:
            # 使用过去20根K线计算波动率
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            current_volatility = volatility.iloc[-1]
            return current_volatility
        except Exception as e:
            self.logger.error(f"计算波动率时发生错误: {str(e)}")
            return 0.01  # 返回默认值

    def check_risk_conditions(self, df, position=None):
        """检查风控条件"""
        try:
            # 计算当前波动率
            current_volatility = self.calculate_volatility(df)
            
            # 获取最新价格
            current_price = df['close'].iloc[-1]
            
            # 1. 波动率条件
            volatility_threshold = 0.02  # 2%的波动率阈值
            if current_volatility > volatility_threshold:
                self.logger.info(f"波动率({current_volatility:.4f})超过阈值({volatility_threshold:.4f})，暂停交易")
                return False
                
            # 2. 价格剧烈波动条件
            price_change = df['close'].pct_change()
            if abs(price_change.iloc[-1]) > 0.03:  # 3%的价格变动阈值
                self.logger.info(f"价格变动({price_change.iloc[-1]:.4f})过大，暂停交易")
                return False
                
            # 3. 趋势强度检查
            ema_short = talib.EMA(df['close'], timeperiod=5)
            ema_long = talib.EMA(df['close'], timeperiod=20)
            trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
            if trend_strength < 0.001:  # 0.1%的趋势强度阈值
                self.logger.info("趋势强度不足，避免交易")
                return False
                
            # 4. 成交量条件
            volume_ma = df['volume'].rolling(window=20).mean()
            if df['volume'].iloc[-1] < volume_ma.iloc[-1] * 0.5:  # 成交量低于均值的50%
                self.logger.info("成交量不足，避免交易")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"检查风控条件时发生错误: {str(e)}")
            return False

    def generate_signals(self, df):
        """优化的信号生成逻辑"""
        try:
            if len(df) < 50:
                return 0
            
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 检查风控条件
            if not self.check_risk_conditions(df, position):
                return 0
                
            # 降低训练频率，每50根K线重新训练一次
            if len(df) % 50 == 0:
                self.train_model(df)
            
            # 获取预测
            prediction = self.predict(df)
            self.logger.info(f"AI预测值: {prediction:.4f}")
            
            # 动态调整信号阈值
            current_volatility = self.calculate_volatility(df)
            base_threshold = 0.60
            
            # 根据波动率调整阈值
            if current_volatility < 0.005:  # 低波动
                dynamic_threshold = base_threshold + 0.05
            elif current_volatility > 0.015:  # 高波动
                dynamic_threshold = base_threshold - 0.05
            else:
                dynamic_threshold = base_threshold
                
            # 生成交易信号
            if prediction > dynamic_threshold:
                return 1
            elif prediction < (1 - dynamic_threshold):
                return -1
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"生成信号时发生错误: {str(e)}")
            return 0
