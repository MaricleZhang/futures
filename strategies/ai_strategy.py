import numpy as np
import pandas as pd
import tensorflow as tf
import os
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
        
        self.model = None
        self.scaler = StandardScaler()
        self.lookback_period = 60
        
        # 信号控制参数
        self.signal_threshold = 3
        self.max_sustain = 10
        self.volatility_window = 20
        
        self.last_signal = 0
        self.signal_count = 0
        self.optimal_params = {
            'signal_threshold': [2, 3, 4],
            'max_sustain': [5, 10, 15]
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
        """构建LSTM模型"""
        # 配置GPU内存
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU配置错误: {e}")
        
        # 原模型构建代码
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 7)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def prepare_features(self, df):
        """准备特征数据"""
        # 计算技术指标
        df['RSI'] = talib.RSI(df['close'])
        df['MACD'], df['MACD_SIGNAL'], _ = talib.MACD(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # 计算价格变化百分比
        df['price_change'] = df['close'].pct_change()
        
        # 计算成交量变化
        df['volume_change'] = df['volume'].pct_change()
        
        # 使用前向填充处理NaN值
        df = df.ffill()
        # 如果还有NaN值（比如第一行），使用后向填充
        df = df.bfill()
        
        # 选择特征
        features = ['close', 'volume', 'RSI', 'MACD', 'ATR', 'price_change', 'volume_change']
        X = df[features].values
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        return X

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
            
            # 创建训练标签（1表示上涨，0表示下跌）
            y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]  # 去掉最后一个没有下一个价格的数据点
            
            # 创建序列数据
            X_seq = self.create_sequences(X[:-1])  # 去掉最后一个数据点，因为它没有对应的y值
            
            # 确保X和y的长度匹配
            y = y[self.lookback_period:]  # 从lookback_period开始取y值
            
            if len(X_seq) != len(y):
                raise ValueError(f"数据维度不匹配: X shape = {X_seq.shape}, y length = {len(y)}")
            
            # 训练模型
            self.model.fit(X_seq, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
        except Exception as e:
            print(f"训练模型时发生错误: {str(e)}")
            raise

    @tf.function(experimental_compile=True)
    def async_predict(self, input_data):
        return self.model(input_data)
    
    def predict(self, df):
        try:
            features = self.prepare_features(df)
            sequence = self.create_sequences(features)[-1]
            prediction = self.async_predict(np.expand_dims(sequence, axis=0))
            return float(prediction.numpy()[0][0])
        except Exception as e:
            print(f"预测过程中发生错误: {str(e)}")
            return 0.5

    def validate_signal(self, new_signal, df):
        """多维信号验证
        Args:
            new_signal (int): 原始信号
            df (DataFrame): 包含OHLCV数据
        Returns:
            int: 验证后的信号
        """
        # 动量过滤
        momentum = df['close'].iloc[-5:].pct_change().mean()
        
        # 量能验证
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        
        # 趋势过滤
        if new_signal > 0 and df['close'].iloc[-1] < df['ma20'].iloc[-1]:
            return 0
        if new_signal < 0 and df['close'].iloc[-1] > df['ma20'].iloc[-1]:
            return 0
            
        # 波动率过滤
        if abs(momentum) < 0.005 and volume_ma < 1e5:
            return 0
            
        return new_signal

    def calculate_volatility(self, df):
        """计算波动率"""
        return df['close'].rolling(self.volatility_window).std().iloc[-1] / df['close'].iloc[-1]

    def calculate_position_size(self, portfolio_value, volatility):
        """基于波动率的头寸控制"""
        risk_per_trade = 0.02  # 2%风险敞口
        return (portfolio_value * risk_per_trade) / volatility

    def generate_signals(self, df):
        """生成交易信号"""
        try:
            # 确保数据足够训练
            if len(df) < 200:  # 增加最小所需历史数据量
                return 0
            
            # 降低训练频率，每300根K线重新训练一次
            if len(df) % 300 == 0:
                self.train_model(df)
            
            # 获取预测
            prediction = self.predict(df)
            self.logger.info(f"AI预测值: {prediction:.4f}")
            
            # 使用更严格的信号阈值，并添加信号持续性检查
            new_signal = 0
            if prediction > 0.8:  # 提高做多阈值
                new_signal = 1
                self.logger.info("预测看多信号")
            elif prediction < 0.2:  # 降低做空阈值
                new_signal = -1
                self.logger.info("预测看空信号")
            else:
                self.logger.info("预测观望信号")
                
            # 动态调整阈值
            current_volatility = self.calculate_volatility(df)
            self.signal_threshold = 3 if current_volatility < 0.02 else 2
            
            # 验证信号有效性
            validated_signal = self.validate_signal(new_signal, df)
            
            if validated_signal != 0:
                self.signal_count += 1
            else:
                self.signal_count = max(0, self.signal_count - 1)
            
            # 信号持续控制
            if self.signal_threshold <= self.signal_count <= self.max_sustain:
                self.last_signal = validated_signal
            elif self.signal_count > self.max_sustain:
                self.signal_count = 0
                self.last_signal = 0
            
            return self.last_signal
            
        except Exception as e:
            print(f"生成信号时发生错误: {str(e)}")
            return 0  # 发生错误时不操作

    def optimize_parameters(self, historical_data):
        best_sharpe = -np.inf
        for threshold in self.optimal_params['signal_threshold']:
            for sustain in self.optimal_params['max_sustain']:
                self.signal_threshold = threshold
                self.max_sustain = sustain
                returns = self.backtest_signal_logic(historical_data).diff()
                sharpe = np.mean(returns) / np.std(returns)
                if sharpe > best_sharpe:
                    best_params = (threshold, sustain)
        return best_params

    def backtest_signal_logic(self, historical_data):
        """信号逻辑回溯测试
        Args:
            historical_data (DataFrame): 完整历史数据
        Returns:
            Series: 信号序列
        """
        signals = pd.Series(index=historical_data.index)
        
        for i in range(self.lookback_period, len(historical_data)):
            df_window = historical_data.iloc[:i]
            try:
                signals.iloc[i] = self.generate_signals(df_window)
            except Exception as e:
                print(f"在{i}处回测失败: {str(e)}")
                signals.iloc[i] = 0
        
        return signals
