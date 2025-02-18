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
import time

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
        
        # ROI跟踪参数
        self.initial_capital = 10000  # 初始资金
        self.total_pnl = 0  # 总盈亏
        self.trades_count = 0  # 交易次数
        self.winning_trades = 0  # 盈利交易次数
        self.roi_log_interval = 60  # ROI日志记录间隔(秒)
        self.last_roi_log_time = time.time()
        
        self.last_signal = 0
        self.signal_count = 0
        self.optimal_params = {
            'signal_threshold': [1, 2, 3],
            'max_sustain': [3, 5, 7]
        }
        
        self._verify_tf_config()
        self.build_model()
        self.last_position = None
    
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
        
        # 改进的模型结构
        self.model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(self.lookback_period, self.feature_count)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        # 使用固定学习率的Adam优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
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
            
            # 创建标签
            future_returns = df['close'].pct_change(-1).shift(1)  # 使用未来收益率
            y = np.zeros(len(future_returns))
            y[future_returns > 0.001] = 1  # 上涨信号
            y[future_returns < -0.001] = 0  # 下跌信号
            y = y[:-1]  # 移除最后一个无效数据
            
            # 创建序列数据
            X_seq = self.create_sequences(X[:-1])
            y = y[self.lookback_period:]  # 调整标签长度匹配序列
            
            if len(X_seq) != len(y):
                raise ValueError(f"数据维度不匹配: X shape = {X_seq.shape}, y length = {len(y)}")
            
            # 数据集分割
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # 定义回调函数
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.0001
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # 评估模型性能
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)[:2]
            self.logger.info(f"模型训练完成 - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练过程中发生错误: {str(e)}")
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
            
            self.logger.info(f"prediction:{prediction:.4f}")

            # 添加预测置信度检查
            prediction_confidence = abs(prediction - 0.5)
            if prediction_confidence < 0.1:  # 如果预测置信度过低
                self.logger.info("预测置信度过低，建议观望")
                return 0.5
                
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {str(e)}")
            return 0.5

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

    def calculate_roi(self):
        """计算并记录ROI指标"""
        try:
            current_time = time.time()
            position = self.trader.get_position()
            
            # 计算已实现盈亏
            realized_pnl = self.total_pnl
            
            # 计算未实现盈亏
            unrealized_pnl = 0
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    entry_price = float(position['info'].get('entryPrice', 0))
                    current_price = self.trader.get_market_price()
                    unrealized_pnl = (current_price - entry_price) * abs(position_amt)
                    if position_amt < 0:  # 如果是空仓，盈亏需要取反
                        unrealized_pnl = -unrealized_pnl
            
            # 计算总ROI
            total_pnl = realized_pnl + unrealized_pnl
            roi = (total_pnl / self.initial_capital) * 100
            
            # 计算胜率
            win_rate = (self.winning_trades / self.trades_count * 100) if self.trades_count > 0 else 0
            
            # 定期记录ROI
            if current_time - self.last_roi_log_time >= self.roi_log_interval:
                self.logger.info(
                    f"ROI统计 - 总收益率: {roi:.2f}% | "
                    f"已实现盈亏: {realized_pnl:.2f} | "
                    f"未实现盈亏: {unrealized_pnl:.2f} | "
                    f"交易次数: {self.trades_count} | "
                    f"胜率: {win_rate:.2f}%"
                )
                self.last_roi_log_time = current_time
            
            return roi, realized_pnl, unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"计算ROI时发生错误: {str(e)}")
            return 0, 0, 0

    def update_trade_stats(self, pnl):
        """更新交易统计"""
        self.trades_count += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

    def on_position_closed(self, position, close_price):
        """处理持仓平仓事件"""
        try:
            if position is None:
                return
                
            # 计算本次交易的盈亏
            position_amt = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            pnl = (close_price - entry_price) * abs(position_amt)
            if position_amt < 0:  # 如果是空仓，盈亏需要取反
                pnl = -pnl
                
            # 更新交易统计
            self.update_trade_stats(pnl)
            
            # 记录本次交易的ROI
            initial_margin = float(position['info'].get('initialMargin', 0))
            if initial_margin > 0:
                trade_roi = (pnl / initial_margin) * 100
                self.logger.info(
                    f"交易结束 - 盈亏: {pnl:.2f} USDT | "
                    f"收益率: {trade_roi:.2f}% | "
                    f"开仓价: {entry_price:.2f} | "
                    f"平仓价: {close_price:.2f} | "
                    f"仓位大小: {abs(position_amt):.4f}"
                )
                
        except Exception as e:
            self.logger.error(f"处理平仓事件时发生错误: {str(e)}")

    def generate_signals(self, df):
        """优化的信号生成逻辑"""
        try:
            if len(df) < 50:
                return 0
            
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果有持仓被平掉，处理平仓事件
            if self.last_position is not None and (position is None or float(position['info'].get('positionAmt', 0)) == 0):
                close_price = df['close'].iloc[-1]
                self.on_position_closed(self.last_position, close_price)
            
            # 更新上一次持仓状态
            self.last_position = position
            
            # 更新ROI统计
            self.calculate_roi()
            
            # 降低训练频率，每50根K线重新训练一次
            if len(df) % 50 == 0:
                self.train_model(df)
            
            # 获取预测
            prediction = self.predict(df)
            self.logger.info(f"AI预测值: {prediction:.4f}")

            # 检查风控条件
            if not self.check_risk_conditions(df, position):
                return 0
            
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
