import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import talib
import time
import os
from datetime import datetime
from strategies.base_strategy import BaseStrategy
import config

class DeepLearningStrategy(BaseStrategy):
    """DeepLearningStrategy - 深度学习预测策略
    
    基于LSTM深度学习模型的交易策略，使用1分钟K线数据，
    预测未来15根K线的价格走势，结合多个技术指标和量价特征
    生成交易信号。
    
    特点：
    1. 深度学习：使用LSTM神经网络捕捉价格序列的时间依赖性
    2. 多特征融合：整合OHLCV数据、技术指标和市场情绪指标
    3. 预测未来15根K线：分析短期价格走势并预测转折点
    4. 动态调整：根据预测准确度动态调整交易参数
    """
    
    MODEL_NAME = "DeepLearningStrategy"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化深度学习策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 模型参数
        self.sequence_length = 60  # 输入序列长度（过去的K线数量）
        self.prediction_length = 15  # 预测序列长度（未来的K线数量）
        self.feature_count = 19  # 特征数量
        self.batch_size = 32
        self.epochs = 50
        self.patience = 15  # 早停耐心值
        
        # K线设置
        self.kline_interval = '1m'  # 1分钟K线
        self.training_lookback = 1000  # 训练数据回看周期
        self.check_interval = 60  # 检查间隔(秒)
        self.retraining_interval = 3600  # 1小时重新训练一次
        
        # 策略参数
        self.confidence_threshold = 0.65  # 交易信号置信度阈值
        self.stop_loss_percent = config.DEFAULT_STOP_LOSS_PERCENT / 100
        self.take_profit_percent = config.DEFAULT_TAKE_PROFIT_PERCENT / 100
        self.trend_threshold = 0.005  # 价格变化超过0.5%才视为有趋势
        
        # 模型和状态
        self.model = None
        self.last_training_time = 0
        self.last_signal = 0
        self.model_path = 'models'
        
        # 性能统计
        self.predictions = []
        self.actual_results = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # 创建模型目录
        os.makedirs(self.model_path, exist_ok=True)
        
        # 初始化模型
        self.initialize_model()
        
        # 尝试加载已有模型或训练新模型
        self.load_or_train_model()
    
    def initialize_model(self):
        """初始化LSTM模型"""
        try:
            # 创建LSTM模型
            model = Sequential()
            
            # 第一个LSTM层
            model.add(LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.feature_count)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 第二个LSTM层
            model.add(LSTM(64, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 全连接层
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            # 输出层 - 预测未来15个K线的价格变化率
            model.add(Dense(self.prediction_length))
            
            # 编译模型
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            self.model = model
            self.logger.info("LSTM模型初始化成功")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
    
    def load_or_train_model(self):
        """加载已有模型或训练新模型"""
        model_file = os.path.join(self.model_path, f"{self.trader.symbol}_{self.MODEL_NAME}.h5")
        
        # 尝试加载已有模型
        if os.path.exists(model_file):
            try:
                self.model = load_model(model_file)
                self.logger.info(f"已加载模型: {model_file}")
                return
            except Exception as e:
                self.logger.error(f"加载模型失败: {str(e)}, 将训练新模型")
        
        # 获取历史数据并训练模型
        try:
            self.logger.info("获取历史K线数据...")
            historical_data = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根{self.kline_interval}K线数据")
                if self.train_model(historical_data):
                    self.save_model()
                    self.logger.info("模型训练和保存完成")
                else:
                    self.logger.error("模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
        except Exception as e:
            self.logger.error(f"初始化训练失败: {str(e)}")
    
    def save_model(self):
        """保存模型到文件"""
        if self.model:
            model_file = os.path.join(self.model_path, f"{self.trader.symbol}_{self.MODEL_NAME}.h5")
            try:
                self.model.save(model_file)
                self.logger.info(f"模型已保存至: {model_file}")
            except Exception as e:
                self.logger.error(f"保存模型失败: {str(e)}")
    
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < self.sequence_length + self.prediction_length:
                self.logger.error("K线数据不足，无法准备特征")
                return None, None
            
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算价格变化率
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 计算技术指标
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
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
                nbdevdn=2,
                matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            df['bb_position'] = (df['close'] - lower) / (upper - lower)
            
            # 移动平均线
            df['sma_5'] = talib.SMA(df['close'].values, timeperiod=5)
            df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            
            # 计算均线交叉指标
            df['sma_5_10_cross'] = ((df['sma_5'] > df['sma_10']) & (df['sma_5'].shift(1) <= df['sma_10'].shift(1))).astype(int) - \
                                  ((df['sma_5'] < df['sma_10']) & (df['sma_5'].shift(1) >= df['sma_10'].shift(1))).astype(int)
            
            # 动量指标
            df['momentum_1'] = df['close'] - df['close'].shift(1)
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            
            # 成交量指标
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']
            
            # 计算波动率
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 创建用于训练的特征矩阵和标签
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'log_returns', 
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position',
                'sma_5_10_cross', 
                'momentum_1', 'momentum_5', 'momentum_10',
                'volume_ratio', 'volatility'
            ]
            
            # 确保特征数量与初始化时设置的一致
            assert len(feature_columns) == self.feature_count, f"特征数量不匹配: {len(feature_columns)} vs {self.feature_count}"
            
            # 移除NaN值
            df = df.dropna()
            
            if len(df) < self.sequence_length + self.prediction_length:
                self.logger.error("有效数据不足，无法准备特征")
                return None, None
            
            # 准备特征和标签
            X = []
            y = []
            
            for i in range(len(df) - self.sequence_length - self.prediction_length + 1):
                # 特征序列
                X_sequence = df[feature_columns].iloc[i:i+self.sequence_length].values
                
                # 标签序列 - 未来15个K线的价格变化率
                y_sequence = df['returns'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length].values
                
                X.append(X_sequence)
                y.append(y_sequence)
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            return None, None
    
    def train_model(self, klines):
        """训练深度学习模型"""
        try:
            # 准备特征和标签数据
            X, y = self.prepare_features(klines)
            if X is None or y is None:
                return False
            
            # 检查数据形状
            self.logger.info(f"训练数据形状 - X: {X.shape}, y: {y.shape}")
            
            # 如果数据量不足，不进行训练
            if len(X) < 100:
                self.logger.warning("训练数据不足，需要至少100个样本")
                return False
            
            # 划分训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # 设置早停回调函数
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            )
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # 记录训练结果
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            self.logger.info(f"模型训练完成 - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            
            # 更新最后训练时间
            self.last_training_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def predict_future_prices(self, klines):
        """预测未来价格走势"""
        try:
            if not isinstance(klines, list) or len(klines) < self.sequence_length:
                self.logger.error("K线数据不足，无法进行预测")
                return None
            
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算所有需要的特征
            # 计算价格变化率
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 计算技术指标
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
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
                nbdevdn=2,
                matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            df['bb_position'] = (df['close'] - lower) / (upper - lower)
            
            # 移动平均线
            df['sma_5'] = talib.SMA(df['close'].values, timeperiod=5)
            df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            
            # 计算均线交叉指标
            df['sma_5_10_cross'] = ((df['sma_5'] > df['sma_10']) & (df['sma_5'].shift(1) <= df['sma_10'].shift(1))).astype(int) - \
                                  ((df['sma_5'] < df['sma_10']) & (df['sma_5'].shift(1) >= df['sma_10'].shift(1))).astype(int)
            
            # 动量指标
            df['momentum_1'] = df['close'] - df['close'].shift(1)
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            
            # 成交量指标
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']
            
            # 计算波动率
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 定义特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'log_returns', 
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position',
                'sma_5_10_cross', 
                'momentum_1', 'momentum_5', 'momentum_10',
                'volume_ratio', 'volatility'
            ]
            
            # 移除NaN值
            df = df.dropna()
            
            if len(df) < self.sequence_length:
                self.logger.error("有效数据不足，无法进行预测")
                return None
            
            # 获取最近的特征序列
            X_pred = df[feature_columns].iloc[-self.sequence_length:].values
            
            # 重塑为模型输入形状
            X_pred = X_pred.reshape(1, self.sequence_length, self.feature_count)
            
            # 使用模型预测未来15个K线的价格变化率
            predicted_returns = self.model.predict(X_pred)[0]
            
            # 获取最新价格
            latest_price = df['close'].iloc[-1]
            
            # 计算预测的未来价格
            predicted_prices = [latest_price]
            
            for ret in predicted_returns:
                next_price = predicted_prices[-1] * (1 + ret)
                predicted_prices.append(next_price)
            
            # 移除第一个元素（当前价格）
            predicted_prices = predicted_prices[1:]
            
            # 计算预测趋势
            # 上涨趋势: 1, 下跌趋势: -1, 震荡/盘整: 0
            price_diff = predicted_prices[-1] - latest_price
            price_change_percent = price_diff / latest_price
            
            if price_change_percent > self.trend_threshold:  # 上涨趋势
                trend = 1
                confidence = min(price_change_percent * 100, 1.0)  # 最大置信度为1.0
            elif price_change_percent < -self.trend_threshold:  # 下跌趋势
                trend = -1
                confidence = min(abs(price_change_percent) * 100, 1.0)
            else:  # 震荡
                trend = 0
                confidence = 0.5  # 震荡趋势的置信度默认为0.5
            
            return {
                'predicted_prices': predicted_prices,
                'trend': trend,
                'confidence': confidence,
                'price_change_percent': price_change_percent,
                'latest_price': latest_price
            }
            
        except Exception as e:
            self.logger.error(f"预测未来价格失败: {str(e)}")
            return None
    
    def update_prediction_accuracy(self, prediction, actual_price):
        """更新预测准确性统计"""
        try:
            if prediction is None:
                return
                
            predicted_trend = prediction['trend']
            
            # 计算实际价格变化
            price_change = (actual_price - prediction['latest_price']) / prediction['latest_price']
            
            # 确定实际趋势
            if price_change > self.trend_threshold:
                actual_trend = 1
            elif price_change < -self.trend_threshold:
                actual_trend = -1
            else:
                actual_trend = 0
                
            # 更新统计
            self.total_predictions += 1
            if predicted_trend == actual_trend:
                self.correct_predictions += 1
                
            # 记录预测准确性
            accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
            self.logger.info(f"预测准确性: {accuracy:.4f} ({self.correct_predictions}/{self.total_predictions})")
            
            # 根据预测准确性动态调整置信度阈值
            if self.total_predictions > 20:
                self.confidence_threshold = max(0.5, min(0.8, 1 - accuracy))
                
        except Exception as e:
            self.logger.error(f"更新预测准确性失败: {str(e)}")
    
    def generate_signal(self, klines=None):
        """生成交易信号"""
        try:
            current_time = time.time()
            
            # 检查是否需要重新训练模型
            if current_time - self.last_training_time > self.retraining_interval:
                self.logger.info("模型重新训练时间到，准备重新训练")
                if klines and len(klines) >= self.training_lookback:
                    training_klines = klines[-self.training_lookback:]
                else:
                    training_klines = self.trader.get_klines(
                        interval=self.kline_interval,
                        limit=self.training_lookback
                    )
                
                if training_klines and len(training_klines) > 0:
                    if self.train_model(training_klines):
                        self.save_model()
                        self.logger.info("模型重新训练和保存完成")
                else:
                    self.logger.error("获取训练数据失败，无法重新训练模型")
            
            # 获取最新K线数据（如果未提供）
            if klines is None or len(klines) < self.sequence_length + 20:
                klines = self.trader.get_klines(
                    interval=self.kline_interval,
                    limit=self.sequence_length + 50
                )
            
            if not klines or len(klines) < self.sequence_length:
                self.logger.error("K线数据不足，无法生成信号")
                return 0
            
            # 预测未来价格
            prediction = self.predict_future_prices(klines)
            if prediction is None:
                self.logger.error("价格预测失败，无法生成信号")
                return 0
            
            # 记录预测结果
            self.logger.info(f"预测趋势: {prediction['trend']}, 置信度: {prediction['confidence']:.4f}, "
                           f"预测价格变化: {prediction['price_change_percent']:.4%}")
            
            # 记录上一次的预测，以便后续验证
            self.predictions.append({
                'timestamp': datetime.now().timestamp(),
                'prediction': prediction,
                'signal': None  # 将在生成信号后更新
            })
            
            # 验证之前的预测（如果有）
            if len(self.predictions) > self.prediction_length:
                old_prediction = self.predictions[0]
                time_elapsed = datetime.now().timestamp() - old_prediction['timestamp']
                
                # 如果已经过去了15分钟，验证预测准确性
                if time_elapsed >= 900:  # 15分钟 = 900秒
                    current_price = float(klines[-1][4])  # 当前价格
                    self.update_prediction_accuracy(old_prediction['prediction'], current_price)
                    self.predictions.pop(0)  # 移除已验证的预测
            
            # 获取持仓状态
            position = self.trader.get_position()
            position_size = float(position['info'].get('positionAmt', 0)) if position else 0
            
            # 根据预测结果和置信度生成交易信号
            signal = 0
            if prediction['trend'] == 1 and prediction['confidence'] > self.confidence_threshold:
                # 上涨趋势，生成买入信号
                if position_size <= 0:  # 如果没有多仓或有空仓
                    self.logger.info(f"生成买入信号 - 预测价格上涨: {prediction['price_change_percent']:.4%}, 置信度: {prediction['confidence']:.4f}")
                    signal = 1  # 买入信号
            elif prediction['trend'] == -1 and prediction['confidence'] > self.confidence_threshold:
                # 下跌趋势，生成卖出信号
                if position_size >= 0:  # 如果没有空仓或有多仓
                    self.logger.info(f"生成卖出信号 - 预测价格下跌: {prediction['price_change_percent']:.4%}, 置信度: {prediction['confidence']:.4f}")
                    signal = -1  # 卖出信号
            
            # 更新最后一次预测的信号
            if self.predictions:
                self.predictions[-1]['signal'] = signal
                
            # 记录信号
            self.last_signal = signal
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            return 0
    
    def should_close_position(self, position_info, current_price):
        """判断是否应该平仓"""
        if not position_info:
            return False
            
        entry_price = float(position_info['entryPrice'])
        position_amount = float(position_info['positionAmt'])
        
        # 计算盈亏百分比
        if position_amount > 0:  # 多仓
            profit_percent = (current_price - entry_price) / entry_price
        else:  # 空仓
            profit_percent = (entry_price - current_price) / entry_price
            
        # 止损或止盈
        if profit_percent <= -self.stop_loss_percent:
            self.logger.info(f"触发止损: {profit_percent:.2%}")
            return True
        elif profit_percent >= self.take_profit_percent:
            self.logger.info(f"触发止盈: {profit_percent:.2%}")
            return True
            
        # 如果最新信号与持仓方向相反，也平仓
        if (position_amount > 0 and self.last_signal == -1) or (position_amount < 0 and self.last_signal == 1):
            self.logger.info(f"信号反转平仓: 当前持仓方向: {'多' if position_amount > 0 else '空'}, 最新信号: {self.last_signal}")
            return True
            
        return False
