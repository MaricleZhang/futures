"""
深度学习交易策略 - 15分钟时间框架
使用LSTM神经网络预测价格走势并自动交易
"""
import numpy as np
import pandas as pd
import talib
import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import logging
from strategies.base_strategy import BaseStrategy
import config

class DeepLearningStrategy15m(BaseStrategy):
    """DeepLearningStrategy15m - 基于深度学习的交易策略
    
    使用LSTM神经网络根据历史数据预测价格走势，自动执行买卖交易。
    在15分钟时间框架上运行。
    
    特点:
    1. 使用LSTM深度学习模型捕捉价格模式
    2. 自动特征工程和技术指标计算
    3. 周期性模型重训练以适应市场变化
    4. 内置风控措施包括止损、止盈和追踪止损
    5. 智能信号生成机制，减少假信号和过度交易
    """
    
    def __init__(self, trader):
        """初始化深度学习策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '15m'       # 15分钟K线
        self.check_interval = 300         # 检查信号间隔(秒)
        self.lookback_period = 200        # 计算指标所需的K线数量
        self.training_lookback = 1000     # 训练数据回看周期
        
        # 模型参数
        self.sequence_length = 20         # LSTM输入序列长度
        self.n_features = None            # 特征数量(动态确定)
        self.n_lstm_units = 64            # LSTM单元数量
        self.learning_rate = 0.001        # 优化器学习率
        self.batch_size = 32              # 训练批次大小
        self.epochs = 100                 # 训练最大轮数
        
        # 训练设置
        self.retraining_interval = 24 * 60 * 60  # 每天重新训练一次(秒)
        self.last_training_time = 0
        self.model_path = 'models'
        
        # 交易参数
        self.signal_threshold = 0.65      # 买卖信号阈值
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)
        self.stop_loss_pct = 0.02         # 止损比例
        self.take_profit_pct = 0.05       # 止盈比例
        self.trailing_stop = True         # 启用追踪止损
        self.trailing_stop_activation = 0.02  # 激活追踪止损的利润比例
        self.trailing_stop_distance = 0.01    # 追踪止损距离
        
        # 状态变量
        self.model = None
        self.scaler = StandardScaler()
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化LSTM模型"""
        # 创建模型目录
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        model_file = os.path.join(self.model_path, f"{self.trader.symbol}_lstm.h5")
        
        # 尝试加载已有模型
        if os.path.exists(model_file):
            try:
                self.model = load_model(model_file)
                self.logger.info(f"加载已有模型成功: {model_file}")
                return
            except Exception as e:
                self.logger.error(f"加载已有模型失败: {str(e)}")
        
        # 如果没有现有模型或加载失败，创建新模型
        self.logger.info("创建新的LSTM模型")
        # 实际模型创建会在第一次训练时进行，因为需要知道特征数量
    
    def _create_model(self, n_features):
        """创建新的LSTM模型
        
        Args:
            n_features: 特征数量
        
        Returns:
            keras.Model: 创建的LSTM模型
        """
        model = Sequential()
        
        # LSTM层
        model.add(LSTM(self.n_lstm_units, return_sequences=True, 
                       input_shape=(self.sequence_length, n_features),
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        model.add(LSTM(self.n_lstm_units // 2, return_sequences=False,
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # 输出层
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))  # 3个输出: 买入, 观望, 卖出
        
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.logger.info(f"LSTM模型创建成功: 输入特征{n_features}个, 序列长度{self.sequence_length}")
        return model
    
    def prepare_data(self, klines):
        """准备数据用于预测和训练
        
        Args:
            klines: K线数据
            
        Returns:
            pandas.DataFrame: 处理后的数据
        """
        try:
            # 转换K线为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算技术指标
            df = self._calculate_indicators(df)
            
            # 删除NaN值
            df = df.dropna()
            
            return df
        except Exception as e:
            self.logger.error(f"准备数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _calculate_indicators(self, df):
        """计算技术指标
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            pandas.DataFrame: 添加了指标的DataFrame
        """
        try:
            # 基础价格特征
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 移动平均线
            df['sma_7'] = talib.SMA(df['close'].values, timeperiod=7)
            df['sma_25'] = talib.SMA(df['close'].values, timeperiod=25)
            df['sma_99'] = talib.SMA(df['close'].values, timeperiod=99)
            df['ema_7'] = talib.EMA(df['close'].values, timeperiod=7)
            df['ema_25'] = talib.EMA(df['close'].values, timeperiod=25)
            df['ema_99'] = talib.EMA(df['close'].values, timeperiod=99)
            
            # 价格相对于移动平均线的比率
            df['close_sma_7_ratio'] = df['close'] / df['sma_7']
            df['close_sma_25_ratio'] = df['close'] / df['sma_25']
            df['sma_7_sma_25_ratio'] = df['sma_7'] / df['sma_25']
            
            # 动量指标
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['cci_14'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['adx_14'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # 布林带
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            df['bb_position'] = (df['close'] - lower) / (upper - lower)
            
            # 成交量指标
            df['volume_sma_7'] = talib.SMA(df['volume'].values, timeperiod=7)
            df['volume_ratio'] = df['volume'] / df['volume_sma_7']
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            
            # 波动率指标
            df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['atr_percent'] = df['atr_14'] / df['close'] * 100
            
            # 动量和趋势方向
            df['mom_7'] = talib.MOM(df['close'].values, timeperiod=7)
            df['mom_14'] = talib.MOM(df['close'].values, timeperiod=14)
            
            return df
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def prepare_features(self, df):
        """准备模型特征
        
        Args:
            df: 带有指标的DataFrame
            
        Returns:
            pandas.DataFrame: 特征DataFrame
        """
        try:
            # 选择要使用的特征
            feature_columns = [
                'returns', 'log_returns', 
                'close_sma_7_ratio', 'close_sma_25_ratio', 'sma_7_sma_25_ratio',
                'rsi_14', 'cci_14', 'adx_14', 
                'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position',
                'volume_ratio',
                'atr_percent'
            ]
            
            # 创建额外特征
            df['volatility'] = df['returns'].rolling(window=7).std()
            df['direction'] = np.where(df['close'].diff() > 0, 1, -1)
            df['trend_7'] = np.where(df['sma_7'] > df['sma_7'].shift(1), 1, -1)
            df['trend_25'] = np.where(df['sma_25'] > df['sma_25'].shift(1), 1, -1)
            df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
            df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
            
            # 添加这些新特征
            feature_columns.extend(['volatility', 'direction', 'trend_7', 'trend_25', 
                                   'rsi_oversold', 'rsi_overbought'])
            
            # 删除NaN值
            features = df[feature_columns].dropna()
            
            return features
        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_labels(self, df, future_bars=5, threshold=0.005):
        """生成训练标签
        
        Args:
            df: 带有OHLC数据的DataFrame
            future_bars: 向前看多少个K线
            threshold: 价格变化阈值
            
        Returns:
            pandas.DataFrame: 标签DataFrame
        """
        try:
            # 计算不同时间范围的未来收益率
            df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
            df['future_return_3'] = df['close'].shift(-3) / df['close'] - 1
            df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
            
            # 创建加权未来收益率
            df['future_return'] = (df['future_return_1'] * 0.5 + 
                                  df['future_return_3'] * 0.3 + 
                                  df['future_return_5'] * 0.2)
            
            # 创建标签
            df['label'] = 0  # 0 = 观望
            df.loc[df['future_return'] > threshold, 'label'] = 1  # 1 = 买入
            df.loc[df['future_return'] < -threshold, 'label'] = 2  # 2 = 卖出
            
            # 创建one-hot编码标签
            labels = pd.get_dummies(df['label'], prefix='action')
            
            # 确保有全部3个列
            for i in range(3):
                col = f'action_{i}'
                if col not in labels.columns:
                    labels[col] = 0
            
            return labels
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def prepare_sequences(self, features, labels=None):
        """为LSTM准备序列数据
        
        Args:
            features: 特征DataFrame
            labels: 标签DataFrame (可选)
            
        Returns:
            numpy.ndarray: 序列数据
        """
        try:
            X = []
            y = [] if labels is not None else None
            
            # 保存特征数量
            self.n_features = features.shape[1]
            
            # 创建序列
            for i in range(len(features) - self.sequence_length):
                X.append(features.iloc[i:i+self.sequence_length].values)
                
                if labels is not None:
                    y.append(labels.iloc[i+self.sequence_length].values)
            
            X = np.array(X)
            
            if labels is not None:
                y = np.array(y)
                return X, y
            else:
                return X
        except Exception as e:
            self.logger.error(f"准备序列数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def train_model(self, klines):
        """训练LSTM模型
        
        Args:
            klines: K线数据
            
        Returns:
            bool: 训练是否成功
        """
        try:
            self.logger.info("开始训练LSTM模型...")
            
            # 准备数据
            df = self.prepare_data(klines)
            if df is None or len(df) < self.sequence_length + 10:
                self.logger.error("训练数据不足")
                return False
            
            # 准备特征和标签
            features = self.prepare_features(df)
            labels = self.generate_labels(df)
            
            if features is None or labels is None:
                return False
            
            # 确保索引匹配
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # 标准化特征
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
            
            # 准备序列
            X, y = self.prepare_sequences(features_scaled, labels)
            
            if X is None or y is None:
                return False
            
            # 划分训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"训练数据: {len(X_train)}个序列, 验证数据: {len(X_val)}个序列")
            
            # 如果模型未初始化，创建新模型
            if self.model is None:
                self.model = self._create_model(self.n_features)
            
            # 提前停止和模型检查点
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_path, f"{self.trader.symbol}_lstm.h5"),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # 更新上次训练时间
            self.last_training_time = time.time()
            
            # 记录训练结果
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            self.logger.info(f"训练完成. 验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            
            return True
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def should_retrain(self):
        """检查是否需要重新训练模型"""
        current_time = time.time()
        if current_time - self.last_training_time >= self.retraining_interval:
            return True
        return False
    
    def predict(self, klines):
        """使用模型进行预测
        
        Args:
            klines: K线数据
            
        Returns:
            numpy.ndarray: 预测结果 [卖出概率, 观望概率, 买入概率]
        """
        try:
            # 检查模型是否已初始化
            if self.model is None:
                self.logger.error("模型未初始化")
                return None
            
            # 准备数据
            df = self.prepare_data(klines)
            if df is None or len(df) < self.sequence_length:
                self.logger.error("预测数据不足")
                return None
            
            # 准备特征
            features = self.prepare_features(df)
            if features is None:
                return None
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
            
            # 准备预测序列（只使用最新序列）
            X = self.prepare_sequences(features_scaled)
            if X is None or len(X) == 0:
                return None
            
            # 获取最新序列
            X_latest = X[-1].reshape(1, self.sequence_length, self.n_features)
            
            # 进行预测
            prediction = self.model.predict(X_latest)[0]
            
            return prediction
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_signal(self, klines):
        """生成交易信号
        
        Args:
            klines: K线数据
            
        Returns:
            int: 1(买入), -1(卖出), 0(观望)
        """
        try:
            # 检查是否需要重新训练
            if self.should_retrain():
                self.logger.info("模型需要重新训练...")
                
                # 获取更多数据用于训练
                training_klines = self.trader.get_klines(
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                
                if training_klines and len(training_klines) > self.sequence_length + 10:
                    self.train_model(training_klines)
                else:
                    self.logger.warning("重新训练的数据不足")
            
            # 进行预测
            prediction = self.predict(klines)
            if prediction is None:
                return 0
            
            # 提取概率
            sell_prob = prediction[0]  # action_0
            hold_prob = prediction[1]  # action_1
            buy_prob = prediction[2]   # action_2
            
            # 记录预测结果
            self.logger.info(f"预测结果: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
            
            # 根据概率生成信号
            if buy_prob > self.signal_threshold and buy_prob > sell_prob:
                return 1  # 买入信号
            elif sell_prob > self.signal_threshold and sell_prob > buy_prob:
                return -1  # 卖出信号
            else:
                return 0  # 观望信号
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """监控当前持仓并决定是否平仓"""
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
                
                # 获取最新预测，检查趋势是否反转
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                prediction = self.predict(klines)
                
                if prediction is not None:
                    # 提取概率
                    sell_prob = prediction[0]  # action_0
                    hold_prob = prediction[1]  # action_1
                    buy_prob = prediction[2]   # action_2
                    
                    # 检查趋势反转
                    if position_side == "多" and sell_prob > self.signal_threshold:
                        self.logger.info(f"模型指示趋势反转 (卖出概率: {sell_prob:.4f})，平多仓")
                        self.trader.close_position()
                        return
                    elif position_side == "空" and buy_prob > self.signal_threshold:
                        self.logger.info(f"模型指示趋势反转 (买入概率: {buy_prob:.4f})，平空仓")
                        self.trader.close_position()
                        return
        
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def run(self):
        """运行策略"""
        # 初始训练
        try:
            self.logger.info("开始初始模型训练...")
            
            # 获取历史K线数据进行训练
            klines = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if klines and len(klines) > self.sequence_length + 10:
                if self.train_model(klines):
                    self.logger.info("初始模型训练完成")
                else:
                    self.logger.error("初始模型训练失败")
            else:
                self.logger.error("初始训练数据不足")
        
        except Exception as e:
            self.logger.error(f"初始训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # 运行父类的常规监控循环
        super().run()
