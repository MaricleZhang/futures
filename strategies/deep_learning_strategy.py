import numpy as np
import pandas as pd
import talib
import time
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
import os
import warnings
import config
from strategies.base_strategy import BaseStrategy

class DeepLearningLSTMStrategy(BaseStrategy):
    """DeepLearningLSTMStrategy - 深度学习LSTM策略
    
    基于LSTM（长短期记忆网络）的深度学习交易策略，使用5分钟K线数据。
    该策略结合CNN和LSTM进行时间序列预测，并利用注意力机制增强对重要模式的识别。
    
    特点:
    1. 混合CNN-LSTM架构: CNN提取局部特征，LSTM捕捉时间序列关系
    2. 多时间框架分析: 同时考虑不同时间粒度的市场数据
    3. 特征工程: 结合技术指标、价格形态和波动特征
    4. 概率预测: 输出买入/卖出/观望的概率分布，用于风险管理
    5. 定期再训练: 适应市场不断变化的特征
    """
    
    MODEL_NAME = "DeepLearningLSTM"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化深度学习LSTM策略"""
        super().__init__(trader)

        # 抑制TensorFlow警告
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings('ignore', message='.*tf.data functions')
        
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '5m'      # 5分钟K线
        self.check_interval = 60        # 检查信号间隔(秒)
        self.lookback_window = 30       # 时序窗口大小
        self.training_lookback = 1000   # 训练所需K线数量
        
        # 模型路径
        self.model_dir = "models"
        self.model_path = f"{self.model_dir}/lstm_model_{self.trader.symbol}.h5"
        self.scaler_path = f"{self.model_dir}/lstm_scaler_{self.trader.symbol}.pkl"
        
        # 模型参数
        self.batch_size = 32
        self.epochs = 50
        self.input_dim = 30  # 输入特征维度
        self.output_classes = 3  # 输出类别: [卖出, 观望, 买入]
        self.n_steps = self.lookback_window  # 时序步数
        self.patience = 10  # 早停耐心值
        
        # 风险控制参数
        self.max_position_hold_time = 240  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.10      # 目标利润率 10%
        self.stop_loss_pct = 0.01         # 止损率 1%
        self.confidence_threshold = 0.45  # 交易信号置信度阈值
        self.max_trades_per_day = 6        # 每日最大交易次数
        
        # 交易状态
        self.trade_count_day = 0         # 当前日交易次数
        self.last_trade_day = None       # 上次交易的日期
        self.position_entry_time = None  # 开仓时间
        self.position_entry_price = None # 开仓价格
        self.last_action = 1             # 上一次动作，初始为观望
        
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # 标准化器
        self.scaler = None
            
        # 初始化模型
        self.model = self._build_model()
        
        # 尝试加载模型
        if not self._load_model():
            self.logger.info("未找到保存的模型，将进行初始训练")
            self._initial_training()
        
        self.last_training_time = time.time()
        self.retraining_interval = 7200  # 每2小时重新训练一次
        
    def _build_model(self):
        """构建深度学习模型"""
        try:
            # CNN-LSTM混合模型
            
            # 输入层
            input_layer = Input(shape=(self.n_steps, self.input_dim))
            
            # CNN部分 - 提取局部特征
            conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
            pool1 = MaxPooling1D(pool_size=2)(conv1)
            conv2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(pool1)
            pool2 = MaxPooling1D(pool_size=2)(conv2)
            
            # LSTM部分 - 捕捉时间序列特征
            lstm1 = LSTM(128, return_sequences=True)(pool2)
            dropout1 = Dropout(0.3)(lstm1)
            batch_norm1 = BatchNormalization()(dropout1)
            
            lstm2 = LSTM(64, return_sequences=False)(batch_norm1)
            dropout2 = Dropout(0.3)(lstm2)
            batch_norm2 = BatchNormalization()(dropout2)
            
            # 全连接层
            dense1 = Dense(64, activation='relu')(batch_norm2)
            dropout3 = Dropout(0.3)(dense1)
            
            # 输出层 - 概率分布
            output_layer = Dense(self.output_classes, activation='softmax')(dropout3)
            
            # 构建模型
            model = Model(inputs=input_layer, outputs=output_layer)
            
            # 编译模型
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info("深度学习模型构建成功")
            return model
            
        except Exception as e:
            self.logger.error(f"构建模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def _load_model(self):
        """加载保存的模型"""
        try:
            import pickle
            
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.logger.info(f"模型加载成功: {self.model_path}")
                
                # 加载标准化器
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.logger.info(f"标准化器加载成功: {self.scaler_path}")
                    return True
                else:
                    self.logger.warning(f"未找到标准化器: {self.scaler_path}")
                    return False
            else:
                self.logger.info(f"未找到模型文件: {self.model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _save_model(self):
        """保存模型到文件"""
        try:
            import pickle
            
            # 保存模型
            self.model.save(self.model_path)
            self.logger.info(f"模型保存成功: {self.model_path}")
            
            # 保存标准化器
            if self.scaler is not None:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                self.logger.info(f"标准化器保存成功: {self.scaler_path}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def monitor_position(self):
        """监控当前持仓状态并管理风险
        
        该函数负责:
        1. 检查当前持仓状态
        2. 计算持仓盈亏
        3. 执行止盈止损策略
        4. 检查持仓时间是否超过最大限制
        5. 获取并处理新的交易信号
        """
        try:
            # 获取当前持仓信息
            position = self.trader.get_position()
            # 使用get_market_price而不是get_latest_price，以兼容MockTrader
            current_price = self.trader.get_market_price()
            
            # 检查是否有持仓
            if position and position['positionAmt'] != 0:
                # 有持仓，检查是否需要平仓
                position_size = float(position['positionAmt'])
                entry_price = float(position['entryPrice'])
                unrealized_pnl = float(position['unrealizedProfit'])
                position_side = 'LONG' if position_size > 0 else 'SHORT'
                
                # 计算持仓时间
                current_time = time.time()
                if self.position_entry_time:
                    hold_time_minutes = (current_time - self.position_entry_time) / 60
                    
                    # 计算盈亏百分比
                    if position_side == 'LONG':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:  # SHORT
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    self.logger.info(f"当前持仓: {position_side}, 持仓量: {position_size}, 入场价: {entry_price}, "  
                                    f"当前价: {current_price}, 盈亏: {pnl_pct:.2%}, 持仓时间: {hold_time_minutes:.1f}分钟")
                    
                    # 止盈检查
                    if pnl_pct >= self.profit_target_pct:
                        self.logger.info(f"达到止盈条件 {pnl_pct:.2%} >= {self.profit_target_pct:.2%}, 平仓")
                        self._close_position()
                        return
                    
                    # 止损检查
                    if pnl_pct <= -self.stop_loss_pct:
                        self.logger.info(f"达到止损条件 {pnl_pct:.2%} <= -{self.stop_loss_pct:.2%}, 平仓")
                        self._close_position()
                        return
                    
                    # 最大持仓时间检查
                    if hold_time_minutes >= self.max_position_hold_time:
                        self.logger.info(f"达到最大持仓时间 {hold_time_minutes:.1f} >= {self.max_position_hold_time}, 平仓")
                        self._close_position()
                        return
                else:
                    # 如果有持仓但没有记录入场时间，则更新入场时间和价格
                    self.position_entry_time = current_time
                    self.position_entry_price = entry_price
                    self.logger.info(f"更新持仓记录: {position_side}, 入场价: {entry_price}, 入场时间: {datetime.fromtimestamp(current_time)}")
            else:
                # 无持仓，检查是否有新信号
                self.position_entry_time = None
                self.position_entry_price = None
                
                # 检查是否需要重新训练模型
                current_time = time.time()
                if current_time - self.last_training_time > self.retraining_interval:
                    self.logger.info("定期重新训练模型")
                    self._retrain_model()
                    self.last_training_time = current_time
                
                # 获取新的交易信号
                self._check_trading_signal()
                
        except Exception as e:
            self.logger.error(f"监控持仓出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _close_position(self):
        """平仓操作"""
        try:
            position = self.trader.get_position()
            if position and float(position['positionAmt']) != 0:
                position_size = float(position['positionAmt'])
                position_side = 'LONG' if position_size > 0 else 'SHORT'
                
                # 计算盈亏
                entry_price = float(position['entryPrice'])
                # 使用get_market_price而不是get_latest_price，以兼容MockTrader
                current_price = self.trader.get_market_price()
                if position_side == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # 执行平仓
                if position_side == 'LONG':
                    self.trader.sell(abs(position_size))
                    self.logger.info(f"平多仓: 数量={abs(position_size)}, 盈亏={pnl_pct:.2%}")
                else:  # SHORT
                    self.trader.buy(abs(position_size))
                    self.logger.info(f"平空仓: 数量={abs(position_size)}, 盈亏={pnl_pct:.2%}")
                
                # 重置持仓状态
                self.position_entry_time = None
                self.position_entry_price = None
                self.last_action = 1  # 重置为观望状态
                
        except Exception as e:
            self.logger.error(f"平仓操作失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _initial_training(self):
        """初始训练"""
        try:
            self.logger.info("开始初始训练...")
            
            # 获取历史K线数据
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
            
            if klines is not None and len(klines) >= 500:
                self.logger.info(f"获取到 {len(klines)} 根K线数据用于初始训练")
                
                # 训练模型
                self.train_model(klines)
                self._save_model()
                
                self.logger.info("初始训练完成")
            else:
                self.logger.error("获取K线数据失败或数据不足，无法进行初始训练")
                
        except Exception as e:
            self.logger.error(f"初始训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < self.lookback_window + 10:
                self.logger.error(f"K线数据不足，需要至少 {self.lookback_window + 10} 根")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 时间特征
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # === 1. 价格特征 ===
            df['return'] = df['close'].pct_change()  # 收益率
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))  # 对数收益率
            df['range'] = df['high'] - df['low']  # 价格范围
            df['body'] = abs(df['close'] - df['open'])  # K线实体大小
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)  # 上影线
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']  # 下影线
            
            # 价格与前N根K线的相对位置
            for n in [5, 10, 20]:
                df[f'rel_position_{n}'] = (df['close'] - df['low'].rolling(n).min()) / (df['high'].rolling(n).max() - df['low'].rolling(n).min())
            
            # === 2. 技术指标特征 ===
            # 移动平均线
            for n in [5, 10, 20, 50]:
                df[f'sma_{n}'] = df['close'].rolling(window=n).mean()
                df[f'ema_{n}'] = df['close'].ewm(span=n, adjust=False).mean()
                df[f'distance_to_sma_{n}'] = (df['close'] / df[f'sma_{n}'] - 1) * 100
            
            # 波动率指标
            df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['atr_ratio'] = df['atr_14'] / df['close'] * 100  # ATR占价格百分比
            df['bollinger_width'] = talib.STDDEV(df['close'].values, timeperiod=20) * 2 / df['close'] * 100
            
            # 动量指标
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_6'] = talib.RSI(df['close'].values, timeperiod=6)
            df['rsi_diff'] = df['rsi_6'] - df['rsi_14']  # RSI发散
            
            df['cci_14'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['mfi_14'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist
            
            # 随机指标
            df['slowk'], df['slowd'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            
            # === 3. 成交量特征 ===
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            df['volume_change'] = df['volume'].pct_change()
            
            # 价格与成交量关系
            df['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])
            
            # 成交量波动
            df['volume_std_10'] = df['volume'].rolling(10).std() / df['volume_sma_10']
            
            # === 4. 趋势强度特征 ===
            # ADX - 趋势强度
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # 价格趋势
            for n in [5, 10, 20]:
                df[f'price_trend_{n}'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n) * 100
            
            # === 5. 市场微观结构特征 ===
            # 高低价极值
            df['highlow_ratio'] = df['high'] / df['low']
            
            # 填充NaN值
            df = df.bfill().fillna(0)
            
            # 选择特征列
            feature_cols = [
                'return', 'log_return', 'range', 'body', 'upper_shadow', 'lower_shadow',
                'rel_position_5', 'rel_position_10', 'rel_position_20',
                'distance_to_sma_5', 'distance_to_sma_10', 'distance_to_sma_20', 'distance_to_sma_50',
                'atr_ratio', 'bollinger_width',
                'rsi_14', 'rsi_diff', 'cci_14', 'mfi_14',
                'macd', 'macdsignal', 'macdhist',
                'slowk', 'slowd',
                'volume_ratio', 'volume_change', 'price_volume_corr', 'volume_std_10',
                'adx', 'plus_di', 'minus_di',
                'price_trend_5', 'price_trend_10', 'price_trend_20',
                'highlow_ratio'
            ]
            
            # 确保我们有足够的特征
            feature_cols = feature_cols[:self.input_dim]
            if len(feature_cols) < self.input_dim:
                self.logger.warning(f"特征数量不足，需要 {self.input_dim}，仅有 {len(feature_cols)}")
                # 复制一些特征以达到所需数量
                while len(feature_cols) < self.input_dim:
                    feature_cols.append(feature_cols[0])
            
            # 提取特征
            features = df[feature_cols].values
            
            return features, df
            
        except Exception as e:
            self.logger.error(f"准备特征数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
            
    def generate_labels(self, df):
        """生成标签数据"""
        try:
            # 计算未来N根K线的价格变化
            future_windows = [1, 3, 6, 12]
            future_weights = [0.4, 0.3, 0.2, 0.1]  # 权重，近期更重要
            
            labels = np.zeros(len(df))
            
            for i, window in enumerate(future_windows):
                # 计算未来window根K线的价格变化百分比
                future_change = df['close'].shift(-window) / df['close'] - 1
                
                # 定义上涨/下跌阈值，根据ATR动态调整
                threshold = df['atr_ratio'] * 0.5  # 使用ATR的一半作为动态阈值
                
                # 生成信号: 1=买入, 0=观望, -1=卖出
                temp_labels = np.zeros(len(df))
                temp_labels[future_change > threshold] = 1  # 未来价格上涨超过阈值
                temp_labels[future_change < -threshold] = -1  # 未来价格下跌超过阈值
                
                # 加权添加到最终标签
                labels += temp_labels * future_weights[i]
            
            # 根据加权结果确定最终标签
            final_labels = np.zeros(len(df), dtype=int)
            final_labels[labels > 0.2] = 2  # 买入信号 (映射到2)
            final_labels[labels < -0.2] = 0  # 卖出信号 (映射到0)
            final_labels[(labels >= -0.2) & (labels <= 0.2)] = 1  # 观望信号 (映射到1)
            
            return final_labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def prepare_sequences(self, features, labels):
        """将特征和标签数据转换为序列格式"""
        try:
            # 确保数据足够多
            if len(features) < self.n_steps:
                self.logger.error(f"数据不足，需要至少 {self.n_steps} 条数据，但只有 {len(features)} 条")
                return None, None
            
            X, y = [], []
            
            # 对于标签映射: -1->0, 0->1, 1->2
            category_mapping = {-1: 0, 0: 1, 1: 2}
            
            for i in range(len(features) - self.n_steps):
                # 提取特征序列
                X.append(features[i:i + self.n_steps])
                
                # 使用序列之后的标签作为预测目标
                label = labels[i + self.n_steps]
                y.append(label)
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 将标签转换为one-hot编码
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=self.output_classes)
            
            return X, y_onehot
            
        except Exception as e:
            self.logger.error(f"准备序列数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
            
    def train_model(self, klines):
        """训练模型"""
        try:
            self.logger.info("开始训练模型...")
            
            # 准备特征和标签
            features, df = self.prepare_features(klines)
            if features is None or df is None:
                self.logger.error("特征准备失败，无法训练模型")
                return False
                
            # 标准化特征 - 使用sklearn的StandardScaler
            from sklearn.preprocessing import StandardScaler
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized_features = self.scaler.fit_transform(features)
            else:
                normalized_features = self.scaler.transform(features)
            
            # 生成标签
            labels = self.generate_labels(df)
            if labels is None:
                self.logger.error("标签生成失败，无法训练模型")
                return False
                
            # 准备序列数据
            X, y = self.prepare_sequences(normalized_features, labels)
            if X is None or y is None:
                self.logger.error("序列数据准备失败，无法训练模型")
                return False
                
            self.logger.info(f"准备完成: X形状={X.shape}, y形状={y.shape}")
            
            # 计算样本权重 - 解决类别不平衡
            class_weights = {}
            for i in range(self.output_classes):
                class_weights[i] = len(y) / (self.output_classes * np.sum(y[:, i]))
                
            # 创建回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.patience,
                    restore_best_weights=True
                )
            ]
            
            # 训练集/验证集分割
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # 评估模型
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            self.logger.info(f"验证集上的性能: 损失={val_loss:.4f}, 准确率={val_acc:.4f}")
            
            # 保存模型
            self._save_model()
            
            self.last_training_time = time.time()
            self.logger.info("模型训练完成")
            
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
        
    def _check_trading_signal(self):
        """检查并执行交易信号"""
        try:
            # 获取K线数据
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_window + 50)
            if not klines or len(klines) < self.lookback_window:
                self.logger.warning(f"获取K线数据不足: {len(klines) if klines else 0} < {self.lookback_window}")
                return
            
            # 生成交易信号
            signal = self.generate_signal(klines)
            
            # 根据信号执行交易
            current_price = self.trader.get_market_price()
            
            # 获取账户余额
            balance = self.trader.get_balance()
            available_balance = float(balance['free'])
            
            # 计算交易量
            position_size = self._calculate_position_size(available_balance, current_price)
            
            if signal > 0:  # 买入信号
                self.logger.info(f"生成买入信号, 当前价格: {current_price}, 交易量: {position_size}")
                self.trader.buy(position_size)
                self.position_entry_time = time.time()
                self.position_entry_price = current_price
                
            elif signal < 0:  # 卖出信号
                self.logger.info(f"生成卖出信号, 当前价格: {current_price}, 交易量: {position_size}")
                self.trader.sell(position_size)
                self.position_entry_time = time.time()
                self.position_entry_price = current_price
                
            else:  # 观望信号
                self.logger.info("生成观望信号, 不执行交易")
                
        except Exception as e:
            self.logger.error(f"检查交易信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _calculate_position_size(self, available_balance, current_price):
        """计算交易量"""
        try:
            # 使用可用余额的30%进行交易
            trade_amount = available_balance * 0.3
            
            # 计算可交易的数量
            position_size = trade_amount / current_price
            
            # 四舍五入到小数点3位
            position_size = round(position_size, 3)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算交易量出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.001  # 返回最小交易量
    
    def generate_signal(self, klines):
        """生成交易信号
        
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            # 检查是否需要重新训练
            if self.should_retrain():
                self.logger.info("模型需要重新训练")
                training_klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
                self.train_model(training_klines)
            
            # 准备特征
            features, df = self.prepare_features(klines)
            if features is None or df is None:
                self.logger.error("特征准备失败，无法生成信号")
                return 0
                
            # 标准化特征
            normalized_features = self.scaler.transform(features)
            
            # 准备输入序列 - 使用最后lookback_window个数据点
            input_sequence = normalized_features[-self.lookback_window:].reshape(1, self.lookback_window, self.input_dim)
            
            # 模型预测
            prediction = self.model.predict(input_sequence, verbose=0)[0]
            
            # 获取最高概率的类别
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            
            self.logger.info(f"预测类别: {pred_class}, 置信度: {confidence:.4f}")
            self.logger.info(f"各类别概率: 卖出={prediction[0]:.4f}, 观望={prediction[1]:.4f}, 买入={prediction[2]:.4f}")
            
            # 检查置信度是否足够高
            if confidence < self.confidence_threshold:
                self.logger.info(f"预测置信度 {confidence:.4f} 低于阈值 {self.confidence_threshold}，生成观望信号")
                return 0  # 观望信号
                
            # 映射类别到信号: [0->-1, 1->0, 2->1]
            signal_mapping = {0: -1, 1: 0, 2: 1}
            signal = signal_mapping[pred_class]
            
            # 记录信号
            self.last_action = pred_class
            
            # 检查交易频率限制
            current_day = datetime.now().day
            if current_day != self.last_trade_day:
                self.last_trade_day = current_day
                self.trade_count_day = 0
                
            if signal != 0:  # 如果不是观望信号，增加计数
                self.trade_count_day += 1
                
            if self.trade_count_day > self.max_trades_per_day:
                self.logger.info(f"已达到每日最大交易次数({self.max_trades_per_day})，强制生成观望信号")
                return 0
                
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0  # 错误时返回观望信号
            
    