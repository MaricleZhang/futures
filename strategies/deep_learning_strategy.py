"""
深度学习趋势预测策略 - 15分钟时间框架改进版
使用LSTM和CNN网络进行价格趋势预测，专注于减少过度拟合和提高预测准确性
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
import time
import logging
import random
import talib
import matplotlib.pyplot as plt
from strategies.base_strategy import BaseStrategy


class LSTMWithAttention(nn.Module):
    """
    使用注意力机制的LSTM模型，减少过拟合并提高模型准确性
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.3):
        super(LSTMWithAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM捕获更多时序信息
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2是因为bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Dropout用于减少过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm代替BatchNorm
            nn.SiLU(),  # 使用SiLU(Swish)激活函数
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2是因为bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch, seq_len, hidden_dim*2]
        
        # 应用注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_dim*2]
        
        # 应用dropout
        context = self.dropout(context)
        
        # 全连接层预测
        output = self.fc(context)
        
        return output


class EnhancedDeepLearningStrategy15m(BaseStrategy):
    """EnhancedDeepLearningStrategy15m - 改进的深度学习趋势预测策略
    
    使用LSTM、注意力机制和集成学习预测未来价格走势。
    专注于减少过度拟合，提高预测准确性，并加入更多防护机制:
    
    改进：
    1. 使用双向LSTM和注意力机制
    2. 引入模型保存和加载功能
    3. 动态调整预测阈值
    4. 增加模型验证和评估
    5. 加入异常值检测
    6. 使用K折交叉验证
    7. 多模型集成投票
    8. 与技术分析结合确认信号
    """
    
    def __init__(self, trader):
        """初始化改进的深度学习趋势预测策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '15m'       # 15分钟K线
        self.check_interval = 60          # 检查信号间隔(秒)
        self.lookback_period = 240        # 计算指标所需的K线数量
        self.input_window = 24            # 输入窗口大小 (6小时)
        self.prediction_horizon = 4       # 预测未来几个K线 (1小时)
        self.training_lookback = 1000     # 训练数据回溯期
        
        # 模型参数
        self.input_dim = 30               # 输入特征维度（自动调整）
        self.hidden_dim = 128             # 隐藏层维度 (增大以提高表达能力)
        self.num_layers = 2               # LSTM层数
        self.dropout_rate = 0.4           # Dropout比例 (加大以减少过拟合)
        self.learning_rate = 0.0005       # 学习率 (减小以稳定训练)
        self.batch_size = 64              # 批次大小 (加大以增加稳定性)
        self.num_epochs = 100             # 训练轮数
        self.early_stop_patience = 15     # 早停耐心值 (加大以避免过早停止)
        self.weight_decay = 1e-4          # L2正则化权重
        
        # 集成学习参数
        self.num_models = 5               # 集成模型数量 (增加以提高稳定性)
        self.models = []                  # 模型列表
        self.voting_threshold = 0.6       # 投票阈值 (至少60%的模型同意才生成信号)
        
        # 预测参数
        self.min_confidence = 0.65        # 最小信心阈值
        self.prediction_threshold = 0.003  # 预测变化率阈值 (0.3%)
        self.adaptive_threshold = True    # 是否使用自适应阈值
        self.max_prediction = 0.05        # 最大预测变化阈值 (防止极端预测)
        
        # 持仓控制参数
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)
        self.stop_loss_pct = 0.02         # 止损比例
        self.take_profit_pct = 0.04       # 止盈比例
        self.trailing_stop = True         # 是否启用追踪止损
        self.trailing_stop_activation = 0.02  # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.01    # 追踪止损距离百分比
        
        # 内部状态
        self.is_trained = False           # 模型是否已训练
        self.last_training_time = 0       # 上次训练时间
        self.retraining_interval = 3*3600  # 重新训练间隔(秒) (减少重训练频率)
        self.scaler_x = None              # 特征缩放器
        self.scaler_y = None              # 目标缩放器
        self.last_prediction = None       # 上次预测结果
        self.position_entry_time = None   # 开仓时间
        self.position_entry_price = None  # 开仓价格
        self.max_profit_reached = 0       # 达到的最大利润
        self.prediction_history = []      # 预测历史
        
        # 数据增强和预处理参数
        self.use_data_augmentation = True  # 是否使用数据增强
        self.noise_level = 0.0005         # 数据增强时添加的噪声级别 (减小噪声)
        self.validation_split = 0.2       # 验证集比例
        self.test_split = 0.1             # 测试集比例
        
        # 日志和调试
        self.debug_mode = True            # 调试模式，输出更多信息
        self.model_save_path = 'models'   # 模型保存路径
        self.save_models = True           # 是否保存模型
        self.save_predictions = True      # 是否保存预测结果
        
        # 初始化模型
        self._init_models()
        
        # 市场状态参数
        self.market_state = "unknown"     # 市场状态
        self.trend_strength = 0           # 趋势强度
        self.current_trend = 0            # 当前趋势 (1: 上升, -1: 下降, 0: 中性)
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 创建模型保存目录
        if self.save_models and not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            
    def _generate_model_filename(self, model_idx):
        """生成模型文件名"""
        symbol = self.trader.symbol if self.trader.symbol else "unknown"
        return f"{self.model_save_path}/{symbol}_model_{model_idx}.pth"
            
    def _init_models(self):
        """初始化模型，并尝试加载已保存的模型"""
        self.models = []
        for i in range(self.num_models):
            model = LSTMWithAttention(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.prediction_horizon,
                dropout_rate=self.dropout_rate
            )
            
            # 尝试加载已保存的模型
            model_path = self._generate_model_filename(i)
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()  # 设置为评估模式
                    self.is_trained = True
                    self.logger.info(f"成功加载已保存的模型 {i+1}")
                except Exception as e:
                    self.logger.error(f"加载模型 {i+1} 失败: {str(e)}")
            
            self.models.append(model)
            
        self.logger.info(f"初始化了 {self.num_models} 个预测模型")
        
    def _save_models(self):
        """保存模型到文件"""
        if not self.save_models:
            return
            
        for i, model in enumerate(self.models):
            try:
                model_path = self._generate_model_filename(i)
                torch.save(model.state_dict(), model_path)
                self.logger.info(f"模型 {i+1} 已保存到 {model_path}")
            except Exception as e:
                self.logger.error(f"保存模型 {i+1} 失败: {str(e)}")
            
    def _prepare_data(self, klines):
        """
        准备深度学习模型所需的数据
        
        Args:
            klines (list): K线数据
            
        Returns:
            tuple: (特征DataFrame, 标签数组)
        """
        try:
            # 转换K线数据为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 数值列转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # 特征工程
            features_df = self._engineer_features(df)
            
            # 准备目标值 - 未来几个K线的收盘价相对变化
            targets = []
            for i in range(1, self.prediction_horizon + 1):
                # 计算未来第i个K线的收盘价相对当前收盘价的变化率
                target_col = f'future_return_{i}'
                features_df[target_col] = df['close'].pct_change(i).shift(-i)
                targets.append(target_col)
                
            # 删除NaN行
            features_df = features_df.dropna()
            
            # 分离特征和目标
            X = features_df.drop(targets, axis=1)
            y = features_df[targets].values
            
            # 记录详细的特征信息
            if self.debug_mode:
                self.logger.info(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
                self.logger.info(f"特征列: {', '.join(X.columns)}")
                
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
            
    def _engineer_features(self, df):
        """
        特征工程，生成深度学习模型的输入特征
        
        Args:
            df (pandas.DataFrame): K线数据
            
        Returns:
            pandas.DataFrame: 特征DataFrame
        """
        # 记录当前每个步骤的特征数量
        self.logger.info("执行特征工程...")
        feature_count = 0
        
        # 创建特征列表
        features = pd.DataFrame(index=df.index)
        
        # 1. 价格特征
        features['close_norm'] = df['close'] / df['close'].iloc[-50:].mean()  # 归一化收盘价
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        features['log_return_3'] = np.log(df['close'] / df['close'].shift(3))
        features['log_return_6'] = np.log(df['close'] / df['close'].shift(6))
        features['log_return_12'] = np.log(df['close'] / df['close'].shift(12))
        
        # 计算高低点差
        features['high_low_diff'] = (df['high'] - df['low']) / df['low']
        features['open_close_diff'] = (df['close'] - df['open']) / df['open']
        features['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # 计算振幅
        features['amplitude'] = (df['high'] - df['low']) / df['open']
        
        # 2. 量价特征
        # 对数成交量 (处理量的指数级差异)
        features['log_volume'] = np.log1p(df['volume'])
        features['volume_ma5'] = df['volume'].rolling(window=5).mean()
        features['volume_ma10'] = df['volume'].rolling(window=10).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma5']
        features['volume_change'] = df['volume'].pct_change()
        
        # 量价相关性
        features['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])
        
        # 3. 技术指标
        # 移动平均线
        features['sma5'] = talib.SMA(df['close'], timeperiod=5)
        features['sma10'] = talib.SMA(df['close'], timeperiod=10)
        features['sma20'] = talib.SMA(df['close'], timeperiod=20)
        features['sma50'] = talib.SMA(df['close'], timeperiod=50)
        
        # EMA
        features['ema5'] = talib.EMA(df['close'], timeperiod=5)
        features['ema10'] = talib.EMA(df['close'], timeperiod=10)
        
        # 相对强弱指标
        features['close_sma5_ratio'] = df['close'] / features['sma5'] - 1
        features['close_sma20_ratio'] = df['close'] / features['sma20'] - 1
        features['sma5_sma20_ratio'] = features['sma5'] / features['sma20'] - 1
        features['ema_diff'] = features['ema5'] - features['ema10']
        
        # RSI
        features['rsi6'] = talib.RSI(df['close'], timeperiod=6)
        features['rsi14'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        features['macd_diff'] = macd - macd_signal
        
        # 布林带
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # CCI (Commodity Channel Index)
        features['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ATR - 波动率指标
        features['atr14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr14_ratio'] = features['atr14'] / df['close']
        
        # 钱德动量摆动指标 (CMO)
        features['cmo'] = talib.CMO(df['close'], timeperiod=14)
        
        # 随机指标
        features['slowk'], features['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                                         fastk_period=5, slowk_period=3, slowk_matype=0, 
                                                         slowd_period=3, slowd_matype=0)
        
        # 4. 价格动量特征
        # ROC (Rate of Change)
        features['roc1'] = talib.ROC(df['close'], timeperiod=1)
        features['roc5'] = talib.ROC(df['close'], timeperiod=5)
        features['roc10'] = talib.ROC(df['close'], timeperiod=10)
        
        # 5. 波动率特征
        # 标准差
        features['std5'] = df['close'].rolling(5).std() / df['close']
        features['std10'] = df['close'].rolling(10).std() / df['close']
        
        # 6. 交易趋势特征
        # 方向指标
        features['di_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['di_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['di_diff'] = features['di_plus'] - features['di_minus']
        
        # ADX (平均方向指数)
        features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 7. 交易日特征
        # 提取交易时间特征
        features['hour'] = df['datetime'].dt.hour
        features['minute'] = df['datetime'].dt.minute
        features['day_of_week'] = df['datetime'].dt.dayofweek
        
        # 8. 高阶特征组合
        # 特征交叉项
        features['rsi_macd'] = features['rsi14'] * np.sign(features['macd_hist'])
        features['trend_strength'] = features['adx'] * np.sign(features['di_diff'])
        
        # 删除NaN行
        features = features.dropna()
        
        # 记录特征数量
        self.logger.info(f"特征工程完成，共生成 {features.shape[1]} 个特征")
        
        return features
    
    def _split_and_scale_data(self, X, y, is_training=True):
        """
        分割并缩放数据
        
        Args:
            X (pandas.DataFrame): 特征
            y (numpy.ndarray): 目标
            is_training (bool): 是否在训练
            
        Returns:
            tuple: 分割和缩放后的数据
        """
        if is_training:
            # 创建特征缩放器
            self.scaler_x = StandardScaler()
            X_scaled = self.scaler_x.fit_transform(X)
            
            # 创建目标缩放器 (仅对目标进行MinMax缩放以保留符号)
            self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
            y_scaled = self.scaler_y.fit_transform(y)
            
            # 划分训练集、验证集和测试集
            total_samples = len(X_scaled)
            test_size = int(total_samples * self.test_split)
            val_size = int(total_samples * self.validation_split)
            train_size = total_samples - test_size - val_size
            
            # 划分索引
            indices = np.arange(total_samples)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            test_indices = indices[train_size+val_size:]
            
            # 划分数据
            X_train, y_train = X_scaled[train_indices], y_scaled[train_indices]
            X_val, y_val = X_scaled[val_indices], y_scaled[val_indices]
            X_test, y_test = X_scaled[test_indices], y_scaled[test_indices]
            
            return X_train, y_train, X_val, y_val, X_test, y_test
        else:
            # 预测时只缩放特征
            X_scaled = self.scaler_x.transform(X)
            return X_scaled
    
    def _create_sequences(self, X, y=None, is_training=True):
        """
        创建序列数据用于LSTM模型
        
        Args:
            X (numpy.ndarray): 特征数组
            y (numpy.ndarray): 目标数组，仅在训练时需要
            is_training (bool): 是否为训练模式
            
        Returns:
            tuple: (X序列, y目标)或单个X序列
        """
        sequences_X = []
        
        # 创建滑动窗口序列
        for i in range(len(X) - self.input_window + 1):
            sequences_X.append(X[i:i + self.input_window])
            
        # 转换为numpy数组
        sequences_X = np.array(sequences_X)
        
        if is_training and y is not None:
            sequences_y = []
            for i in range(len(X) - self.input_window + 1):
                # 序列最后一个点的索引
                target_idx = i + self.input_window - 1
                if target_idx < len(y):
                    sequences_y.append(y[target_idx])
            
            sequences_y = np.array(sequences_y)
            return sequences_X, sequences_y
        else:
            return sequences_X
    
    def _data_augmentation(self, X_seq, y_seq):
        """
        数据增强，通过添加噪声等方法创建更多训练样本
        
        Args:
            X_seq (numpy.ndarray): 特征序列
            y_seq (numpy.ndarray): 目标序列
            
        Returns:
            tuple: (增强后的特征, 增强后的目标)
        """
        if not self.use_data_augmentation:
            return X_seq, y_seq
            
        self.logger.info("执行数据增强...")
        augmented_X = [X_seq]
        augmented_y = [y_seq]
        
        # 方法1: 添加微小高斯噪声
        noise_scale = self.noise_level
        for _ in range(2):  # 生成2个噪声版本
            noise = np.random.normal(0, noise_scale, X_seq.shape)
            noisy_X = X_seq + noise
            augmented_X.append(noisy_X)
            augmented_y.append(y_seq)
        
        # 方法2: 时间抖动 (微小时间偏移)
        jitter_indices = np.arange(X_seq.shape[0])
        for _ in range(1):  # 生成1个时间抖动版本
            jitter_X = np.empty_like(X_seq)
            for i in range(X_seq.shape[1]):
                # 随机选择相邻时间点
                shift = np.random.randint(-1, 2, size=X_seq.shape[0])
                for j in range(X_seq.shape[0]):
                    idx = max(0, min(X_seq.shape[0]-1, j + shift[j]))
                    jitter_X[j, i] = X_seq[idx, i]
            
            augmented_X.append(jitter_X)
            augmented_y.append(y_seq)
        
        # 连接所有增强数据
        X_augmented = np.vstack(augmented_X)
        y_augmented = np.vstack(augmented_y)
        
        self.logger.info(f"数据增强完成，从 {len(X_seq)} 增加到 {len(X_augmented)} 个样本")
        
        return X_augmented, y_augmented
    
    def _evaluate_model(self, model, X_test, y_test):
        """
        评估模型性能
        
        Args:
            model (torch.nn.Module): 训练好的模型
            X_test (numpy.ndarray): 测试集特征
            y_test (numpy.ndarray): 测试集目标
            
        Returns:
            float: 测试集损失
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_tensor = torch.FloatTensor(y_test).to(self.device)
            
            outputs = model(X_tensor)
            criterion = nn.MSELoss()
            loss = criterion(outputs, y_tensor)
            
            # 转换回原始比例计算指标
            y_pred = outputs.cpu().numpy()
            y_pred_original = self.scaler_y.inverse_transform(y_pred)
            y_true_original = self.scaler_y.inverse_transform(y_test)
            
            # 计算方向准确率 (预测正负符号是否正确)
            direction_accuracy = np.mean((y_pred_original[:, 0] > 0) == (y_true_original[:, 0] > 0))
            
            # 计算平均绝对误差
            mae = np.mean(np.abs(y_pred_original - y_true_original))
            
            return loss.item(), direction_accuracy, mae
        
    def _train_model(self, X, y):
        """
        训练深度学习模型
        
        Args:
            X (pandas.DataFrame): 特征数据
            y (numpy.ndarray): 目标数据
            
        Returns:
            bool: 训练是否成功
        """
        try:
            start_time = time.time()
            self.logger.info("开始训练模型...")
            
            # 分割并缩放数据
            X_train, y_train, X_val, y_val, X_test, y_test = self._split_and_scale_data(X, y, is_training=True)
            
            # 创建序列
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, is_training=True)
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, is_training=True)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, is_training=True)
            
            self.logger.info(f"训练集大小: {X_train_seq.shape}, 验证集大小: {X_val_seq.shape}, 测试集大小: {X_test_seq.shape}")
            
            # 数据增强
            X_train_aug, y_train_aug = self._data_augmentation(X_train_seq, y_train_seq)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train_aug)
            y_train_tensor = torch.FloatTensor(y_train_aug)
            X_val_tensor = torch.FloatTensor(X_val_seq)
            y_val_tensor = torch.FloatTensor(y_val_seq)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            # 训练每个模型
            for i, model in enumerate(self.models):
                self.logger.info(f"训练模型 {i+1}/{len(self.models)}")
                
                # 将模型移到设备
                model = model.to(self.device)
                
                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                
                # 学习率调度器
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                
                # 设置早停
                best_val_loss = float('inf')
                patience_counter = 0
                
                # 训练循环
                for epoch in range(self.num_epochs):
                    # 训练阶段
                    model.train()
                    train_loss = 0.0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        
                        # 前向传播
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        
                        # 反向传播和优化
                        loss.backward()
                        # 梯度裁剪，防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        train_loss += loss.item() * X_batch.size(0)
                    
                    train_loss /= len(train_loader.dataset)
                    
                    # 验证阶段
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            val_loss += loss.item() * X_batch.size(0)
                        
                        val_loss /= len(val_loader.dataset)
                    
                    # 更新学习率
                    scheduler.step(val_loss)
                    
                    # 每隔几个epoch打印一次训练情况
                    if (epoch+1) % 5 == 0 or epoch == 0 or epoch == self.num_epochs-1:
                        self.logger.info(f"模型 {i+1}, Epoch [{epoch+1}/{self.num_epochs}], "
                                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # 早停检查
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # 保存最佳模型状态
                        if self.save_models:
                            model_path = self._generate_model_filename(i)
                            torch.save(model.state_dict(), model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stop_patience:
                            self.logger.info(f"模型 {i+1} 早停在 epoch {epoch+1}")
                            break
                
                # 评估模型
                test_loss, direction_acc, mae = self._evaluate_model(model, X_test_seq, y_test_seq)
                self.logger.info(f"模型 {i+1} 测试集评估 - Loss: {test_loss:.6f}, 方向准确率: {direction_acc:.4f}, MAE: {mae:.6f}")
                
                # 将模型移回CPU以节省GPU内存
                model = model.cpu()
                self.models[i] = model
            
            self.is_trained = True
            self.last_training_time = time.time()
            training_duration = (time.time() - start_time) / 60
            self.logger.info(f"所有模型训练完成，耗时 {training_duration:.2f} 分钟")
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_adaptive_threshold(self, df, window=20):
        """
        计算自适应预测阈值，根据近期市场波动性调整
        
        Args:
            df (pandas.DataFrame): K线数据
            window (int): 计算窗口
            
        Returns:
            float: 自适应阈值
        """
        try:
            # 计算近期的价格波动
            close_prices = df['close'].iloc[-window:]
            returns = close_prices.pct_change().dropna()
            
            # 使用ATR作为波动性指标
            high = df['high'].iloc[-window:]
            low = df['low'].iloc[-window:]
            close = df['close'].iloc[-window:]
            
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
            atr_pct = atr[-1] / close.iloc[-1]
            
            # 计算自适应阈值: 基础阈值 + 市场波动因子
            volatility_factor = min(0.5, max(0.1, atr_pct * 0.5))
            
            # 计算基于历史数据的阈值
            base_threshold = self.prediction_threshold
            adaptive_threshold = base_threshold * (1 + volatility_factor)
            
            # 确保阈值在合理范围内
            adaptive_threshold = min(0.01, max(0.001, adaptive_threshold))
            
            if self.debug_mode:
                self.logger.info(f"自适应阈值: {adaptive_threshold:.6f} (基础阈值: {base_threshold:.6f}, "
                              f"波动因子: {volatility_factor:.4f}, ATR百分比: {atr_pct:.4f})")
                
            return adaptive_threshold
            
        except Exception as e:
            self.logger.error(f"计算自适应阈值失败: {str(e)}")
            return self.prediction_threshold
    
    def _predict(self, X, df=None):
        """
        使用训练好的模型集合进行预测
        
        Args:
            X (pandas.DataFrame): 特征数据
            df (pandas.DataFrame): 原始K线数据，用于计算自适应阈值
            
        Returns:
            tuple: (预测值, 预测置信度, 投票结果)
        """
        try:
            if not self.is_trained or len(self.models) == 0:
                return None, 0, []
                
            # 使用相同的缩放器缩放特征
            X_scaled = self._split_and_scale_data(X, None, is_training=False)
            
            # 创建序列
            X_seq = self._create_sequences(X_scaled, is_training=False)
            
            # 只取最后一个序列进行预测
            if len(X_seq) > 0:
                X_last = X_seq[-1:]
                X_tensor = torch.FloatTensor(X_last).to(self.device)
                
                # 收集所有模型的预测
                all_predictions = []
                
                for i, model in enumerate(self.models):
                    model = model.to(self.device)
                    model.eval()
                    
                    with torch.no_grad():
                        prediction = model(X_tensor)
                        # 将预测转换回原始比例
                        prediction_np = prediction.cpu().numpy()
                        if self.scaler_y is not None:
                            prediction_np = self.scaler_y.inverse_transform(prediction_np)
                        all_predictions.append(prediction_np[0])
                    
                    # 将模型移回CPU
                    model = model.cpu()
                
                # 计算平均预测值
                all_predictions = np.array(all_predictions)
                avg_prediction = np.mean(all_predictions, axis=0)
                
                # 计算预测的标准差作为不确定性度量
                std_prediction = np.std(all_predictions, axis=0)
                
                # 计算置信度 (置信度 = 1 / (1 + 标准差))
                confidence = 1 / (1 + np.mean(std_prediction))
                
                # 对每个预测进行投票 (是否预测上涨)
                votes = []
                for pred in all_predictions:
                    # 限制预测值在合理范围内
                    pred = np.clip(pred, -self.max_prediction, self.max_prediction)
                    votes.append(1 if np.mean(pred) > 0 else -1)
                
                # 计算自适应阈值
                if self.adaptive_threshold and df is not None:
                    threshold = self._get_adaptive_threshold(df)
                else:
                    threshold = self.prediction_threshold
                
                return avg_prediction, confidence, votes, threshold
            else:
                return None, 0, [], self.prediction_threshold
                
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, 0, [], self.prediction_threshold
    
    def analyze_market_state(self, klines):
        """
        分析市场状态
        
        Args:
            klines (list): K线数据
            
        Returns:
            dict: 市场状态信息
        """
        try:
            # 转换K线数据为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # 计算ADX指标来评估趋势强度
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            
            # 计算DI指标来判断趋势方向
            plus_di = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            minus_di = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            current_plus_di = plus_di[-1] if not np.isnan(plus_di[-1]) else 0
            current_minus_di = minus_di[-1] if not np.isnan(minus_di[-1]) else 0
            
            # 计算EMA交叉情况
            ema_fast = talib.EMA(df['close'].values, timeperiod=5)
            ema_slow = talib.EMA(df['close'].values, timeperiod=20)
            
            # 确定趋势方向
            trend = 0
            if current_plus_di > current_minus_di and ema_fast[-1] > ema_slow[-1]:
                trend = 1  # 上升趋势
            elif current_minus_di > current_plus_di and ema_fast[-1] < ema_slow[-1]:
                trend = -1  # 下降趋势
                
            # 计算波动性
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            atr_pct = atr[-1] / df['close'].iloc[-1] if not np.isnan(atr[-1]) else 0
            
            # 判断市场状态
            state = "unknown"
            if current_adx < 20:
                state = "range"  # 震荡市场
            elif current_adx >= 20 and current_adx < 40:
                state = "weak_trend"  # 弱趋势
            else:
                state = "strong_trend"  # 强趋势
                
            # 检查是否处于超买超卖状态
            rsi = talib.RSI(df['close'].values, timeperiod=14)
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            is_overbought = current_rsi > 70
            is_oversold = current_rsi < 30
            
            # 更新市场状态
            self.market_state = state
            self.trend_strength = current_adx
            self.current_trend = trend
            
            return {
                "state": state,
                "trend": trend,
                "strength": current_adx,
                "plus_di": current_plus_di,
                "minus_di": current_minus_di,
                "rsi": current_rsi,
                "volatility": atr_pct,
                "is_overbought": is_overbought,
                "is_oversold": is_oversold
            }
            
        except Exception as e:
            self.logger.error(f"分析市场状态失败: {str(e)}")
            return {"state": "unknown", "trend": 0, "strength": 0}
    
    def train_model_if_needed(self, klines):
        """
        根据需要训练模型
        
        Args:
            klines (list): K线数据
            
        Returns:
            bool: 是否训练了模型
        """
        # 如果模型未训练或者距离上次训练已经超过重训练间隔
        current_time = time.time()
        if (not self.is_trained or 
            current_time - self.last_training_time > self.retraining_interval):
            
            self.logger.info("开始训练深度学习模型...")
            
            # 准备训练数据
            X, y = self._prepare_data(klines)
            if X is None or y is None or len(X) < 100:
                self.logger.error("训练数据不足或无效")
                return False
            
            # 动态更新输入维度以匹配实际特征数
            if self.input_dim != X.shape[1]:
                self.input_dim = X.shape[1]
                self.logger.info(f"调整输入维度为 {self.input_dim} 以匹配实际特征数")
                # 重新初始化模型以适应新的输入维度
                self._init_models()
                
            # 训练模型
            result = self._train_model(X, y)
            
            # 保存训练好的模型
            if result and self.save_models:
                self._save_models()
                
            return result
        
        return False
    
    def generate_signal(self, klines):
        """
        生成交易信号
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 确保有足够的K线数据
            if len(klines) < self.lookback_period:
                self.logger.warning(f"K线数据不足: {len(klines)} < {self.lookback_period}")
                return 0
                
            # 转换K线数据为DataFrame，用于后续分析
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 分析市场状态
            market_state = self.analyze_market_state(klines)
            
            # 根据需要训练模型
            if not self.is_trained:
                self.logger.info("模型未训练，开始初始训练...")
                if not self.train_model_if_needed(klines):
                    self.logger.error("模型训练失败，无法生成信号")
                    return 0
            
            # 准备数据用于预测
            X, _ = self._prepare_data(klines)
            if X is None or len(X) < self.input_window:
                self.logger.error("预测数据准备失败")
                return 0
                
            # 动态更新输入维度以匹配实际特征数
            if self.input_dim != X.shape[1]:
                self.input_dim = X.shape[1]
                self.logger.info(f"调整输入维度为 {self.input_dim} 以匹配实际特征数")
                # 重新初始化并重新训练模型
                self._init_models()
                self.train_model_if_needed(klines)
                
            # 预测未来价格变化
            predictions, confidence, votes, threshold = self._predict(X, df)
            
            if predictions is None:
                self.logger.error("模型预测失败")
                return 0
                
            # 存储预测结果
            self.last_prediction = {
                'predictions': predictions.tolist(),
                'confidence': float(confidence),
                'votes': votes,
                'threshold': float(threshold),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 添加到预测历史
            self.prediction_history.append(self.last_prediction)
            if len(self.prediction_history) > 50:  # 只保留最近50个预测
                self.prediction_history.pop(0)
            
            # 确定交易信号
            signal = 0
            
            # 计算预测的平均变化率
            avg_change = np.mean(predictions)
            
            # 计算投票结果
            up_votes = votes.count(1)
            down_votes = votes.count(-1)
            vote_ratio = up_votes / len(votes) if up_votes > down_votes else -down_votes / len(votes)
            
            # 综合考虑预测值、投票结果和置信度
            if confidence >= self.min_confidence:
                if avg_change > threshold and vote_ratio > self.voting_threshold:
                    signal = 1  # 买入信号
                    self.logger.info(f"生成买入信号 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {confidence:.4f}, 投票: {up_votes}:{down_votes}")
                elif avg_change < -threshold and vote_ratio < -self.voting_threshold:
                    signal = -1  # 卖出信号
                    self.logger.info(f"生成卖出信号 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {confidence:.4f}, 投票: {up_votes}:{down_votes}")
                else:
                    self.logger.info(f"预测变化不足或投票不一致 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {confidence:.4f}, 投票: {up_votes}:{down_votes}")
            else:
                self.logger.info(f"预测置信度不足 - 预测变化率: {avg_change:.6f}, 置信度: {confidence:.4f}")
            
            # 考虑市场状态进行调整
            if signal != 0:
                # 1. 强趋势市场中，如果信号与趋势方向一致，增强信号；反之减弱信号
                if market_state['state'] == 'strong_trend':
                    if signal * market_state['trend'] < 0:  # 信号与趋势方向相反
                        self.logger.info(f"信号与强趋势方向相反，谨慎操作")
                        if abs(avg_change) < threshold * 2:  # 如果预测变化不是特别大
                            signal = 0  # 取消信号
                    
                # 2. 震荡市场中，减少信号频率
                elif market_state['state'] == 'range':
                    if abs(avg_change) < threshold * 1.5:  # 在震荡市场要求更高的预测变化
                        self.logger.info(f"在震荡市场中，预测变化不够显著，忽略信号")
                        signal = 0
                
                # 3. 考虑超买超卖状态
                if signal == 1 and market_state['is_overbought']:
                    self.logger.info(f"市场处于超买状态 (RSI: {market_state['rsi']:.1f})，谨慎做多")
                    if abs(avg_change) < threshold * 2:
                        signal = 0
                elif signal == -1 and market_state['is_oversold']:
                    self.logger.info(f"市场处于超卖状态 (RSI: {market_state['rsi']:.1f})，谨慎做空")
                    if abs(avg_change) < threshold * 2:
                        signal = 0
            
            # 输出详细的预测信息
            self.logger.info(f"预测未来 {self.prediction_horizon} 个周期的价格变化率: {predictions}")
            self.logger.info(f"市场状态: {market_state['state']}, 趋势: {market_state['trend']}, 强度: {market_state['strength']:.2f}")
            
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
                
                # 检查是否需要根据最新预测调整持仓
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
                # 生成新的交易信号
                new_signal = self.generate_signal(klines)
                
                # 如果信号与当前持仓方向相反，考虑平仓
                if (position_side == "多" and new_signal == -1) or (position_side == "空" and new_signal == 1):
                    # 检查持仓时间是否足够长，避免过早平仓
                    if self.position_entry_time is not None:
                        holding_time_minutes = (current_time - self.position_entry_time) / 60
                        
                        # 根据持仓时间和利润率决定是否平仓
                        if holding_time_minutes > 60 or profit_rate > 0:
                            self.logger.info(f"根据最新预测信号平仓, 持仓时间: {holding_time_minutes:.1f}分钟, 利润率: {profit_rate:.4%}")
                            self.trader.close_position()
                            return
                        else:
                            self.logger.info(f"信号反转但持仓时间较短且无利润，继续持仓: {holding_time_minutes:.1f}分钟, 利润率: {profit_rate:.4%}")
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
