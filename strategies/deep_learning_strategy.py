"""
深度学习趋势预测策略 - 15分钟时间框架修复版
修复预测缓存和模型多样性问题
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from datetime import datetime, timedelta
import time
import logging
import random
import talib
import json
import copy
from strategies.base_strategy import BaseStrategy


# 设置不同的模型架构，增加多样性
class LSTMModel(nn.Module):
    """基础LSTM模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output


class BidirectionalLSTMModel(nn.Module):
    """双向LSTM模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.3):
        super(BidirectionalLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output


class LSTMWithAttention(nn.Module):
    """带注意力机制的LSTM模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.3):
        super(LSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        context = self.dropout(context)
        output = self.fc(context)
        
        return output


class CNNLSTMModel(nn.Module):
    """CNN + LSTM混合模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_len=24, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # 1D CNN层用于提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM层处理序列
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层用于预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        # 输入形状变换: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # CNN特征提取
        cnn_out = self.cnn(x)
        
        # 形状变换回: [batch, seq_len, features]
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # LSTM处理
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        
        # 只使用最后一个时间步
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 预测
        output = self.fc(last_output)
        
        return output


class GRUModel(nn.Module):
    """GRU模型，通常比LSTM更简单，训练更快"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.3):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        gru_out, _ = self.gru(x, h0)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output


class FixedEnhancedDeepLearningStrategy15m(BaseStrategy):
    """FixedEnhancedDeepLearningStrategy15m - 修复版增强深度学习策略
    
    修复了预测总是相同的问题，并增加了模型多样性
    """
    
    def __init__(self, trader):
        """初始化修复版深度学习策略"""
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
        self.input_dim = 56               # 输入特征维度（自动调整）
        self.hidden_dim = 128             # 隐藏层维度
        self.num_layers = 2               # LSTM层数
        self.dropout_rate = 0.4           # Dropout比例
        self.learning_rate = 0.001        # 学习率
        self.batch_size = 64              # 批次大小
        self.num_epochs = 100             # 训练轮数
        self.early_stop_patience = 15     # 早停耐心值
        self.weight_decay = 1e-4          # L2正则化权重
        
        # 集成学习参数
        self.num_models = 5               # 集成模型数量
        self.models = []                  # 模型列表
        self.voting_threshold = 0.6       # 投票阈值
        self.model_seeds = [42, 123, 456, 789, 101]  # 不同的随机种子增加模型多样性
        
        # 预测参数
        self.min_confidence = 0.6         # 最小信心阈值
        self.base_prediction_threshold = 0.003  # 基础预测变化率阈值
        self.adaptive_threshold = True    # 是否使用自适应阈值
        self.max_prediction = 0.05        # 最大预测变化阈值
        
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
        self.retraining_interval = 3*3600  # 重新训练间隔(秒)
        self.feature_scalers = {}         # 特征缩放器字典
        self.target_scaler = None         # 目标缩放器
        self.last_prediction = None       # 上次预测结果
        self.position_entry_time = None   # 开仓时间
        self.position_entry_price = None  # 开仓价格
        self.max_profit_reached = 0       # 达到的最大利润
        
        # 预测历史跟踪
        self.prediction_history = []      # 预测历史
        self.max_history_size = 20        # 最大历史记录数
        self.prediction_consistency_threshold = 0.9  # 预测一致性阈值，超过则警告
        self.confidence_adjustment_factor = 0.8     # 连续预测错误时的信任度调整因子
        self.consecutive_wrong_predictions = 0      # 连续错误预测计数
        self.max_consecutive_wrong = 3              # 最大允许连续错误预测
        self.prediction_error_window = 5            # 用于计算预测准确率的窗口大小
        self.prediction_accuracy = []               # 预测准确率历史
        
        # 数据增强和预处理参数
        self.use_data_augmentation = True  # 是否使用数据增强
        self.noise_level = 0.001          # 数据增强时添加的噪声级别
        self.validation_split = 0.2       # 验证集比例
        self.test_split = 0.1             # 测试集比例
        
        # 日志和调试
        self.debug_mode = True            # 调试模式，输出更多信息
        self.model_save_path = 'models'   # 模型保存路径
        self.save_models = True           # 是否保存模型
        self.save_predictions = True      # 是否保存预测结果
        
        # 动态阈值调整
        self.threshold_history = []       # 阈值历史
        self.threshold_adjustment_window = 10  # 阈值调整窗口
        self.current_prediction_threshold = self.base_prediction_threshold  # 当前阈值
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 创建模型保存目录
        if self.save_models and not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            
        # 初始化模型
        self._init_models()
        
        # 市场状态参数
        self.market_state = "unknown"     # 市场状态
        self.trend_strength = 0           # 趋势强度
        self.current_trend = 0            # 当前趋势 (1: 上升, -1: 下降, 0: 中性)
            
    def _generate_model_filename(self, model_idx):
        """生成模型文件名，包含创建时间戳以避免缓存问题"""
        symbol = self.trader.symbol if self.trader.symbol else "unknown"
        timestamp = int(time.time())
        return f"{self.model_save_path}/{symbol}_model_{model_idx}_{timestamp}.pth"
            
    def _init_models(self):
        """初始化不同架构的模型，增加多样性"""
        self.models = []
        model_classes = [
            LSTMModel,
            BidirectionalLSTMModel,
            LSTMWithAttention,
            CNNLSTMModel,
            GRUModel
        ]
        
        for i in range(self.num_models):
            # 选择模型类型，确保使用不同架构
            model_class = model_classes[i % len(model_classes)]
            
            # 为每个模型设置不同种子
            torch.manual_seed(self.model_seeds[i])
            np.random.seed(self.model_seeds[i])
            random.seed(self.model_seeds[i])
            
            # 初始化模型，部分模型需要额外参数
            if model_class == CNNLSTMModel:
                model = model_class(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    output_dim=self.prediction_horizon,
                    seq_len=self.input_window,
                    dropout_rate=self.dropout_rate
                )
            else:
                model = model_class(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    output_dim=self.prediction_horizon,
                    dropout_rate=self.dropout_rate
                )
            
            # 尝试加载已保存的模型
            model_found = False
            if os.path.exists(self.model_save_path):
                for file_name in os.listdir(self.model_save_path):
                    if f"{self.trader.symbol}_model_{i}_" in file_name and file_name.endswith(".pth"):
                        try:
                            model_path = os.path.join(self.model_save_path, file_name)
                            model.load_state_dict(torch.load(model_path, map_location=self.device))
                            model.eval()
                            self.is_trained = True
                            self.logger.info(f"成功加载模型 {i+1} (类型: {model_class.__name__}) 从 {file_name}")
                            model_found = True
                            break
                        except Exception as e:
                            self.logger.error(f"加载模型 {i+1} 失败: {str(e)}")
            
            if not model_found:
                self.logger.info(f"初始化新模型 {i+1} (类型: {model_class.__name__})")
            
            self.models.append(model)
            
        self.logger.info(f"初始化了 {self.num_models} 个多样化预测模型")
        
    def _save_models(self):
        """保存模型到文件，使用新文件名避免缓存问题"""
        if not self.save_models:
            return
            
        # 清理旧模型文件
        if os.path.exists(self.model_save_path):
            for file_name in os.listdir(self.model_save_path):
                if f"{self.trader.symbol}_model_" in file_name and file_name.endswith(".pth"):
                    try:
                        os.remove(os.path.join(self.model_save_path, file_name))
                    except Exception as e:
                        self.logger.error(f"删除旧模型文件失败: {str(e)}")
            
        # 保存新模型
        for i, model in enumerate(self.models):
            try:
                model_path = self._generate_model_filename(i)
                torch.save(model.state_dict(), model_path)
                self.logger.info(f"模型 {i+1} ({model.__class__.__name__}) 已保存到 {model_path}")
            except Exception as e:
                self.logger.error(f"保存模型 {i+1} 失败: {str(e)}")
            
    def _prepare_data(self, klines, force_recalculate=False):
        """
        准备深度学习模型所需的数据
        
        Args:
            klines (list): K线数据
            force_recalculate (bool): 是否强制重新计算
            
        Returns:
            tuple: (特征DataFrame, 标签数组)
        """
        try:
            start_time = time.time()
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
                
            # 如果特征数量与预期不符，更新input_dim
            if X.shape[1] != self.input_dim:
                old_dim = self.input_dim
                self.input_dim = X.shape[1]
                self.logger.info(f"更新输入维度: {old_dim} -> {self.input_dim}")
                
            elapsed_time = time.time() - start_time
            if self.debug_mode and elapsed_time > 1.0:
                self.logger.info(f"数据准备耗时: {elapsed_time:.2f}秒")
                
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
        start_time = time.time()
        
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
        
        # 记录特征数量和处理时间
        elapsed_time = time.time() - start_time
        self.logger.info(f"特征工程完成，共生成 {features.shape[1]} 个特征 (耗时: {elapsed_time:.2f}秒)")
        
        return features
    
    def _normalize_features(self, X, y=None, is_training=False):
        """
        标准化特征和目标
        
        Args:
            X (pandas.DataFrame): 特征
            y (numpy.ndarray): 目标
            is_training (bool): 是否在训练
            
        Returns:
            tuple: (标准化后的特征, 标准化后的目标)
        """
        try:
            # 对于每个特征列使用适当的缩放器
            X_scaled = pd.DataFrame(index=X.index)
            
            for column in X.columns:
                if is_training or column not in self.feature_scalers:
                    # 选择合适的缩放器
                    if 'ratio' in column or 'norm' in column or 'diff' in column or 'position' in column:
                        # 这些特征已经具有相对比例，使用MinMaxScaler
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                    elif 'log_return' in column or 'pct_change' in column or 'roc' in column:
                        # 这些特征可能有较大的异常值，使用RobustScaler
                        scaler = RobustScaler()
                    else:
                        # 其他一般特征使用StandardScaler
                        scaler = StandardScaler()
                    
                    # 训练缩放器
                    self.feature_scalers[column] = scaler
                    X_scaled[column] = scaler.fit_transform(X[column].values.reshape(-1, 1)).flatten()
                else:
                    # 使用已有的缩放器
                    scaler = self.feature_scalers[column]
                    X_scaled[column] = scaler.transform(X[column].values.reshape(-1, 1)).flatten()
            
            # 对目标进行缩放
            if y is not None and is_training:
                self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                y_scaled = self.target_scaler.fit_transform(y)
                return X_scaled, y_scaled
            elif y is not None:
                y_scaled = self.target_scaler.transform(y)
                return X_scaled, y_scaled
            else:
                return X_scaled
                
        except Exception as e:
            self.logger.error(f"特征标准化失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            if y is not None:
                return X, y
            else:
                return X
    
    def _create_sequences(self, X, y=None, is_training=True):
        """
        创建序列数据用于LSTM模型
        
        Args:
            X (pandas.DataFrame): 特征数据
            y (numpy.ndarray): 目标数据，仅在训练时需要
            is_training (bool): 是否为训练模式
            
        Returns:
            tuple: (X序列, y目标)或单个X序列
        """
        # 转为numpy数组
        X_array = X.values if hasattr(X, 'values') else X
        
        if X_array.shape[0] <= self.input_window:
            self.logger.warning(f"数据长度 {X_array.shape[0]} 小于等于输入窗口大小 {self.input_window}")
            
            if is_training:
                return np.array([]), np.array([])
            else:
                return np.array([])
                
        sequences_X = []
        
        # 创建滑动窗口序列
        for i in range(len(X_array) - self.input_window + 1):
            sequences_X.append(X_array[i:i + self.input_window])
            
        # 转换为numpy数组
        sequences_X = np.array(sequences_X)
        
        if is_training and y is not None:
            sequences_y = []
            for i in range(len(X_array) - self.input_window + 1):
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
        for _ in range(1):  # 生成1个噪声版本
            noise = np.random.normal(0, noise_scale, X_seq.shape)
            noisy_X = X_seq + noise
            augmented_X.append(noisy_X)
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
            y_pred_original = self.target_scaler.inverse_transform(y_pred)
            y_true_original = self.target_scaler.inverse_transform(y_test)
            
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
            self.logger.info(f"开始训练模型 (特征维度: {X.shape[1]}, 数据量: {X.shape[0]})")
            
            # 标准化特征和目标
            X_scaled, y_scaled = self._normalize_features(X, y, is_training=True)
            
            # 划分训练集、验证集和测试集
            total_samples = len(X_scaled)
            test_size = int(total_samples * self.test_split)
            val_size = int(total_samples * self.validation_split)
            train_size = total_samples - test_size - val_size
            
            # 确保数据量足够
            if train_size < 100 or val_size < 30 or test_size < 30:
                self.logger.error(f"训练数据不足: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
                return False
            
            # 划分索引
            all_indices = np.arange(total_samples)
            
            # 使用随机种子打乱索引
            np.random.seed(42)
            np.random.shuffle(all_indices)
            
            train_indices = all_indices[:train_size]
            val_indices = all_indices[train_size:train_size+val_size]
            test_indices = all_indices[train_size+val_size:]
            
            # 划分数据
            X_train = X_scaled.iloc[train_indices]
            y_train = y_scaled[train_indices]
            X_val = X_scaled.iloc[val_indices]
            y_val = y_scaled[val_indices]
            X_test = X_scaled.iloc[test_indices]
            y_test = y_scaled[test_indices]
            
            # 创建序列
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, is_training=True)
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, is_training=True)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, is_training=True)
            
            if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
                self.logger.error("序列化后的数据为空")
                return False
            
            self.logger.info(f"训练集大小: {X_train_seq.shape}, 验证集大小: {X_val_seq.shape}, 测试集大小: {X_test_seq.shape}")
            
            # 数据增强 (仅对训练集)
            X_train_aug, y_train_aug = self._data_augmentation(X_train_seq, y_train_seq)
            
            # 训练每个模型
            for i, model in enumerate(self.models):
                self.logger.info(f"训练模型 {i+1}/{len(self.models)} (类型: {model.__class__.__name__})")
                
                # 使用不同的随机种子
                torch.manual_seed(self.model_seeds[i])
                np.random.seed(self.model_seeds[i])
                
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
                    
                    # 每隔5个epoch打印一次训练情况
                    if (epoch+1) % 5 == 0 or epoch == 0 or epoch == self.num_epochs-1:
                        self.logger.info(f"模型 {i+1}, Epoch [{epoch+1}/{self.num_epochs}], "
                                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # 早停检查
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
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
            
            # 保存模型
            self._save_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_adaptive_threshold(self, df):
        """
        计算自适应预测阈值，根据近期市场波动性调整
        
        Args:
            df (pandas.DataFrame): K线数据
            
        Returns:
            float: 自适应阈值
        """
        try:
            # 计算近期ATR
            high = df['high'].values[-20:]
            low = df['low'].values[-20:]
            close = df['close'].values[-20:]
            
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_pct = atr[-1] / close[-1]
            
            # 波动倍数，最小0.1，最大0.5
            volatility_factor = min(0.5, max(0.1, atr_pct * 0.5))
            
            # 计算自适应阈值
            adaptive_threshold = self.base_prediction_threshold * (1 + volatility_factor)
            
            # 确保阈值在合理范围内
            adaptive_threshold = min(0.01, max(0.001, adaptive_threshold))
            
            self.logger.info(f"自适应阈值: {adaptive_threshold:.6f} (基础阈值: {self.base_prediction_threshold:.6f}, "
                          f"波动因子: {volatility_factor:.4f}, ATR百分比: {atr_pct:.4f})")
                
            return adaptive_threshold
            
        except Exception as e:
            self.logger.error(f"计算自适应阈值失败: {str(e)}")
            return self.base_prediction_threshold
    
    def _check_prediction_consistency(self, current_prediction, last_predictions):
        """
        检查当前预测与历史预测的一致性，防止模型预测卡住
        
        Args:
            current_prediction: 当前预测值
            last_predictions: 历史预测值列表
            
        Returns:
            bool: 是否一致性过高（警告信号）
        """
        if not last_predictions or len(last_predictions) < 2:
            return False
            
        # 比较当前预测和上一次预测
        last_prediction = last_predictions[-1]['predictions']
        
        # 计算预测差异
        diff = np.mean(np.abs(np.array(current_prediction) - np.array(last_prediction)))
        
        # 如果差异极小，认为预测卡住了
        if diff < 0.0001:
            self.logger.warning(f"预测差异极小: {diff:.8f}，可能存在预测缓存问题")
            return True
            
        return False
    
    def _predict(self, X, df=None):
        """
        使用训练好的模型集合进行预测
        
        Args:
            X (pandas.DataFrame): 特征数据
            df (pandas.DataFrame): 原始K线数据，用于计算自适应阈值
            
        Returns:
            tuple: (预测值, 预测置信度, 投票结果, 自适应阈值)
        """
        try:
            start_time = time.time()
            
            if not self.is_trained or len(self.models) == 0:
                self.logger.error("模型未训练，无法进行预测")
                return None, 0, [], self.base_prediction_threshold
                
            # 标准化特征
            X_scaled = self._normalize_features(X)
            
            # 创建序列
            X_seq = self._create_sequences(X_scaled, is_training=False)
            
            if len(X_seq) == 0:
                self.logger.error("序列化数据为空，无法预测")
                return None, 0, [], self.base_prediction_threshold
                
            # 只取最后一个序列进行预测
            X_last = X_seq[-1:]
            X_tensor = torch.FloatTensor(X_last).to(self.device)
            
            # 收集所有模型的预测
            all_predictions = []
            model_predictions = []  # 保存每个模型的详细预测
            
            for i, model in enumerate(self.models):
                # 将模型移到正确的设备
                model = model.to(self.device)
                model.eval()
                
                with torch.no_grad():
                    # 确保输入尺寸正确
                    prediction = model(X_tensor)
                    
                    # 将预测转换回原始比例
                    prediction_np = prediction.cpu().numpy()
                    if self.target_scaler is not None:
                        prediction_np = self.target_scaler.inverse_transform(prediction_np)
                        
                    # 保存详细的单模型预测
                    model_predictions.append({
                        'model_idx': i,
                        'model_type': model.__class__.__name__,
                        'raw_prediction': prediction_np[0].tolist(),
                    })
                    
                    all_predictions.append(prediction_np[0])
                
                # 将模型移回CPU以节省内存
                model = model.cpu()
            
            # 计算平均预测值和标准差
            all_predictions = np.array(all_predictions)
            avg_prediction = np.mean(all_predictions, axis=0)
            std_prediction = np.std(all_predictions, axis=0)
            
            # 确保预测值在合理范围内
            avg_prediction = np.clip(avg_prediction, -self.max_prediction, self.max_prediction)
            
            # 计算置信度 (方式1: 基于标准差)
            confidence_std = 1 / (1 + np.mean(std_prediction) * 10)
            
            # 计算每个模型对未来趋势方向的投票
            votes = []
            for pred in all_predictions:
                # 剪裁预测值到合理范围
                pred = np.clip(pred, -self.max_prediction, self.max_prediction)
                votes.append(1 if np.mean(pred) > 0 else -1)
            
            # 方式2: 根据一致性投票计算置信度
            agreement_ratio = max(votes.count(1), votes.count(-1)) / len(votes)
            confidence_vote = agreement_ratio
            
            # 综合计算置信度 (给投票一致性更高的权重)
            confidence = 0.3 * confidence_std + 0.7 * confidence_vote
            
            # 计算自适应阈值
            threshold = self._get_adaptive_threshold(df) if df is not None else self.base_prediction_threshold
            
            # 检查预测一致性
            is_consistent = self._check_prediction_consistency(avg_prediction, self.prediction_history)
            if is_consistent:
                # 如果预测一致性过高，降低置信度
                confidence *= 0.8
                self.logger.warning(f"因预测一致性过高，降低置信度至 {confidence:.4f}")
            
            # 记录详细的预测信息
            prediction_detail = {
                'timestamp': time.time(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': avg_prediction.tolist(),
                'confidence': float(confidence),
                'votes': votes,
                'threshold': float(threshold),
                'std_prediction': std_prediction.tolist(),
                'model_predictions': model_predictions,
                'elapsed_time': time.time() - start_time
            }
            
            # 添加到预测历史
            self.prediction_history.append(prediction_detail)
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history.pop(0)
            
            # 记录预测性能
            if self.debug_mode:
                self.logger.info(f"预测耗时: {prediction_detail['elapsed_time']:.4f}秒")
                # 输出每个模型的预测
                for i, model_pred in enumerate(model_predictions):
                    model_type = model_pred['model_type']
                    pred_values = np.array(model_pred['raw_prediction'])
                    pred_avg = np.mean(pred_values)
                    pred_dir = "买入" if pred_avg > 0 else "卖出"
                    self.logger.info(f"模型 {i+1} ({model_type}) 预测: {pred_avg:.6f} ({pred_dir})")
            
            return avg_prediction, confidence, votes, threshold
                
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, 0, [], self.base_prediction_threshold
    
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
                "is_overbought": is_overbought,
                "is_oversold": is_oversold
            }
            
        except Exception as e:
            self.logger.error(f"分析市场状态失败: {str(e)}")
            return {"state": "unknown", "trend": 0, "strength": 0}
    
    def validate_prediction_vs_reality(self, current_price, last_price, last_prediction):
        """
        将上一次的预测与实际价格变化对比，用于动态调整预测的可信度
        
        Args:
            current_price (float): 当前价格
            last_price (float): 上一次价格
            last_prediction (dict): 上一次预测
            
        Returns:
            bool: 预测是否与实际变化方向一致
        """
        if last_prediction is None:
            return None
            
        # 计算实际变化率
        actual_change = (current_price - last_price) / last_price
        
        # 获取上一次预测的平均变化率
        predicted_change = np.mean(last_prediction.get('predictions', [0]))
        
        # 判断方向是否一致
        actual_direction = 1 if actual_change > 0 else -1
        predicted_direction = 1 if predicted_change > 0 else -1
        
        # 方向一致性
        direction_match = actual_direction == predicted_direction
        
        # 更新计数器
        if direction_match:
            self.consecutive_wrong_predictions = 0
        else:
            self.consecutive_wrong_predictions += 1
            self.logger.warning(f"预测方向与实际不符: 预测={predicted_change:.6f}, 实际={actual_change:.6f}, 连续错误次数={self.consecutive_wrong_predictions}")
        
        # 记录预测准确率
        self.prediction_accuracy.append(direction_match)
        if len(self.prediction_accuracy) > self.prediction_error_window:
            self.prediction_accuracy.pop(0)
        
        # 计算近期预测准确率
        recent_accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy) if self.prediction_accuracy else 0.5
        
        # 记录对比信息
        self.logger.info(f"预测 vs 实际: 预测方向={'上涨' if predicted_direction > 0 else '下跌'} ({predicted_change:.6f}), "
                      f"实际方向={'上涨' if actual_direction > 0 else '下跌'} ({actual_change:.6f}), "
                      f"方向{'一致' if direction_match else '不一致'}, 准确率={recent_accuracy:.2f}")
        
        return direction_match
    
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
        
        # 检查是否需要强制重训练
        force_retrain = False
        if self.consecutive_wrong_predictions >= self.max_consecutive_wrong:
            self.logger.warning(f"连续 {self.consecutive_wrong_predictions} 次预测错误，强制重新训练模型")
            force_retrain = True
            self.consecutive_wrong_predictions = 0
            
        if (not self.is_trained or force_retrain or 
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
            start_time = time.time()
            
            # 确保有足够的K线数据
            if len(klines) < self.lookback_period:
                self.logger.warning(f"K线数据不足: {len(klines)} < {self.lookback_period}")
                return 0
                
            # 转换K线数据为DataFrame，用于后续分析
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 获取当前价格和上一次价格
            current_price = df['close'].iloc[-1]
            last_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            
            # 验证上一次预测准确性
            if self.last_prediction:
                self.validate_prediction_vs_reality(current_price, last_price, self.last_prediction)
            
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
            
            # 确定交易信号
            signal = 0
            
            # 计算预测的平均变化率
            avg_change = np.mean(predictions)
            
            # 计算投票结果
            up_votes = votes.count(1)
            down_votes = votes.count(-1)
            vote_ratio = up_votes / len(votes) if up_votes > down_votes else -down_votes / len(votes)
            
            # 如果连续预测错误，降低置信度
            adjusted_confidence = confidence
            if self.consecutive_wrong_predictions > 0:
                adjustment_factor = self.confidence_adjustment_factor ** self.consecutive_wrong_predictions
                adjusted_confidence *= adjustment_factor
                self.logger.info(f"调整信心度: {confidence:.4f} -> {adjusted_confidence:.4f} (因连续 {self.consecutive_wrong_predictions} 次预测错误)")
            
            # 综合考虑预测值、投票结果和置信度
            if adjusted_confidence >= self.min_confidence:
                if avg_change > threshold and vote_ratio > self.voting_threshold:
                    signal = 1  # 买入信号
                    self.logger.info(f"生成买入信号 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {adjusted_confidence:.4f}, 投票: {up_votes}:{down_votes}")
                elif avg_change < -threshold and vote_ratio < -self.voting_threshold:
                    signal = -1  # 卖出信号
                    self.logger.info(f"生成卖出信号 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {adjusted_confidence:.4f}, 投票: {up_votes}:{down_votes}")
                else:
                    self.logger.info(f"预测变化不足或投票不一致 - 预测变化率: {avg_change:.6f} (阈值: {threshold:.6f}), "
                                  f"置信度: {adjusted_confidence:.4f}, 投票: {up_votes}:{down_votes}")
            else:
                self.logger.info(f"预测置信度不足 - 预测变化率: {avg_change:.6f}, 调整后置信度: {adjusted_confidence:.4f} (原始置信度: {confidence:.4f})")
            
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
                if signal == 1 and market_state.get('is_overbought', False):
                    self.logger.info(f"市场处于超买状态 (RSI: {market_state.get('rsi', 0):.1f})，谨慎做多")
                    if abs(avg_change) < threshold * 2:
                        signal = 0
                elif signal == -1 and market_state.get('is_oversold', False):
                    self.logger.info(f"市场处于超卖状态 (RSI: {market_state.get('rsi', 0):.1f})，谨慎做空")
                    if abs(avg_change) < threshold * 2:
                        signal = 0
            
            # 输出详细的预测信息
            self.logger.info(f"预测未来 {self.prediction_horizon} 个周期的价格变化率: {predictions}")
            self.logger.info(f"市场状态: {market_state['state']}, 趋势: {market_state['trend']}, 强度: {market_state['strength']:.2f}")
            
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                self.logger.info(f"信号生成耗时: {elapsed_time:.2f}秒")
            
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
