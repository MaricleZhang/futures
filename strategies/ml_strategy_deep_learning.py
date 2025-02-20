import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from strategies.ml_strategy import MLStrategy
from utils.logger import Logger
import config
import talib
from sklearn.preprocessing import StandardScaler
import time
import logging

class DeepLearningModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # 增加网络深度和宽度
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3)
        
        # 添加Dropout和BatchNorm来防止过拟合
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.layer4(x)))
        x = self.dropout(x)
        x = F.relu(self.layer5(x))
        x = F.softmax(self.output(x), dim=1)
        return x

class DeepLearningStrategy(MLStrategy):
    """DeepTradeMaster - 深度学习交易策略
    
    一个基于深度神经网络的期货交易策略模型，使用多层感知机(MLP)结构，
    结合技术指标、市场情绪和波动特征来预测市场走势。
    
    特点：
    1. 多特征融合：整合技术指标、市场情绪和波动性等多维度特征
    2. 三分类预测：买入、卖出、持仓三种行为的概率预测
    3. 动态阈值：使用置信度阈值过滤低置信度的交易信号
    4. 早停机制：防止过拟合，提高模型泛化能力
    """
    
    MODEL_NAME = "DeepTradeMaster"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化深度学习策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 深度学习特定参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.learning_rate = 0.001
        self.n_epochs = 20
        self.confidence_threshold = 0.9
        
        # 初始化StandardScaler
        self.scaler = StandardScaler()
        
        # 初始化模型
        self.initialize_model()
        
        # 获取历史数据进行训练
        try:
            # 主要使用1小时周期的数据
            self.logger.info("获取历史K线数据...")
            historical_data = self.trader.get_klines(interval='1h', limit=1000)
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根K线数据")
                # 训练模型
                if self.train_model(historical_data):
                    self.logger.info("模型训练成功")
                else:
                    self.logger.error("模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
                
        except Exception as e:
            self.logger.error(f"初始化深度学习策略失败: {str(e)}")

    def initialize_model(self):
        """初始化深度学习模型"""
        try:
            # 获取输入维度
            feature_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'STOCH_K', 'STOCH_D',
                'returns', 'log_returns', 'volatility', 'trend_strength', 'volume_ratio',
                'ATR', 'BB_width', 'BB_position', 'market_sentiment',
                'volume_momentum', 'volume_trend', 'price_momentum',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                'ADX', 'PLUS_DI', 'MINUS_DI',
                'NATR', 'MOM', 'ROC',
                'OBV', 'AD',
                'BB_upper', 'BB_middle', 'BB_lower',
                'price_volatility',
                'support', 'resistance', 'price_to_support', 'price_to_resistance'
            ]
            input_size = len(feature_columns)
            self.logger.info(f"模型输入维度: {input_size}")
            
            # 创建模型实例
            self.model = DeepLearningModel(input_size).to(self.device)
            
            # 定义损失函数和优化器
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            self.logger.info("深度学习模型初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化深度学习模型失败: {str(e)}")

    def prepare_features(self, klines):
        """准备特征数据，增加深度学习特定的特征"""
        try:
            # 首先将K线数据转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 确保数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算基础技术指标
            try:
                # 价格趋势特征
                df['EMA_5'] = talib.EMA(df['close'], timeperiod=5)
                df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
                df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
                df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
                
                # 趋势强度
                df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
                df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
                
                # 波动率指标
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
                
                # 动量指标
                df['MOM'] = talib.MOM(df['close'], timeperiod=10)
                df['ROC'] = talib.ROC(df['close'], timeperiod=10)
                
                # 成交量分析
                df['OBV'] = talib.OBV(df['close'], df['volume'])
                df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
                
                # RSI和MACD
                df['RSI'] = talib.RSI(df['close'], timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(df['close'])
                df['MACD'] = macd
                df['MACD_SIGNAL'] = macd_signal
                df['MACD_HIST'] = macd_hist
                
                # 随机指标
                df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
                
                # 布林带
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                df['BB_upper'] = upper
                df['BB_middle'] = middle
                df['BB_lower'] = lower
                df['BB_width'] = (upper - lower) / middle
                df['BB_position'] = (df['close'] - lower) / (upper - lower)
                
                # 市场状态特征
                df['trend_strength'] = abs(df['EMA_20'] - df['EMA_50']) / df['EMA_50']
                df['price_volatility'] = df['ATR'] / df['close']
                df['volume_trend'] = df['volume'] / df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
                
                # 动量和趋势
                df['price_momentum'] = df['close'].pct_change(5)
                df['volume_momentum'] = df['volume'].pct_change(5)
                
                # 市场情绪
                df['market_sentiment'] = (
                    (df['RSI'] / 100) * 0.3 +  # RSI贡献
                    (df['BB_position']) * 0.3 +  # 布林带位置贡献
                    (np.where(df['MACD'] > df['MACD_SIGNAL'], 1, -1)) * 0.2 +  # MACD贡献
                    (df['ROC'] / 100) * 0.2  # ROC贡献
                )
                
                # 计算支撑和阻力位
                df['support'] = df['low'].rolling(20).min()
                df['resistance'] = df['high'].rolling(20).max()
                df['price_to_support'] = (df['close'] - df['support']) / df['support']
                df['price_to_resistance'] = (df['resistance'] - df['close']) / df['close']
                
                # 计算收益率
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close']/df['close'].shift(1))
                
                # 计算波动率
                df['volatility'] = df['returns'].rolling(window=20).std()
                
                self.logger = self.get_logger()
                self.logger.info("计算技术指标成功")
            except Exception as e:
                self.logger.error(f"计算技术指标失败: {str(e)}")
                return pd.DataFrame()

            # 填充NaN值
            df = df.bfill().ffill()
            self.logger.info("填充NaN值成功")
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备深度学习特征时出错: {str(e)}")
            return pd.DataFrame()

    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            # 准备特征数据
            df = self.prepare_features(klines)
            if df.empty:
                self.logger.error("特征数据为空")
                return None
            
            # 检查是否所有特征都存在
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                self.logger.error(f"缺少特征列: {missing_features}")
                self.logger.info(f"现有的列: {list(df.columns)}")
                return None
            
            # 获取最新的特征数据
            latest_features = df[self.feature_columns].iloc[-1:].values
            
            # 使用训练好的scaler进行标准化
            if not hasattr(self.scaler, 'mean_'):
                self.logger.error("StandardScaler未经过训练")
                return None
                
            latest_features = self.scaler.transform(latest_features)
            
            # 转换为PyTorch张量
            features_tensor = torch.FloatTensor(latest_features).to(self.device)
            
            # 生成预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # 获取预测概率
                probs = probabilities[0]
                max_prob = torch.max(probs).item()
                
                self.logger = self.get_logger()
                
                # 记录预测概率（0=卖出，1=买入，2=持仓）
                self.logger.info(f"预测概率: 卖出={probs[0]:.4f}, 买入={probs[1]:.4f}, 持仓={probs[2]:.4f}")
                
                # 记录市场状态
                self.logger.info(f"市场状态: RSI={df['RSI'].iloc[-1]:.2f}, "
                               f"MACD={df['MACD'].iloc[-1]:.2f}, "
                               f"ADX={df['ADX'].iloc[-1]:.2f}, "
                               f"波动率={df['volatility'].iloc[-1]:.4f}, "
                               f"趋势强度={df['trend_strength'].iloc[-1]:.4f}, "
                               f"市场情绪={df['market_sentiment'].iloc[-1]:.4f}")
                
                # 根据置信度阈值生成信号
                if max_prob >= self.confidence_threshold:
                    signal = torch.argmax(probs).item()
                    # 将模型输出转换为交易信号（1=买入，-1=卖出，0=持仓）
                    signal = 1 if signal == 1 else (-1 if signal == 0 else 0)
                else:
                    self.logger.info(f"预测置信度{max_prob:.4f}低于阈值{self.confidence_threshold}，保持观望")
                    signal = 0  # 保持观望
                    
                return signal
                
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            return None

    def update_metrics(self, signal, actual_return):
        """更新模型评估指标
        
        Args:
            signal: 模型生成的信号 (1: 买入, -1: 卖出, 0: 持仓)
            actual_return: 实际收益率
        """
        if signal == 0:  # 如果没有交易，不更新指标
            return
            
        self.metrics['total_trades'] += 1
        
        # 根据信号和实际收益计算是否盈利
        trade_profit = actual_return * signal  # 如果做空，需要反转收益
        
        if trade_profit > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['total_profit'] += trade_profit
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['total_loss'] += abs(trade_profit)
            
        # 更新胜率
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # 更新盈亏比
        if self.metrics['total_loss'] > 0:
            self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
            
        # 更新平均盈亏
        if self.metrics['winning_trades'] > 0:
            self.metrics['avg_profit'] = self.metrics['total_profit'] / self.metrics['winning_trades']
        if self.metrics['losing_trades'] > 0:
            self.metrics['avg_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']
            
        # 记录预测结果
        self.metrics['predictions'].append({
            'signal': signal,
            'return': actual_return,
            'profit': trade_profit
        })
        
        # 定期输出评估指标
        if self.metrics['total_trades'] % 10 == 0:  # 每10笔交易输出一次
            self.log_metrics()
            
    def log_metrics(self):
        """输出评估指标"""
        self.logger.info("=== 模型评估指标 ===")
        self.logger.info(f"总交易次数: {self.metrics['total_trades']}")
        self.logger.info(f"胜率: {self.metrics['win_rate']:.2%}")
        self.logger.info(f"盈亏比: {self.metrics['profit_factor']:.2f}")
        self.logger.info(f"平均盈利: {self.metrics['avg_profit']:.4f}")
        self.logger.info(f"平均亏损: {self.metrics['avg_loss']:.4f}")
        self.logger.info(f"总盈利: {self.metrics['total_profit']:.4f}")
        self.logger.info(f"总亏损: {self.metrics['total_loss']:.4f}")
        self.logger.info("==================")

    def train_model(self, historical_data):
        """训练深度学习模型"""
        try:
            # 准备训练数据
            df = self.prepare_features(historical_data)
            if df.empty:
                self.logger.error("准备训练数据失败")
                return False
                
            # 生成标签（0=卖出，1=买入，2=持仓）
            future_returns = df['close'].shift(-1) / df['close'] - 1
            labels = np.full(len(df), 2)  # 默认持仓
            labels[future_returns > 0.001] = 1  # 买入
            labels[future_returns < -0.001] = 0  # 卖出
            
            # 删除最后一行（因为没有未来数据）
            df = df.iloc[:-1]
            labels = labels[:-1]
            
            # 准备特征数据
            feature_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'STOCH_K', 'STOCH_D',
                'returns', 'log_returns', 'volatility', 'trend_strength', 'volume_ratio',
                'ATR', 'BB_width', 'BB_position', 'market_sentiment',
                'volume_momentum', 'volume_trend', 'price_momentum',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                'ADX', 'PLUS_DI', 'MINUS_DI',
                'NATR', 'MOM', 'ROC',
                'OBV', 'AD',
                'BB_upper', 'BB_middle', 'BB_lower',
                'price_volatility',
                'support', 'resistance', 'price_to_support', 'price_to_resistance'
            ]
            features = df[feature_columns].values
            
            # 保存特征列名，用于后续预测
            self.feature_columns = feature_columns
            
            # 标准化特征
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
            
            # 转换为PyTorch张量
            features_tensor = torch.FloatTensor(features)
            labels_tensor = torch.LongTensor(labels)
            
            # 创建数据集和数据加载器
            dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)
            
            # 初始化早停
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            best_model_state = None
            
            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
            
            # 训练模型
            for epoch in range(self.n_epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # 验证阶段
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_labels)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        total += batch_labels.size(0)
                        correct += predicted.eq(batch_labels).sum().item()
                
                # 计算平均损失
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                accuracy = 100. * correct / total
                
                self.logger.info(f'Epoch {epoch+1}/{self.n_epochs}:')
                self.logger.info(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
                
                # 更新学习率
                scheduler.step(avg_val_loss)
                
                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f'Early stopping at epoch {epoch+1}')
                        break
            
            # 加载最佳模型状态
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            
            self.logger.info("模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"训练深度学习模型失败: {str(e)}")
            return False
