import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import talib
from sklearn.preprocessing import StandardScaler
import time
import logging
from strategies.base_strategy import BaseStrategy
import config
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

class DeepLearningStrategy(BaseStrategy):
    """DeepLearningStrategy - 深度学习交易策略
    
    基于深度神经网络的期货交易策略，使用1分钟K线数据，
    结合技术指标、市场微观结构和波动特征来预测市场走势。
    
    特点：
    1. 深度特征：使用多层神经网络自动学习特征表示
    2. 批归一化：使用BatchNorm加速训练并提高模型稳定性
    3. Dropout正则化：防止过拟合
    4. 动态止损：基于市场波动性动态调整止损位
    5. 分钟级实时预测：每分钟更新一次预测
    """
    
    MODEL_NAME = "DeepLearning"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化深度学习策略"""
        super().__init__(trader)
        
        # 设置设备(GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # K线设置
        self.kline_interval = '1m'
        self.check_interval = 60
        self.predict_periods = 15  # 预测未来3根K线
        
        # 模型参数
        self.input_size = 30
        self.batch_size = 64
        self.learning_rate = 0.0005
        self.n_epochs = 100
        
        # 交易参数
        self.confidence_threshold = 0.55
        self.position_holding_time = self.predict_periods  # 持仓时间与预测周期一致
        self.stop_loss_pct = 0.01
        self.take_profit_pct = 0.02
        
        # 数据预处理
        self.scaler = StandardScaler()
        self.lookback = 100
        self.training_data_size = 1000
        self.training_lookback = 1000
        self.retraining_interval = 1000
        self.last_training_time = 0
        
        # 初始化模型
        self.model = None
        self.optimizer = None
        self.initialize_model()
        
        # 模型评估指标
        self.training_losses = []
        self.validation_losses = []
        self.prediction_accuracy = []
        
        # 获取初始训练数据
        self.initial_training()
    
    def initialize_model(self):
        """初始化深度学习模型"""
        class DeepTradingModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layer1 = nn.Linear(input_size, 256)
                self.layer2 = nn.Linear(256, 128)
                self.layer3 = nn.Linear(128, 64)
                self.layer4 = nn.Linear(64, 32)
                self.output = nn.Linear(32, 3)  # 改为3分类：0=下跌，1=盘整，2=上涨
                self.dropout = nn.Dropout(0.3)
                self.layer_norm1 = nn.LayerNorm(256)
                self.layer_norm2 = nn.LayerNorm(128)
                self.layer_norm3 = nn.LayerNorm(64)
                self.layer_norm4 = nn.LayerNorm(32)
                
            def forward(self, x):
                x = self.layer_norm1(F.leaky_relu(self.layer1(x)))
                x = self.dropout(x)
                x = self.layer_norm2(F.leaky_relu(self.layer2(x)))
                x = self.dropout(x)
                x = self.layer_norm3(F.leaky_relu(self.layer3(x)))
                x = self.dropout(x)
                x = self.layer_norm4(F.leaky_relu(self.layer4(x)))
                x = self.dropout(x)
                x = F.softmax(self.output(x), dim=1)
                return x
        
        self.model = DeepTradingModel(self.input_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

    def prepare_features(self, df):
        """准备特征数据
        
        Args:
            df (pd.DataFrame): K线数据
            
        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        df = df.copy()
        
        # 价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # 技术指标
        df['rsi'] = talib.RSI(df['close'].values)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'].values)
        df['slowk'], df['slowd'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
        
        # 波动性指标
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 价格趋势
        for period in [5, 10, 20]:
            df[f'ma_{period}'] = talib.MA(df['close'].values, timeperiod=period)
            df[f'ma_{period}_slope'] = df[f'ma_{period}'].pct_change()
        
        # 成交量特征
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 动量指标
        df['mom'] = talib.MOM(df['close'].values, timeperiod=10)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        # 删除NaN值
        df = df.dropna()
        
        return df
    
    def generate_labels(self, df):
        """生成训练标签
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            
        Returns:
            np.array: 标签数组，0=下跌趋势，1=盘整，2=上涨趋势
        """
        # 计算未来predict_periods期的价格变化百分比
        future_returns = (df['close'].shift(-self.predict_periods) - df['close']) / df['close']
        
        # 定义趋势阈值
        trend_threshold = 0.005  # 0.5%的价格变动作为趋势判断阈值
        
        # 生成趋势标签
        labels = np.zeros(len(df))
        labels[future_returns > trend_threshold] = 2  # 上涨趋势
        labels[future_returns < -trend_threshold] = 0  # 下跌趋势
        labels[(future_returns >= -trend_threshold) & (future_returns <= trend_threshold)] = 1  # 盘整
        
        # 移除最后predict_periods个标签，因为它们没有对应的未来数据
        labels = labels[:-self.predict_periods]
        
        return labels.astype(np.int64)
    
    def train_model(self, features, labels):
        """训练模型
        
        Args:
            features (np.ndarray): 特征数据
            labels (np.ndarray): 标签数据
            
        Returns:
            tuple: (训练损失, 准确率)
        """
        features = torch.FloatTensor(features).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        dataset = TensorDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.batch_size = min(self.batch_size, len(train_dataset), len(val_dataset))
        if self.batch_size < 2:
            self.batch_size = 2
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        best_val_loss = float('inf')
        best_accuracy = 0.0
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.n_epochs):
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                if len(batch_features) < 2:
                    continue
                    
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    if len(batch_features) < 2:
                        continue
                        
                    outputs = self.model(batch_features)
                    val_loss += self.criterion(outputs, batch_labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            if total == 0:
                continue
                
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.logger.info(f"早停触发，在epoch {epoch+1}/{self.n_epochs}")
                break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.n_epochs}] "
                               f"Train Loss: {train_loss/len(train_loader):.4f} "
                               f"Val Loss: {val_loss:.4f} "
                               f"Accuracy: {accuracy:.2f}%")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                if len(batch_features) < 2:
                    continue
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        if total == 0:
            final_accuracy = 0
        else:
            final_accuracy = 100 * correct / total
        
        return train_loss / len(train_loader), final_accuracy
    
    def predict(self, features):
        """生成预测
        
        Args:
            features (np.array): 特征数据
            
        Returns:
            np.array: 预测结果和置信度
        """
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            predictions = self.model(features)
            
        return predictions.cpu().numpy()
    
    def initial_training(self):
        """初始训练"""
        try:
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_data_size)
            if len(klines) < 100:
                self.logger.error(f"获取历史数据不足，当前数据量: {len(klines)}")
                return
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            features_df = self.prepare_features(df)
            if len(features_df) < 50:
                self.logger.error(f"有效特征数据不足，当前数据量: {len(features_df)}")
                return
            
            numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
            self.input_size = len(numeric_features.columns)
            self.logger.info(f"特征维度: {self.input_size}, 预测周期: {self.predict_periods}根K线")
                
            self.initialize_model()
                
            labels = self.generate_labels(features_df)
            
            numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
            features = self.scaler.fit_transform(numeric_features.iloc[:-self.predict_periods])
            
            if len(features) != len(labels):
                self.logger.error(f"特征和标签数量不匹配: 特征={len(features)}, 标签={len(labels)}")
                return
                
            self.logger.info(f"特征列: {', '.join(numeric_features.columns)}")
            
            train_loss, final_accuracy = self.train_model(features, labels)
            self.logger.info(f"初始训练完成，训练损失: {train_loss:.4f} 准确率: {final_accuracy:.2f}%")
            
        except Exception as e:
            self.logger.error(f"初始训练失败: {str(e)}")
    
    def monitor_position(self):
        """监控持仓状态并生成交易信号"""
        try:
            current_time = int(time.time())
            if current_time - self.last_training_time > self.retraining_interval:
                self.logger.info("开始重新训练模型...")
                klines = self.trader.get_klines(
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                if len(klines) >= 100:
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.astype({
                        'open': 'float64',
                        'high': 'float64',
                        'low': 'float64',
                        'close': 'float64',
                        'volume': 'float64'
                    })
                    
                    features_df = self.prepare_features(df)
                    if len(features_df) >= 50:
                        numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
                        self.input_size = len(numeric_features.columns)
                        self.initialize_model()
                        
                        labels = self.generate_labels(features_df)
                        features = self.scaler.fit_transform(numeric_features.iloc[:-self.predict_periods])
                        
                        if len(features) == len(labels):
                            train_loss, final_accuracy = self.train_model(features, labels)
                            self.logger.info(f"模型重新训练完成，训练损失: {train_loss:.4f} 准确率: {final_accuracy:.2f}%")
                            self.last_training_time = current_time
            
            signal = self.generate_signal()
            if signal is None:
                return
                
            position = self.trader.get_position()
            current_price = float(self.trader.get_latest_price())
            
            if signal == 1 and position['size'] <= 0:
                self.trader.open_long(current_price, self.stop_loss_pct, self.take_profit_pct)
                self.logger.info(f"开多仓 价格: {current_price:.2f}")
            elif signal == -1 and position['size'] >= 0:
                self.trader.open_short(current_price, self.stop_loss_pct, self.take_profit_pct)
                self.logger.info(f"开空仓 价格: {current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"监控持仓出错: {str(e)}")

    def generate_signal(self, klines=None):
        """生成交易信号
        
        Args:
            klines (list, optional): K线数据. Defaults to None.
            
        Returns:
            int: 1表示买入信号，-1表示卖出信号，None表示无信号
        """
        try:
            if klines is None:
                klines = self.trader.get_klines(interval=self.kline_interval, 
                                              limit=self.lookback + self.predict_periods)
            
            if len(klines) < self.lookback:
                self.logger.warning(f"K线数据不足，当前: {len(klines)}, 需要: {self.lookback}")
                return None
                
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            features_df = self.prepare_features(df)
            if len(features_df) < 1:
                return None
                
            numeric_features = features_df.select_dtypes(include=['float64', 'int64'])
            features = self.scaler.transform(numeric_features)
            
            predictions = self.predict(features[-1:])
            trend_prob = F.softmax(torch.FloatTensor(predictions), dim=1).numpy()[0]
            
            # 获取最可能的趋势和其概率
            predicted_trend = np.argmax(trend_prob)
            confidence = trend_prob[predicted_trend]
            
            self.logger.info(f"预测趋势: {['下跌', '盘整', '上涨'][predicted_trend]}, 概率分布: 下跌[{trend_prob[0]:.2%}] 盘整[{trend_prob[1]:.2%}] 上涨[{trend_prob[2]:.2%}]")
            
            # 只在高置信度时产生交易信号
            if confidence > self.confidence_threshold:
                if predicted_trend == 2:  # 上涨趋势
                    return 1
                elif predicted_trend == 0:  # 下跌趋势
                    return -1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            return None
