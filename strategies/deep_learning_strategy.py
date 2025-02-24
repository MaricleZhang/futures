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
import ta

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
        
        # 当前仓位数量
        self.current_quantity = None
    
    def initialize_model(self):
        """初始化深度学习模型"""
        class DeepTradingModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layer1 = nn.Linear(input_size, 256)
                self.layer2 = nn.Linear(256, 128)
                self.layer3 = nn.Linear(128, 64)
                self.layer4 = nn.Linear(64, 32)
                self.output = nn.Linear(32, 2)  # 改为2分类：0=卖出，1=买入
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
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 特征DataFrame
        """
        try:
            df = df.copy()  # 创建副本避免修改原始数据
            
            # 基础价格特征
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))
            
            # 趋势指标
            # ADX - 趋势强度指标
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14, fillna=True)
            df['plus_di'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14, fillna=True)
            df['minus_di'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14, fillna=True)
            
            # AROON - 趋势方向和强度
            df['aroon_up'] = ta.trend.aroon_up(df['high'], df['low'], window=14, fillna=True)
            df['aroon_down'] = ta.trend.aroon_down(df['high'], df['low'], window=14, fillna=True)
            df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
            
            # CCI - 顺势指标
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14, fillna=True)
            
            # 趋势斜率
            def calculate_slope(series, period=5):
                try:
                    slopes = []
                    for i in range(len(series) - period + 1):
                        y = series.iloc[i:i+period].values
                        x = np.arange(period)
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    return pd.Series(index=series.index[period-1:], data=slopes)
                except Exception:
                    return pd.Series(index=series.index, data=np.nan)
            
            df['price_slope'] = calculate_slope(df['close'], 5)
            df['volume_slope'] = calculate_slope(df['volume'], 5)
            
            # 趋势一致性
            def trend_consistency(series, period=14):
                try:
                    up_days = (series.diff() > 0).rolling(period).sum()
                    return (up_days / period * 100).round(2)
                except Exception:
                    return pd.Series(index=series.index, data=np.nan)
            
            df['price_consistency'] = trend_consistency(df['close'])
            df['volume_consistency'] = trend_consistency(df['volume'])
            
            # 动量指标
            df['rsi'] = ta.momentum.rsi(df['close'], window=14, fillna=True)
            df['macd'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            
            # 波动率指标
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14, fillna=True)
            df['bbands_width'] = ta.volatility.bollinger_pband(df['close'], window=20, window_dev=2, fillna=True)
            
            # 成交量趋势
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 价格和均线的关系
            for period in [5, 10, 20, 30]:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period, fillna=True)
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period, fillna=True)
                df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
                df[f'distance_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
            
            # 布林带指标
            df['bb_high'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2, fillna=True)
            df['bb_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2, fillna=True)
            df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'], window=20, fillna=True)
            df['bb_width'] = ((df['bb_high'] - df['bb_low']) / df['bb_mid']) * 100
            
            # Keltner通道
            df['kc_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20, fillna=True)
            df['kc_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20, fillna=True)
            
            # 删除包含NaN的行
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征准备出错: {str(e)}")
            return None
    
    def generate_labels(self, df):
        """生成训练标签
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的DataFrame
            
        Returns:
            np.array: 标签数组，0=卖出，1=买入
        """
        # 计算未来N根K线的收益率
        future_returns = df['close'].shift(-self.predict_periods).div(df['close']) - 1
        
        # 设置阈值
        threshold = 0.001 * (self.predict_periods / 5)  # 根据预测周期调整阈值
        
        # 生成标签
        labels = np.zeros(len(df))
        labels[future_returns > threshold] = 1  # 买入
        labels[future_returns < -threshold] = 0  # 卖出
        
        # 删除最后N根K线的标签（因为我们无法知道它们的未来价格）
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
    
    def calculate_position_size(self, features_df):
        """计算开仓数量"""
        try:
            # 获取趋势强度指标
            adx = features_df['adx'].iloc[-1]
            aroon_osc = features_df['aroon_osc'].iloc[-1]
            price_consistency = features_df['price_consistency'].iloc[-1]
            
            # 基础仓位比例
            base_ratio = 0.3
            
            # 根据趋势强度调整仓位
            if adx > 35 and abs(aroon_osc) > 80:  # 强趋势
                position_ratio = base_ratio * 1.2
            elif adx > 25 and abs(aroon_osc) > 60:  # 中等趋势
                position_ratio = base_ratio * 1.0
            else:  # 弱趋势
                position_ratio = base_ratio * 0.8
                
            # 根据价格一致性微调
            if price_consistency > 70:
                position_ratio *= 1.1
            elif price_consistency < 40:
                position_ratio *= 0.9
                
            # 确保不超过最大仓位
            position_ratio = min(position_ratio, 0.5)
            
            # 计算实际下单数量
            balance = float(self.trader.get_balance()['total'])
            position_value = balance * position_ratio
            current_price = features_df['close'].iloc[-1]
            quantity = position_value / current_price
            
            return round(quantity, 1)  # 四舍五入到0.1
            
        except Exception as e:
            self.logger.error(f"计算仓位大小出错: {str(e)}")
            return None

    def should_close_position(self, features_df, current_side):
        """判断是否应该平仓"""
        try:
            # 获取技术指标
            cci = features_df['cci'].iloc[-1]
            rsi = features_df['rsi'].iloc[-1]
            bb_width = features_df['bb_width'].iloc[-1]
            
            # 超买超卖信号
            if current_side == 'long' and (cci > 100 or rsi > 75):
                return True
            if current_side == 'short' and (cci < -100 or rsi < 25):
                return True
                
            # 波动率突变信号
            if bb_width > features_df['bb_width'].rolling(20).mean().iloc[-1] * 1.5:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"判断平仓信号出错: {str(e)}")
            return False
    
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
        """生成交易信号"""
        try:
            if klines is None:
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback)
            if len(klines) < self.lookback:
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
            if len(features_df) < 2:
                self.logger.error("特征数据不足")
                return None
                
            latest_features = features_df.select_dtypes(include=['float64', 'int64']).iloc[-1:]
            features = self.scaler.transform(latest_features)
            
            predictions = self.predict(features)
            prediction = np.argmax(predictions[0])
            probabilities = predictions[0]
            
            # 获取趋势强度指标
            adx = features_df['adx'].iloc[-1]
            aroon_osc = features_df['aroon_osc'].iloc[-1]
            cci = features_df['cci'].iloc[-1]
            price_consistency = features_df['price_consistency'].iloc[-1]
            rsi = features_df['rsi'].iloc[-1]
            
            # 记录预测和趋势信息
            self.logger.info(
                f"预测: {prediction} "
                f"概率分布: 空[{probabilities[0]:.2f}] 多[{probabilities[1]:.2f}] "
                f"趋势指标: ADX[{adx:.1f}] AROON_OSC[{aroon_osc:.1f}] "
                f"CCI[{cci:.1f}] RSI[{rsi:.1f}] "
                f"价格一致性[{price_consistency:.1f}] "
                f"当前价格: {df['close'].iloc[-1]:.2f}"
            )
            
            # 检查是否应该平仓
            position = self.trader.get_position()
            if position and position['size'] != 0:
                current_side = 'long' if position['size'] > 0 else 'short'
                if self.should_close_position(features_df, current_side):
                    self.logger.info(f"触发平仓信号: CCI={cci:.1f} RSI={rsi:.1f}")
                    return -1 if current_side == 'long' else 1
            
            # 根据趋势强度调整置信度要求
            base_threshold = self.confidence_threshold
            if adx > 35 and abs(aroon_osc) > 80:  # 强趋势
                adjusted_threshold = base_threshold * 0.85
            elif adx > 25 and abs(aroon_osc) > 60:  # 中等趋势
                adjusted_threshold = base_threshold * 0.95
            else:  # 弱趋势
                adjusted_threshold = base_threshold * 1.1
            
            confidence = probabilities[prediction]
            if confidence > adjusted_threshold:
                # 计算开仓数量
                quantity = self.calculate_position_size(features_df)
                if quantity:
                    self.current_quantity = quantity
                    if prediction == 1:  # 买入信号
                        return 1
                    else:  # 卖出信号
                        return -1
            
            return None  # 无交易信号
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            return None
