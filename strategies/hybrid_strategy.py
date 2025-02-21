import numpy as np
import pandas as pd
import talib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from strategies.ml_strategy import MLStrategy

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 计算注意力权重
        weights = self.attention(x)
        # 应用注意力权重
        weighted = torch.mul(x, weights)
        return weighted

class DeepNet(nn.Module):
    def __init__(self, input_size):
        super(DeepNet, self).__init__()
        
        # 1. 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 3. 预测层
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 3)
        )
        
        # 4. 残差连接
        self.shortcut = nn.Sequential(
            nn.Linear(input_size, 3),
            nn.BatchNorm1d(3)
        )
        
        # 正则化参数
        self.l1_lambda = 0.0005
        self.l2_lambda = 0.0001
        
    def forward(self, x):
        # 1. 特征提取
        features = self.feature_extractor(x)
        
        # 2. 注意力机制
        attention_weights = self.attention(features)
        attended = features * attention_weights
        
        # 3. 预测
        out = self.predictor(attended)
        
        # 4. 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        return torch.softmax(out, dim=1)
    
    def regularization_loss(self):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param.pow(2))
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

class HybridStrategy(MLStrategy):
    """
    混合策略：结合随机森林和深度学习
    
    特点：
    1. 使用随机森林处理特征选择和基本预测
    2. 使用深度学习处理时序特征和非线性关系
    3. 集成两个模型的预测结果
    4. 动态调整预测权重
    """
    
    MODEL_NAME = "HybridMaster"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化混合策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 模型参数
        self.rf_weight = 0.7  # 随机森林权重增加到0.7
        self.dl_weight = 0.3  # 深度学习权重降低到0.3
        
        # 随机森林参数
        self.n_estimators = 50
        self.max_depth = 4
        self.min_samples_split = 20
        self.min_samples_leaf = 10
        
        # 深度学习参数
        self.initial_lr = 0.001
        self.batch_size = 32
        self.epochs = 10  # 增加训练轮数到10轮
        
        # 通用参数
        self.confidence_threshold = 0.42
        self.kline_interval = '1m'
        self.training_lookback = 1000
        self.retraining_interval = 300
        self.last_training_time = 0
        
        # 初始化模型
        self.rf_model = None
        self.dl_model = None
        self.scaler = StandardScaler()
        
        # 获取历史数据进行训练
        try:
            self.logger.info("获取历史K线数据...")
            historical_data = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根{self.kline_interval}K线数据")
                if self.train_model(historical_data):
                    self.logger.info("模型训练完成")
                else:
                    self.logger.error("模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
        except Exception as e:
            self.logger.error(f"初始化训练失败: {str(e)}")
    
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype(float)
            
            # === 1. 价格特征 ===
            # 价格变化率
            df['price_change_1m'] = df['close'].pct_change()
            df['price_change_3m'] = df['close'].pct_change(3)
            df['price_change_5m'] = df['close'].pct_change(5)
            
            # 价格加速度
            df['price_acceleration'] = df['price_change_1m'].diff()
            
            # === 2. 波动性指标 ===
            # ATR
            df['ATR_short'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 波动率
            df['volatility_1m'] = df['price_change_1m'].rolling(window=5).std()
            df['volatility_3m'] = df['price_change_3m'].rolling(window=5).std()
            
            # === 3. 动量指标 ===
            # RSI
            df['RSI_short'] = talib.RSI(df['close'], timeperiod=6)
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(df['close'], fastperiod=6, slowperiod=13, signalperiod=4)
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_hist'] = hist
            
            # === 4. 趋势指标 ===
            # 布林带
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            df['BB_upper'] = upper
            df['BB_middle'] = middle
            df['BB_lower'] = lower
            
            # ADX
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # === 5. 成交量特征 ===
            # 成交量变化率
            df['volume_change_1m'] = df['volume'].pct_change()
            df['volume_change_3m'] = df['volume'].pct_change(3)
            df['volume_change_5m'] = df['volume'].pct_change(5)
            
            # 成交量趋势
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma10'] = df['volume'].rolling(window=10).mean()
            df['volume_trend'] = df['volume_ma5'] / df['volume_ma10']
            
            # === 6. 蜡烛图特征 ===
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # 删除NaN值
            df = df.ffill().fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None
    
    def generate_labels(self, df):
        """生成训练标签"""
        try:
            if len(df) < 10:  # 增加最小数据要求
                self.logger.error("数据长度不足以生成标签")
                return None
                
            # 计算多个时间窗口的未来收益
            future_returns = {
                '1m': df['close'].pct_change(-1),
                '3m': df['close'].pct_change(-3),
                '5m': df['close'].pct_change(-5),
                '10m': df['close'].pct_change(-10)
            }
            
            # 计算成交量权重
            volume_ma = df['volume'].rolling(5).mean()
            volume_weight = df['volume'] / volume_ma
            volume_weight = volume_weight.clip(0.5, 2)  # 限制权重范围
            
            # 计算波动率
            volatility = df['volatility_1m'].ffill().rolling(5, min_periods=1).mean()
            avg_volatility = volatility.mean()
            
            # 计算趋势强度
            trend_strength = abs(df['close'].pct_change(5))  # 5分钟趋势强度
            
            # 动态阈值
            base_threshold = 0.0006  # 增加基础阈值
            vol_multiplier = 1.0     # 增加波动率影响
            
            # 计算动态阈值
            dynamic_threshold = base_threshold * (1 + vol_multiplier * (volatility / avg_volatility).ffill())
            
            # 初始化标签
            labels = np.zeros(len(df)-10)  # 增加预测窗口
            
            # 确保所有数据都有效
            for key in future_returns:
                future_returns[key] = future_returns[key].ffill()
            
            # 截断数据以匹配长度
            for key in future_returns:
                future_returns[key] = future_returns[key][:-10].values
            dynamic_threshold = dynamic_threshold[:-10].values
            volume_weight = volume_weight[:-10].values
            trend_strength = trend_strength[:-10].values
            
            # 生成标签
            for i in range(len(labels)):
                # 加权计算未来收益，降低短期影响
                weighted_return = (
                    0.3 * future_returns['1m'][i] +
                    0.3 * future_returns['3m'][i] +
                    0.2 * future_returns['5m'][i] +
                    0.2 * future_returns['10m'][i]
                )
                
                # 应用成交量权重和趋势强度
                weighted_return *= volume_weight[i]
                threshold = dynamic_threshold[i] * (1 + trend_strength[i])
                
                # 生成标签
                if weighted_return > threshold:
                    labels[i] = 2  # 买入
                elif weighted_return < -threshold:
                    labels[i] = 0  # 卖出
                else:
                    labels[i] = 1  # 观望
            
            # 记录标签分布
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(['卖出', '观望', '买入'], counts))
            self.logger.info(f"标签分布: 卖出={distribution['卖出']}, "
                           f"观望={distribution['观望']}, "
                           f"买入={distribution['买入']}")
            
            # 计算买卖信号比例
            signal_ratio = (distribution['买入'] + distribution['卖出']) / len(labels) * 100
            self.logger.info(f"买卖信号比例: {signal_ratio:.2f}%")
            
            return labels.astype(int)
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None

    def initialize_models(self, input_size):
        """初始化随机森林和深度学习模型"""
        # 初始化随机森林
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.7
        )
        
        # 初始化深度学习模型
        self.dl_model = DeepNet(input_size)
        self.optimizer = optim.Adam(self.dl_model.parameters(), lr=self.initial_lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_model(self, klines):
        """训练混合模型"""
        try:
            # 准备特征和标签
            features_df = self.prepare_features(klines)
            if features_df is None or features_df.empty:
                return False

            # 生成标签
            labels = self.generate_labels(features_df)
            if labels is None:
                return False

            # 选择特征列
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 准备训练数据
            X = features_df[feature_columns].values
            X = X[:-10]  # 匹配标签长度
            y = labels

            if len(X) != len(y):
                self.logger.error(f"特征和标签长度不匹配: X={len(X)}, y={len(y)}")
                return False

            # 标准化特征
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 计算类别权重
            unique_labels, counts = np.unique(y, return_counts=True)
            total_samples = len(y)
            class_weights = torch.FloatTensor([total_samples / (len(unique_labels) * count) for count in counts])
            
            # 初始化模型
            self.initialize_models(len(feature_columns))

            # 训练随机森林
            self.rf_model.fit(X_scaled, y)
            rf_accuracy = self.rf_model.score(X_scaled, y)
            
            # 准备深度学习数据
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.LongTensor(y)
            
            # 使用数据增强
            augmented_X = []
            augmented_y = []
            
            # 1. 添加高斯噪声
            noise_scale = 0.01
            noisy_X = X_tensor + torch.randn_like(X_tensor) * noise_scale
            augmented_X.append(noisy_X)
            augmented_y.append(y_tensor)
            
            # 2. 特征缩放
            scale_factor = torch.randn(X_tensor.size(0), 1) * 0.1 + 1.0
            scaled_X = X_tensor * scale_factor
            augmented_X.append(scaled_X)
            augmented_y.append(y_tensor)
            
            # 合并原始数据和增强数据
            X_tensor = torch.cat(augmented_X, dim=0)
            y_tensor = torch.cat(augmented_y, dim=0)
            
            # 创建数据集和数据加载器
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True  # 确保batch size固定
            )
            
            # 设置学习率衰减
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.7,
                patience=3,
                verbose=True
            )
            
            # 使用加权交叉熵损失
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # 训练深度学习模型
            self.dl_model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.dl_model(batch_X)
                    
                    # 计算损失
                    loss = criterion(outputs, batch_y)
                    loss += self.dl_model.regularization_loss()
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), max_norm=1.0)
                    
                    # 优化器步进
                    self.optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)
                
                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 4:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if epoch == self.epochs - 1 or epoch % 2 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # 评估深度学习模型
            self.dl_model.eval()
            with torch.no_grad():
                dl_outputs = self.dl_model(torch.FloatTensor(X_scaled))
                dl_predictions = torch.argmax(dl_outputs, dim=1)
                dl_accuracy = (dl_predictions == torch.LongTensor(y)).float().mean().item()
            
            # 记录训练结果
            self.logger.info(f"随机森林准确率: {rf_accuracy:.4f}")
            self.logger.info(f"深度学习准确率: {dl_accuracy:.4f}")
            
            # 记录特征重要性
            feature_importance = dict(zip(feature_columns, self.rf_model.feature_importances_))
            self.logger.info("Top 5 重要特征:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                self.logger.info(f"{feature}: {importance:.4f}")

            return True

        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def predict(self, features):
        """预测并生成交易信号"""
        try:
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled)
            
            # 获取随机森林预测
            rf_probabilities = self.rf_model.predict_proba(features_scaled)
            
            # 深度学习预测
            X_tensor = torch.FloatTensor(features_scaled)
            self.dl_model.eval()
            with torch.no_grad():
                dl_probabilities = self.dl_model(X_tensor).numpy()
            
            # 确保预测结果是标准Python数值而不是numpy数组
            rf_sell = float(rf_probabilities[0][0])
            rf_hold = float(rf_probabilities[0][1])
            rf_buy = float(rf_probabilities[0][2])
            
            dl_sell = float(dl_probabilities[0][0])
            dl_hold = float(dl_probabilities[0][1])
            dl_buy = float(dl_probabilities[0][2])
            
            # 集成预测结果
            ensemble_sell = self.rf_weight * rf_sell + self.dl_weight * dl_sell
            ensemble_hold = self.rf_weight * rf_hold + self.dl_weight * dl_hold
            ensemble_buy = self.rf_weight * rf_buy + self.dl_weight * dl_buy
            
            # 记录预测概率
            self.logger.info(f"随机森林预测: 卖出={rf_sell:.4f}, "
                           f"观望={rf_hold:.4f}, "
                           f"买入={rf_buy:.4f}")
            self.logger.info(f"深度学习预测: 卖出={dl_sell:.4f}, "
                           f"观望={dl_hold:.4f}, "
                           f"买入={dl_buy:.4f}")
            self.logger.info(f"集成预测结果: 卖出={ensemble_sell:.4f}, "
                           f"观望={ensemble_hold:.4f}, "
                           f"买入={ensemble_buy:.4f}")
            
            # 生成交易信号
            ensemble_probs = [ensemble_sell, ensemble_hold, ensemble_buy]
            signal = self.generate_signal(ensemble_probs)
            
            # 记录最终预测和置信度
            if signal == 1:
                self.logger.info(f"最终预测: 买入 (置信度: {ensemble_buy:.4f})")
            elif signal == -1:
                self.logger.info(f"最终预测: 卖出 (置信度: {ensemble_sell:.4f})")
            else:
                self.logger.info(f"最终预测: 观望 (置信度: {ensemble_hold:.4f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return 0  # 发生错误时返回观望信号

    def generate_signal(self, predictions):
        """根据预测结果生成交易信号"""
        try:
            # 确保输入是Python原生float类型
            sell_prob = float(predictions[0])
            hold_prob = float(predictions[1])
            buy_prob = float(predictions[2])
            
            # 计算买卖差值和波动强度
            trade_diff = abs(buy_prob - sell_prob)
            volatility = 1 - hold_prob  # 市场波动性指标
            
            # 设置动态阈值
            base_threshold = self.confidence_threshold
            dynamic_threshold = base_threshold * (1 + volatility)  # 波动大时提高阈值
            
            # 提高信号要求
            min_prob_threshold = 0.45        # 最小概率要求
            min_trade_diff = 0.15           # 最小买卖差值
            min_strength_ratio = 1.5        # 最小强度比
            
            # 观望的情况
            if hold_prob > max(buy_prob, sell_prob) or max(buy_prob, sell_prob) < min_prob_threshold:
                return 0
            
            # 计算买卖信号强度比
            if buy_prob > sell_prob:
                strength_ratio = buy_prob / (sell_prob + hold_prob)
                if (strength_ratio > min_strength_ratio and 
                    trade_diff > min_trade_diff and 
                    buy_prob > dynamic_threshold):
                    return 1  # 买入信号
            else:
                strength_ratio = sell_prob / (buy_prob + hold_prob)
                if (strength_ratio > min_strength_ratio and 
                    trade_diff > min_trade_diff and 
                    sell_prob > dynamic_threshold):
                    return -1  # 卖出信号
                    
            return 0  # 默认观望
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {str(e)}")
            return 0  # 发生错误时返回观望信号

    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            # 准备特征数据
            features_df = self.prepare_features(klines)
            if features_df is None or features_df.empty:
                self.logger.error("特征准备失败")
                return 0
            
            # 选择特征列
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 获取最新的特征数据
            latest_features = features_df[feature_columns].iloc[-1:].values
            
            # 检查是否需要重新训练
            current_time = time.time()
            if current_time - self.last_training_time >= self.retraining_interval:
                self.logger.info("开始重新训练模型...")
                if self.train_model(klines):
                    self.last_training_time = current_time
                    self.logger.info("模型重新训练完成")
                else:
                    self.logger.error("模型重新训练失败")
            
            # 生成交易信号
            signal = self.predict(latest_features)
            
            # 获取当前价格
            current_price = float(klines[-1][4])  # 收盘价
            self.logger.info(f"当前价格: {current_price:.5f}, AI信号: {signal}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0
            
    def generate_signal(self, predictions):
        """根据预测结果生成交易信号"""
        try:
            # 确保输入是Python原生float类型
            sell_prob = float(predictions[0])
            hold_prob = float(predictions[1])
            buy_prob = float(predictions[2])
            
            # 计算买卖差值和波动强度
            trade_diff = abs(buy_prob - sell_prob)
            volatility = 1 - hold_prob  # 市场波动性指标
            
            # 设置动态阈值
            base_threshold = self.confidence_threshold
            dynamic_threshold = base_threshold * (1 + volatility)  # 波动大时提高阈值
            
            # 提高信号要求
            min_prob_threshold = 0.45        # 最小概率要求
            min_trade_diff = 0.15           # 最小买卖差值
            min_strength_ratio = 1.5        # 最小强度比
            
            # 观望的情况
            if hold_prob > max(buy_prob, sell_prob) or max(buy_prob, sell_prob) < min_prob_threshold:
                return 0
            
            # 计算买卖信号强度比
            if buy_prob > sell_prob:
                strength_ratio = buy_prob / (sell_prob + hold_prob)
                if (strength_ratio > min_strength_ratio and 
                    trade_diff > min_trade_diff and 
                    buy_prob > dynamic_threshold):
                    return 1  # 买入信号
            else:
                strength_ratio = sell_prob / (buy_prob + hold_prob)
                if (strength_ratio > min_strength_ratio and 
                    trade_diff > min_trade_diff and 
                    sell_prob > dynamic_threshold):
                    return -1  # 卖出信号
                    
            return 0  # 默认观望
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {str(e)}")
            return 0  # 发生错误时返回观望信号
