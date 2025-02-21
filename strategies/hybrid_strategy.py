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
        self.attention = SelfAttention(input_size)
        
        # 简化网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 3)
        )
        
        # 降低正则化强度
        self.l1_lambda = 0.001
        self.l2_lambda = 0.0005
        
    def forward(self, x):
        # 应用注意力机制
        x = self.attention(x)
        # 通过主网络
        x = self.layers(x)
        return torch.softmax(x, dim=1)
    
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
        self.confidence_threshold = 0.45
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
            if len(df) < 6:
                self.logger.error("数据长度不足以生成标签")
                return None
                
            # 计算多个时间窗口的未来收益
            future_returns_1m = df['close'].pct_change(-1)
            future_returns_3m = df['close'].pct_change(-3)
            future_returns_5m = df['close'].pct_change(-5)
            
            # 计算波动率
            volatility = df['volatility_1m'].ffill().rolling(5, min_periods=1).mean()
            avg_volatility = volatility.mean()
            
            # 动态阈值
            base_threshold = 0.0004
            vol_multiplier = 0.8
            
            # 计算动态阈值
            dynamic_threshold = base_threshold * (1 + vol_multiplier * (volatility / avg_volatility).ffill())
            
            # 初始化标签
            labels = np.zeros(len(df)-5)
            
            # 确保所有数据都有效
            future_returns_1m = future_returns_1m.ffill()
            future_returns_3m = future_returns_3m.ffill()
            future_returns_5m = future_returns_5m.ffill()
            
            # 截断数据以匹配长度
            future_returns_1m = future_returns_1m[:-5].values
            future_returns_3m = future_returns_3m[:-5].values
            future_returns_5m = future_returns_5m[:-5].values
            dynamic_threshold = dynamic_threshold[:-5].values
            
            # 生成标签
            for i in range(len(labels)):
                weighted_return = (
                    0.8 * future_returns_1m[i] +
                    0.15 * future_returns_3m[i] +
                    0.05 * future_returns_5m[i]
                )
                
                if weighted_return > dynamic_threshold[i]:
                    labels[i] = 2  # 买入
                elif weighted_return < -dynamic_threshold[i]:
                    labels[i] = 0  # 卖出
                else:
                    labels[i] = 1  # 持有
            
            if len(labels) == 0:
                self.logger.error("生成的标签为空")
                return None
                
            # 打印标签分布
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique, counts))
            self.logger.info(f"标签分布: 卖出={distribution.get(0, 0)}, "
                           f"观望={distribution.get(1, 0)}, "
                           f"买入={distribution.get(2, 0)}")
            
            # 计算买卖信号比例
            total_samples = len(labels)
            action_ratio = (distribution.get(0, 0) + distribution.get(2, 0)) / total_samples
            self.logger.info(f"买卖信号比例: {action_ratio:.2%}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
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
            X = X[:-5]  # 匹配标签长度
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
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # 设置学习率衰减
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.7,  # 降低学习率衰减速度
                patience=3,
                verbose=True
            )
            
            # 使用加权交叉熵损失
            weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # 训练深度学习模型
            self.dl_model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.dl_model(batch_X)
                    
                    # 计算加权交叉熵损失
                    loss = weighted_criterion(outputs, batch_y)
                    # 添加正则化损失
                    loss += self.dl_model.regularization_loss()
                    
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                # 更新学习率
                scheduler.step(avg_loss)
                
                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 4:  # 增加早停耐心值
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if epoch == self.epochs - 1 or epoch % 2 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # 评估深度学习模型
            self.dl_model.eval()
            with torch.no_grad():
                dl_outputs = self.dl_model(X_tensor)
                dl_predictions = torch.argmax(dl_outputs, dim=1)
                dl_accuracy = (dl_predictions == y_tensor).float().mean().item()
            
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
    
    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            # 检查是否需要重新训练
            current_time = time.time()
            if current_time - self.last_training_time > self.retraining_interval:
                self.train_model(klines)
                self.last_training_time = current_time

            # 准备特征
            features = self.prepare_features(klines)
            if features is None or features.empty:
                return 0

            # 选择最新的数据点
            latest_features = features.iloc[-1:]
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = latest_features[feature_columns].values

            # 标准化特征
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                self.train_model(klines)
                return 0

            # 随机森林预测
            rf_probabilities = self.rf_model.predict_proba(X_scaled)[0]
            
            # 深度学习预测
            X_tensor = torch.FloatTensor(X_scaled)
            self.dl_model.eval()
            with torch.no_grad():
                dl_probabilities = self.dl_model(X_tensor).numpy()[0]
            
            # 集成预测结果
            ensemble_probabilities = (
                self.rf_weight * rf_probabilities +
                self.dl_weight * dl_probabilities
            )
            
            # 记录预测概率
            self.logger.info(f"随机森林预测: 卖出={rf_probabilities[0]:.4f}, "
                           f"观望={rf_probabilities[1]:.4f}, "
                           f"买入={rf_probabilities[2]:.4f}")
            self.logger.info(f"深度学习预测: 卖出={dl_probabilities[0]:.4f}, "
                           f"观望={dl_probabilities[1]:.4f}, "
                           f"买入={dl_probabilities[2]:.4f}")
            self.logger.info(f"集成预测结果: 卖出={ensemble_probabilities[0]:.4f}, "
                           f"观望={ensemble_probabilities[1]:.4f}, "
                           f"买入={ensemble_probabilities[2]:.4f}")
            
            # 获取最高概率及其对应的类别
            max_prob = max(ensemble_probabilities)
            predicted_class = np.argmax(ensemble_probabilities)
            
            # 记录预测结果和置信度
            self.logger.info(f"最终预测: {['卖出', '观望', '买入'][predicted_class]} (置信度: {max_prob:.4f})")
            
            # 根据置信度阈值和预测类别生成信号
            if max_prob >= self.confidence_threshold:
                if predicted_class == 0:  # 卖出信号
                    return -1
                elif predicted_class == 2:  # 买入信号
                    return 1
            elif max_prob < 0.4:  # 如果最高概率太低，考虑次高概率
                sorted_probs = sorted(enumerate(ensemble_probabilities), key=lambda x: x[1], reverse=True)
                second_class, second_prob = sorted_probs[1]
                if second_prob >= self.confidence_threshold * 0.9:
                    if second_class == 0:  # 卖出信号
                        return -1
                    elif second_class == 2:  # 买入信号
                        return 1
            
            return 0  # 持仓不变
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0
