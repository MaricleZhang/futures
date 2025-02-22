import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from strategies.base_strategy import BaseStrategy

class RandomForestStrategy(BaseStrategy):
    """RandomForestMaster - 随机森林交易策略
    
    一个基于随机森林的期货交易策略模型，使用集成学习方法，
    结合技术指标、市场情绪和波动特征来预测市场走势。
    
    特点：
    1. 多特征融合：整合技术指标、市场情绪和波动性等多维度特征
    2. 三分类预测：买入、卖出、持仓三种行为的概率预测
    3. 动态阈值：使用置信度阈值过滤低置信度的交易信号
    4. 特征重要性：可解释性强，能够理解每个特征的重要程度
    """
    
    MODEL_NAME = "RandomForestMaster"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化随机森林策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 随机森林特定参数
        self.n_estimators = 200  # 增加树的数量以提高模型稳定性
        self.max_depth = 8      # 增加深度以捕捉更复杂的模式
        self.min_samples_split = 20  # 减少分裂所需样本数以提高灵敏度
        self.min_samples_leaf = 10   # 减少叶节点最小样本数
        self.confidence_threshold = 0.35  # 降低置信度阈值以增加交易频率
        self.prob_diff_threshold = 0.08   # 降低概率差阈值
        
        # K线设置
        self.kline_interval = '1m'  # 使用1分钟K线
        self.training_lookback = 500  # 减少回看周期，关注更近期的数据
        self.retraining_interval = 60  # 1分钟重新训练一次
        
        # 风险控制参数
        self.max_position_hold_time = 30  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.003    # 目标利润率
        self.stop_loss_pct = 0.002        # 止损率
        self.max_trades_per_hour = 12     # 每小时最大交易次数
        self.min_vol_percentile = 30      # 最小成交量百分位
        
        # 初始化StandardScaler
        self.scaler = StandardScaler()
        
        # 初始化模型
        self.initialize_model()
        
        # 模型评估指标
        self.feature_importance = None
        self.val_accuracies = []
        
        # 趋势状态
        self.trend_state = 0  # -1: 下跌, 0: 盘整, 1: 上涨
        self.trend_confidence = 0.0  # 趋势置信度
        
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

    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # 使用balanced而不是balanced_subsample
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.8  # 增加采样比例
        )

    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:  # 确保有足够的数据来计算指标
                self.logger.error("K线数据为空或长度不足")
                return None

            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 基础价格特征
            features['close'] = df['close']
            features['volume'] = df['volume']
            
            # 价格动量指标 - 短周期
            features['momentum_30s'] = df['close'] - df['close'].shift(1)
            features['momentum_1m'] = df['close'] - df['close'].shift(2)
            features['momentum_2m'] = df['close'] - df['close'].shift(4)
            
            # 价格变化率和加速度 - 短周期
            features['price_change_30s'] = df['close'].pct_change(1)
            features['price_change_1m'] = df['close'].pct_change(2)
            features['price_change_2m'] = df['close'].pct_change(4)
            features['price_acceleration'] = features['price_change_30s'] - features['price_change_30s'].shift(1)
            
            # 波动率指标 - 短周期
            features['volatility_1m'] = df['close'].rolling(window=2).std()
            features['volatility_2m'] = df['close'].rolling(window=4).std()
            features['volatility_ratio'] = features['volatility_1m'] / features['volatility_2m']
            
            # 成交量分析 - 短周期
            features['volume_ma_3'] = df['volume'].rolling(window=3).mean()
            features['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            features['volume_ratio_3'] = df['volume'] / features['volume_ma_3']
            features['volume_ratio_5'] = df['volume'] / features['volume_ma_5']
            features['volume_trend'] = features['volume'].pct_change()
            
            # 价格压力指标
            features['high_low_range'] = (df['high'] - df['low']) / df['close']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # 趋势强度指标 - 短周期
            features['trend_strength_1m'] = features['price_change_1m'].abs() * features['volume_ratio_3']
            features['trend_strength_2m'] = features['price_change_2m'].abs() * features['volume_ratio_5']
            
            # 布林带 - 短周期
            for window in [5, 10, 20]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'BB_width_{window}'] = (bb_std * 2) / bb_middle
                features[f'BB_position_{window}'] = (df['close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
            
            # 价格突破指标
            features['breakout_1m'] = df['close'] > df['high'].rolling(window=2).max().shift(1)
            features['breakdown_1m'] = df['close'] < df['low'].rolling(window=2).min().shift(1)
            
            # 删除任何包含NaN的行
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            features = features[feature_columns]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None

    def generate_labels(self, klines):
        """生成训练标签"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return None
                
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算多个时间窗口的未来收益
            future_returns = []
            lookback_windows = [1, 2, 3, 5]  # 减少预测窗口，专注短期
            weights = [0.4, 0.3, 0.2, 0.1]   # 更重视近期收益
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益率
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算技术指标
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # 布林带
            bb_middle = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            bb_upper = bb_middle + bb_std * 2
            bb_lower = bb_middle - bb_std * 2
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # 计算趋势强度
            trend_strength = (
                0.4 * macd_hist / df['close'] +  # MACD趋势
                0.3 * (rsi - 50) / 50 +  # RSI趋势
                0.3 * (2 * bb_position - 1)  # 布林带位置
            )
            
            # 计算波动率
            volatility = df['close'].pct_change().rolling(window=20).std()
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            atr_pct = pd.Series(atr, index=df.index) / df['close']
            
            # 动态阈值计算
            threshold_base = 0.8  # 降低基准阈值以增加信号频率
            threshold_volatility = threshold_base * (0.7 * volatility + 0.3 * atr_pct)
            
            # 趋势调整
            trend_adjustment = abs(trend_strength).rolling(window=3).mean()  # 减少趋势平滑窗口
            buy_threshold = threshold_volatility * (1 + trend_adjustment)
            sell_threshold = -threshold_volatility * (1 + trend_adjustment)
            
            # 生成标签
            labels = pd.Series(1, index=df.index)  # 默认为观望(1)
            
            # 删除所有技术指标计算产生的NaN值
            valid_mask = ~(weighted_returns.isna() | trend_strength.isna() | buy_threshold.isna() | 
                         rsi.isna() | bb_position.isna() | macd_hist.isna())
            
            # 调整后的收益率
            trend_factor = 0.4  # 趋势影响权重
            adjusted_returns = weighted_returns + trend_factor * trend_strength
            
            # 生成信号条件
            buy_condition = (
                (adjusted_returns > buy_threshold) &  # 收益率超过阈值
                (rsi < 70) &  # 非超买
                (bb_position < 0.85)  # 非布林带顶部
            )
            
            sell_condition = (
                (adjusted_returns < sell_threshold) &  # 收益率低于阈值
                (rsi > 30) &  # 非超卖
                (bb_position > 0.15)  # 非布林带底部
            )
            
            # 应用条件并生成标签
            labels[buy_condition & valid_mask] = 2  # 买入
            labels[sell_condition & valid_mask] = 0  # 卖出
            
            # 删除无效数据
            labels = labels[valid_mask]
            
            # 统计标签分布
            label_counts = labels.value_counts()
            self.logger.info(f"标签分布: 卖出={label_counts.get(0, 0)}, "
                           f"观望={label_counts.get(1, 0)}, "
                           f"买入={label_counts.get(2, 0)}")
            
            # 计算买卖信号比例
            total_signals = len(labels)
            action_signals = sum(1 for l in labels if l != 1)
            signal_ratio = (action_signals / total_signals) * 100
            self.logger.info(f"买卖信号比例: {signal_ratio:.2f}%")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            return None

    def train_model(self, klines):
        """训练随机森林模型"""
        try:
            # 准备特征数据
            features = self.prepare_features(klines)
            if features is None:
                return False
                
            # 生成标签
            labels = self.generate_labels(klines)
            if labels is None:
                return False
            
            # 确保特征和标签使用相同的索引
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # 删除任何包含NaN的行
            valid_mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            # 确保数据长度匹配
            if len(features) != len(labels):
                self.logger.error(f"特征和标签长度不匹配: 特征={len(features)}, 标签={len(labels)}")
                return False
                
            # 标准化特征
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(features)
            y = labels
            
            # 计算样本权重以处理类别不平衡
            unique_labels, label_counts = np.unique(y, return_counts=True)
            class_weights = dict(zip(unique_labels, len(y) / (len(unique_labels) * label_counts)))
            
            # 设置随机森林参数
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            # 训练模型
            self.model.fit(X, y)
            
            # 计算训练集准确率
            train_accuracy = self.model.score(X, y)
            self.logger.info(f"模型训练完成，训练集准确率: {train_accuracy:.4f}")
            
            # 获取特征重要性
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # 记录Top 5重要特征
            self.logger.info("Top 5 重要特征:")
            for feature, importance in feature_importance[:5].items():
                self.logger.info(f"{feature}: {importance:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False

    def update_trend_state(self, klines):
        """更新趋势状态"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return False
                
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # 计算多个时间周期的移动平均线
            ma_5 = df['close'].rolling(window=5).mean()
            ma_10 = df['close'].rolling(window=10).mean()
            ma_20 = df['close'].rolling(window=20).mean()
            
            # 计算EMA
            ema_5 = df['close'].ewm(span=5, adjust=False).mean()
            ema_10 = df['close'].ewm(span=10, adjust=False).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # 获取最新值
            latest_ma_trend = (
                0.5 * (ma_5.iloc[-1] - ma_10.iloc[-1]) / ma_10.iloc[-1] +
                0.3 * (ma_10.iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1] +
                0.2 * (ema_5.iloc[-1] - ema_10.iloc[-1]) / ema_10.iloc[-1]
            )
            
            latest_rsi = rsi.iloc[-1]
            latest_macd_hist = macd_hist.iloc[-1]
            prev_macd_hist = macd_hist.iloc[-2]
            
            # 计算短期价格动量
            price_momentum = df['close'].pct_change(3).iloc[-1]
            
            # 计算综合趋势得分 (-1 到 1)
            trend_score = (
                0.4 * np.sign(latest_ma_trend) * min(abs(latest_ma_trend * 10), 1) +  # 均线趋势
                0.3 * ((latest_rsi - 50) / 50) +  # RSI趋势
                0.2 * np.sign(latest_macd_hist) * min(abs(latest_macd_hist * 20), 1) +  # MACD趋势
                0.1 * np.sign(price_momentum) * min(abs(price_momentum * 10), 1)  # 短期动量
            )
            
            # 更新趋势状态和置信度
            trend_threshold = 0.15  # 降低趋势判断阈值
            
            if trend_score > trend_threshold:
                self.trend_state = 1  # 上涨趋势
                self.trend_confidence = min(abs(trend_score), 1.0)
            elif trend_score < -trend_threshold:
                self.trend_state = -1  # 下跌趋势
                self.trend_confidence = min(abs(trend_score), 1.0)
            else:
                self.trend_state = 0  # 无明显趋势
                self.trend_confidence = 0.0
                
            self.logger.info(f"趋势状态: {['下跌', '无趋势', '上涨'][self.trend_state + 1]}, "
                           f"置信度: {self.trend_confidence:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新趋势状态失败: {str(e)}")
            return False

    def generate_signal(self, klines):
        """生成交易信号"""
        try:
            # 准备特征数据
            features = self.prepare_features(klines)
            if features is None:
                return 0
                
            # 更新趋势状态
            self.update_trend_state(klines)
            
            # 使用模型预测
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[-1]
            
            # 获取每个类别的概率
            sell_prob, hold_prob, buy_prob = probabilities
            
            # 调整观望阈值 - 更激进的设置
            base_threshold = 0.35  # 大幅降低观望的基准阈值
            trend_adjustment = 0.15 * self.trend_confidence  # 增加趋势影响
            
            # 当趋势较强时降低观望阈值
            if self.trend_confidence > 0.25:  # 降低趋势判断阈值
                hold_threshold = base_threshold - trend_adjustment
            else:
                hold_threshold = base_threshold
            
            # 获取最高概率的预测
            max_prob = max(probabilities)
            prediction = np.argmax(probabilities)
            
            # 输出预测概率
            self.logger.info(f"预测概率: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
            
            # 检查成交量条件
            df = pd.DataFrame(klines[-20:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            current_volume = float(df['volume'].iloc[-1])
            volume_percentile = pd.to_numeric(df['volume']).rank(pct=True).iloc[-1] * 100
            
            # 成交量过低时不交易
            if volume_percentile < self.min_vol_percentile:
                self.logger.info(f"成交量百分位({volume_percentile:.2f}%)低于阈值({self.min_vol_percentile}%), 保持观望")
                return 1
            
            # 根据趋势调整信号
            if prediction != 1:  # 如果不是观望信号
                # 在强趋势下增强信号
                if self.trend_confidence > 0.2:
                    if self.trend_state == 1 and prediction == 2:  # 上涨趋势且预测买入
                        max_prob *= (1 + self.trend_confidence * 1.5)  # 增加趋势影响
                    elif self.trend_state == -1 and prediction == 0:  # 下跌趋势且预测卖出
                        max_prob *= (1 + self.trend_confidence * 1.5)
                
                # 检查是否满足最小置信度要求
                if max_prob < self.confidence_threshold:
                    self.logger.info(f"信号置信度({max_prob:.4f})低于阈值({self.confidence_threshold}), 保持观望")
                    return 1
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0
