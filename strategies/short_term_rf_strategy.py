import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class ShortTermRFStrategy(BaseRFStrategy):
    """ShortTermRFStrategy - 短线随机森林交易策略
    
    专门针对短线交易优化的随机森林策略模型，使用1分钟K线数据，
    结合多个短期技术指标和成交量特征来捕捉短期价格波动机会。
    
    特点：
    1. 超短期特征：使用30秒到2分钟的短期技术指标
    2. 高频交易：每分钟更新一次模型，快速响应市场变化
    3. 严格风控：包含持仓时间、交易频率、止盈止损等多重风控措施
    4. 成交量过滤：通过成交量分析过滤低流动性交易机会
    """
    
    MODEL_NAME = "ShortTermRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化短线随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数
        self.max_depth = 8          # 树的最大深度
        self.min_samples_split = 20 # 分裂所需最小样本数
        self.min_samples_leaf = 10  # 叶节点最小样本数
        self.confidence_threshold = 0.4  # 信号置信度阈值
        self.prob_diff_threshold = 0.08   # 概率差异阈值
        
        # K线设置
        self.kline_interval = '1m'  # 1分钟K线
        self.training_lookback = 500  # 训练数据回看周期
        self.retraining_interval = 60  # 1分钟重新训练
        
        # 风险控制参数
        self.max_position_hold_time = 30  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.003    # 目标利润率
        self.stop_loss_pct = 0.002        # 止损率
        self.max_trades_per_hour = 12     # 每小时最大交易次数
        self.min_vol_percentile = 30      # 最小成交量百分位
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
    
    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.8
        )
    
    def _initial_training(self):
        """初始化训练"""
        try:
            self.logger.info("获取历史K线数据进行初始训练...")
            historical_data = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根{self.kline_interval}K线数据")
                if self.train_model(historical_data):
                    self.logger.info("初始模型训练完成")
                else:
                    self.logger.error("初始模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
        except Exception as e:
            self.logger.error(f"初始化训练失败: {str(e)}")
    
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 价格动量指标 - 短周期
            features['momentum_1m'] = df['close'] - df['close'].shift(1)
            features['momentum_2m'] = df['close'] - df['close'].shift(2)
            features['momentum_3m'] = df['close'] - df['close'].shift(3)
            
            # 价格变化率和加速度
            features['price_change_1m'] = df['close'].pct_change(1)
            features['price_change_2m'] = df['close'].pct_change(2)
            features['price_change_3m'] = df['close'].pct_change(3)
            features['price_acceleration'] = features['price_change_1m'] - features['price_change_1m'].shift(1)
            
            # 波动率指标
            features['volatility_1m'] = df['close'].rolling(window=2).std()
            features['volatility_2m'] = df['close'].rolling(window=4).std()
            features['volatility_ratio'] = features['volatility_1m'] / features['volatility_2m']
            
            # 成交量分析
            features['volume_ma_3'] = df['volume'].rolling(window=3).mean()
            features['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            features['volume_ratio_3'] = df['volume'] / features['volume_ma_3']
            features['volume_ratio_5'] = df['volume'] / features['volume_ma_5']
            features['volume_trend'] = df['volume'].pct_change()
            
            # 价格压力指标
            features['high_low_range'] = (df['high'] - df['low']) / df['close']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # 趋势强度指标
            features['trend_strength_1m'] = features['price_change_1m'].abs() * features['volume_ratio_3']
            features['trend_strength_2m'] = features['price_change_2m'].abs() * features['volume_ratio_5']
            
            # 布林带 - 多周期
            for window in [5, 10, 20]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'BB_width_{window}'] = (bb_std * 2) / bb_middle
                features[f'BB_position_{window}'] = (df['close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
            
            # 价格突破指标
            features['breakout_1m'] = df['close'] > df['high'].rolling(window=2).max().shift(1)
            features['breakdown_1m'] = df['close'] < df['low'].rolling(window=2).min().shift(1)
            
            # 删除NaN值
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None
    
    def generate_labels(self, klines):
        """生成训练标签"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来收益
            future_returns = []
            lookback_windows = [1, 2, 3, 5]  # 短期预测窗口
            weights = [0.4, 0.3, 0.2, 0.1]   # 权重偏向近期
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算技术指标
            rsi = pd.Series(talib.RSI(df['close'].values, timeperiod=14), index=df.index)
            
            # MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # 布林带
            bb_middle = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            bb_position = (df['close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
            
            # 趋势强度
            trend_strength = (
                0.4 * macd_hist / df['close'] +
                0.3 * (rsi - 50) / 50 +
                0.3 * (2 * bb_position - 1)
            )
            
            # 波动率
            volatility = df['close'].pct_change().rolling(window=20).std()
            atr = pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            atr_pct = atr / df['close']
            
            # 动态阈值
            threshold_base = 0.8
            threshold_volatility = threshold_base * (0.7 * volatility + 0.3 * atr_pct)
            trend_adjustment = abs(trend_strength).rolling(window=3).mean()
            
            buy_threshold = threshold_volatility * (1 + trend_adjustment)
            sell_threshold = -threshold_volatility * (1 + trend_adjustment)
            
            # 生成标签
            labels = pd.Series(0, index=df.index)  # 默认观望
            
            valid_mask = ~(weighted_returns.isna() | trend_strength.isna() | 
                         buy_threshold.isna() | rsi.isna() | bb_position.isna())
            
            # 调整后的收益率
            adjusted_returns = weighted_returns + 0.4 * trend_strength
            
            # 生成信号
            buy_condition = (
                (adjusted_returns > buy_threshold) &
                (rsi < 70) &
                (bb_position < 0.85)
            )
            
            sell_condition = (
                (adjusted_returns < sell_threshold) &
                (rsi > 30) &
                (bb_position > 0.15)
            )
            
            labels[buy_condition & valid_mask] = 1    # 买入
            labels[sell_condition & valid_mask] = -1  # 卖出
            
            labels = labels[valid_mask]
            
            return labels
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None
    
    def train_model(self, klines):
        """训练模型"""
        try:
            features = self.prepare_features(klines)
            if features is None:
                return False
            
            labels = self.generate_labels(klines)
            if labels is None:
                return False
            
            # 确保特征和标签使用相同的索引
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            if len(features) != len(labels):
                self.logger.error("特征和标签长度不匹配")
                return False
            
            # 标准化特征
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(features)
            y = labels
            
            # 计算样本权重
            unique_labels, label_counts = np.unique(y, return_counts=True)
            class_weights = dict(zip(unique_labels, len(y) / (len(unique_labels) * label_counts)))
            
            # 训练模型
            self.model.set_params(class_weight=class_weights)
            self.model.fit(X, y)
            
            # 计算训练集上的预测结果
            y_pred = self.model.predict(X)
            train_accuracy = accuracy_score(y, y_pred)
            
            # 计算各类别的精确率、召回率和F1分数
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # 输出评估指标
            self.logger.info(f"模型评估指标:")
            self.logger.info(f"训练集准确率: {train_accuracy:.4f}")
            self.logger.info(f"加权精确率: {precision:.4f}")
            self.logger.info(f"加权召回率: {recall:.4f}")
            self.logger.info(f"加权F1分数: {f1:.4f}")
            
            # 输出混淆矩阵
            cm = confusion_matrix(y, y_pred)
            self.logger.info(f"混淆矩阵:")
            self.logger.info(f"卖出 观望 买入")
            self.logger.info(f"{cm[0]} # 实际卖出")
            self.logger.info(f"{cm[1]} # 实际观望")
            self.logger.info(f"{cm[2]} # 实际买入")
            
            # 记录特征重要性
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # 输出Top5重要特征
            self.logger.info("Top 5 重要特征:")
            for feature, importance in self.feature_importance[:5].items():
                self.logger.info(f"{feature}: {importance:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
    
    def should_retrain(self):
        """检查是否需要重新训练"""
        current_time = time.time()
        if current_time - self.last_training_time >= self.retraining_interval:
            self.last_training_time = current_time
            return True
        return False
    
    def check_trade_limits(self):
        """检查交易限制"""
        current_hour = pd.Timestamp.now().hour
        
        # 重置每小时交易计数
        if self.last_trade_hour != current_hour:
            self.trade_count_hour = 0
            self.last_trade_hour = current_hour
        
        # 检查是否超过每小时最大交易次数
        # if self.trade_count_hour >= self.max_trades_per_hour:
        #     self.logger.info(f"已达到每小时最大交易次数({self.max_trades_per_hour})")
        #     return False
        
        return True
    
    def check_position_time(self):
        """检查持仓时间"""
        if self.position_entry_time is None:
            return True
        
        elapsed_minutes = (pd.Timestamp.now() - self.position_entry_time).total_seconds() / 60
        if elapsed_minutes >= self.max_position_hold_time:
            self.logger.info(f"已达到最大持仓时间({self.max_position_hold_time}分钟)")
            return False
        
        return True
    