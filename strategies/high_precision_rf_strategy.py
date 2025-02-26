import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from datetime import datetime
import config
from strategies.base_rf_strategy import BaseRFStrategy

class HighPrecisionRFStrategy(BaseRFStrategy):
    """HighPrecisionRFStrategy - 高精度随机森林交易策略
    
    专注于高准确性的随机森林策略模型，使用1分钟K线数据，
    结合多重技术指标、波动率过滤和市场结构分析，
    实现更高的交易信号准确率。
    
    特点:
    1. 多周期分析: 综合1分钟、5分钟和15分钟时间周期的技术指标
    2. 高质量信号: 严格的信号生成标准，减少假突破和噪音影响
    3. 自适应参数: 根据市场波动性动态调整参数
    4. 强化学习机制: 持续跟踪预测表现，优化模型参数
    5. 严格风控: 综合多维度风险管理
    """
    
    MODEL_NAME = "HighPrecisionRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化高精度随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数 - 优化准确率的配置
        self.n_estimators = 350         # 进一步增加树的数量提高稳定性
        self.max_depth = 10             # 适当增加树的深度以捕捉更复杂的模式
        self.min_samples_split = 15     # 减少避免过拟合
        self.min_samples_leaf = 8       # 减少以提高精度但防止过拟合
        self.max_features = 'sqrt'      # 特征选择方式
        self.bootstrap = True           # 使用bootstrap样本
        self.class_weight = 'balanced'  # 平衡类权重
        self.random_state = 42          # 固定随机种子
        
        # 信号阈值设置 - 更宽松的信号生成标准
        self.confidence_threshold = 0.55   # 进一步降低置信度阈值 
        self.prob_diff_threshold = 0.10    # 降低概率差异阈值
        self.min_signal_quality = 0.60     # 降低最小信号质量阈值（进一步允许更多信号）
        
        # K线和训练设置
        self.kline_interval = '1m'        # 1分钟K线
        self.training_lookback = 1000     # 训练数据回看周期增加
        self.retraining_interval = 300    # 5分钟重新训练一次
        self.check_interval = 60          # 信号检查间隔(秒)
        
        # 更激进的风控参数 - 针对日志中看到的持续亏损情况
        self.max_position_hold_time = 90  # 延长最大持仓时间(分钟)
        self.profit_target_pct = 0.008    # 降低目标利润率，更容易达到止盈
        self.stop_loss_pct = 0.018        # 放宽止损率以防止过早止损
        self.max_trades_per_hour = 4      # 减少每小时最大交易次数
        self.min_vol_percentile = 30      # 降低最小成交量百分位要求
        self.max_spread_pct = 0.003       # 增加最大价差容忍度
        
        # 多周期分析设置
        self.multi_timeframe_weights = {
            '1m': 0.6,   # 1分钟K线权重
            '5m': 0.3,   # 5分钟K线权重
            '15m': 0.1   # 15分钟K线权重
        }
        
        # 市场状态变量
        self.market_volatility = None     # 市场波动率
        self.market_trend = None          # 市场趋势
        self.last_signal = 0              # 上一次信号
        self.signal_history = []          # 信号历史
        self.prediction_accuracy = {       # 预测准确率统计
            'total': 0,
            'correct': 0
        }
        
        # 特征存储和管理
        self.feature_names = None        # 存储特征名称
        self.selected_features = None    # 存储被选择的特征名称
        
        # 性能跟踪与资金管理
        self.trade_performance = []       # 交易表现记录
        self.last_model_update = time.time()  # 上次模型更新时间
        self.initial_balance = None       # 初始资金，用于计算盈亏
        self.max_consecutive_losses = 0   # one_off 连续亏损计数
        
        # 增强版止损机制
        self.max_drawdown_pct = 0.045    # 最大回撤百分比 (4.5%)
        self.adv_stop_loss_enabled = True # 启用高级止损
        self.adv_position_management = True # 启用高级仓位管理
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
        """初始化高精度随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数 - 优化准确率的配置
        self.n_estimators = 300         # 增加树的数量提高稳定性
        self.max_depth = 10             # 适当增加树的深度以捕捉更复杂的模式
        self.min_samples_split = 15     # 减少避免过拟合
        self.min_samples_leaf = 8       # 减少以提高精度但防止过拟合
        self.max_features = 'sqrt'      # 特征选择方式
        self.bootstrap = True           # 使用bootstrap样本
        self.class_weight = 'balanced'  # 平衡类权重
        self.random_state = 42          # 固定随机种子
        
        # 信号阈值设置 - 更合理的信号生成标准
        self.confidence_threshold = 0.60   # 置信度阈值
        self.prob_diff_threshold = 0.15    # 概率差异阈值
        self.min_signal_quality = 0.65     # 最小信号质量阈值（降低以允许更多信号）
        
        # K线和训练设置
        self.kline_interval = '1m'        # 1分钟K线
        self.training_lookback = 1000     # 训练数据回看周期增加
        self.retraining_interval = 300    # 5分钟重新训练一次
        self.check_interval = 60          # 信号检查间隔(秒)
        
        # 优化的风控参数
        self.max_position_hold_time = 90  # 延长最大持仓时间(分钟)
        self.profit_target_pct = 0.01     # 提高目标利润率
        self.stop_loss_pct = 0.015        # 放宽止损率以防止过早止损
        self.max_trades_per_hour = 4      # 减少每小时最大交易次数
        self.min_vol_percentile = 30      # 降低最小成交量百分位要求
        self.max_spread_pct = 0.003       # 增加最大价差容忍度
        
        # 多周期分析设置
        self.multi_timeframe_weights = {
            '1m': 0.6,   # 1分钟K线权重
            '5m': 0.3,   # 5分钟K线权重
            '15m': 0.1   # 15分钟K线权重
        }
        
        # 市场状态变量
        self.market_volatility = None     # 市场波动率
        self.market_trend = None          # 市场趋势
        self.last_signal = 0              # 上一次信号
        self.signal_history = []          # 信号历史
        self.prediction_accuracy = {       # 预测准确率统计
            'total': 0,
            'correct': 0
        }
        
        # 特征存储和管理
        self.selected_features = None  # 存储被选择的特征名称
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
        
    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,  # 使用所有CPU核心
        )
        self.logger.info("高精度随机森林模型初始化完成")
        
    def prepare_features(self, klines):
        """准备特征数据 - 扩展更多高质量特征"""
        try:
            if not isinstance(klines, list) or len(klines) < 30:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # === 价格结构特征 ===
            # 计算不同周期的价格变化
            for period in [1, 3, 5, 10, 15, 30]:
                features[f'price_change_{period}'] = df['close'].pct_change(periods=period)
                features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
                
            # 价格距离特征
            features['price_distance_ma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
            features['price_distance_ma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
            
            # 高低价格区间特征
            features['high_low_ratio'] = df['high'] / df['low']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # === 趋势指标 ===
            # EMA交叉和距离
            for fast, slow in [(5, 10), (10, 20), (20, 50)]:
                features[f'ema_{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
                features[f'ema_{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()
                features[f'ema_cross_{fast}_{slow}'] = np.where(
                    features[f'ema_{fast}'] > features[f'ema_{slow}'], 1,
                    np.where(features[f'ema_{fast}'] < features[f'ema_{slow}'], -1, 0)
                )
                features[f'ema_distance_{fast}_{slow}'] = (features[f'ema_{fast}'] - features[f'ema_{slow}']) / features[f'ema_{slow}']
            
            # MACD指标
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            features['macd_hist_change'] = np.array(pd.Series(macd_hist).diff().values)
            features['macd_cross'] = np.where(
                (macd > macd_signal) & (pd.Series(macd).shift(1) <= pd.Series(macd_signal).shift(1)), 1,
                np.where((macd < macd_signal) & (pd.Series(macd).shift(1) >= pd.Series(macd_signal).shift(1)), -1, 0)
            )
            
            # ADX - 趋势强度
            features['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['adx_trend'] = np.where(
                (features['plus_di'] > features['minus_di']) & (features['adx'] > 25), 1,
                np.where((features['plus_di'] < features['minus_di']) & (features['adx'] > 25), -1, 0)
            )
            
            # 抛物线转向SAR
            features['sar'] = talib.SAR(df['high'].values, df['low'].values, acceleration=0.02, maximum=0.2)
            features['sar_trend'] = np.where(df['close'] > features['sar'], 1, -1)
            
            # === 震荡指标 ===
            # RSI
            for period in [6, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
                
            # RSI变化率
            features['rsi_14_change'] = features['rsi_14'].diff()
            features['rsi_divergence'] = np.where(
                (df['close'] > df['close'].shift(5)) & (features['rsi_14'] < features['rsi_14'].shift(5)), -1,
                np.where((df['close'] < df['close'].shift(5)) & (features['rsi_14'] > features['rsi_14'].shift(5)), 1, 0)
            )
            
            # 随机指标
            slowk, slowd = talib.STOCH(
                df['high'].values, 
                df['low'].values, 
                df['close'].values,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            features['slowk'] = slowk
            features['slowd'] = slowd
            features['stoch_cross'] = np.where(
                (slowk > slowd) & (pd.Series(slowk).shift(1) <= pd.Series(slowd).shift(1)), 1,
                np.where((slowk < slowd) & (pd.Series(slowk).shift(1) >= pd.Series(slowd).shift(1)), -1, 0)
            )
            
            # === 波动率指标 ===
            # ATR和相对ATR
            features['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['atr_ratio'] = features['atr'] / df['close']
            
            # 布林带
            for period in [20, 50]:
                upper, middle, lower = talib.BBANDS(
                    df['close'].values,
                    timeperiod=period,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
                features[f'bb_upper_{period}'] = upper
                features[f'bb_middle_{period}'] = middle
                features[f'bb_lower_{period}'] = lower
                features[f'bb_width_{period}'] = (upper - lower) / middle
                features[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
                features[f'bb_breakout_up_{period}'] = np.where(df['close'] > upper, 1, 0)
                features[f'bb_breakout_down_{period}'] = np.where(df['close'] < lower, 1, 0)
            
            # === 成交量指标 ===
            # 成交量变化
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma10'] = df['volume'].rolling(window=10).mean()
            features['volume_ma30'] = df['volume'].rolling(window=30).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma10']
            
            # OBV - 净额成交量
            features['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            features['obv_change'] = pd.Series(features['obv']).diff()
            features['obv_ma10'] = pd.Series(features['obv']).rolling(10).mean()
            features['obv_cross'] = np.where(
                features['obv'] > features['obv_ma10'], 1,
                np.where(features['obv'] < features['obv_ma10'], -1, 0)
            )
            
            # 成交量和价格关系
            features['price_volume_trend'] = np.where(
                (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)), 1,
                np.where((df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume'].shift(1)), -1, 0)
            )
            
            # === 市场结构特征 ===
            # 支撑与阻力
            features['resistance'] = df['high'].rolling(window=20).max()
            features['support'] = df['low'].rolling(window=20).min()
            features['price_to_resistance'] = (features['resistance'] - df['close']) / df['close']
            features['price_to_support'] = (df['close'] - features['support']) / df['close']
            
            # 价格突破
            features['breakout_resistance'] = np.where(
                df['close'] > features['resistance'].shift(1), 1, 0
            )
            features['breakout_support'] = np.where(
                df['close'] < features['support'].shift(1), 1, 0
            )
            
            # 动量指标
            features['momentum_1'] = df['close'] - df['close'].shift(1)
            features['momentum_5'] = df['close'] - df['close'].shift(5)
            features['momentum_10'] = df['close'] - df['close'].shift(10)
            
            # 动量加速度
            features['momentum_acc_5'] = features['momentum_5'] - features['momentum_5'].shift(5)
            
            # 多周期趋势一致性
            features['trend_consistency'] = (
                np.sign(features['price_change_1']) + 
                np.sign(features['price_change_5']) + 
                np.sign(features['price_change_15'])
            ) / 3.0
            
            # === 信号融合特征 ===
            # 技术指标信号综合
            features['combined_signal'] = (
                0.3 * features['macd_cross'] + 
                0.2 * features['adx_trend'] + 
                0.2 * features['stoch_cross'] + 
                0.3 * features['rsi_divergence']
            )
            
            # 删除NaN值
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            # 更新市场状态
            if len(features) > 0:
                self.market_volatility = features['atr_ratio'].iloc[-1]
                self.market_trend = features['adx_trend'].iloc[-1]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None
    
    def generate_labels(self, klines):
        """生成训练标签 - 改进的标签生成方法，更注重准确性"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来价格变化
            future_returns = []
            lookback_windows = [3, 5, 10, 15]  # 不同时间窗口
            weights = [0.1, 0.2, 0.3, 0.4]   # 权重偏向长期结果
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算波动率指标用于动态阈值
            atr = pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            atr_pct = atr / df['close']
            price_volatility = df['close'].pct_change().rolling(window=20).std()
            
            # 计算各种技术指标
            rsi = pd.Series(talib.RSI(df['close'].values, timeperiod=14), index=df.index)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            macd_series = pd.Series(macd, index=df.index)
            macd_signal_series = pd.Series(macd_signal, index=df.index)
            macd_hist_series = pd.Series(macd_hist, index=df.index)
            
            # 趋势强度
            adx = pd.Series(talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            
            # 布林带位置
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            bb_width = (pd.Series(bb_upper, index=df.index) - pd.Series(bb_lower, index=df.index)) / pd.Series(bb_middle, index=df.index)
            bb_position = (df['close'] - pd.Series(bb_lower, index=df.index)) / (pd.Series(bb_upper, index=df.index) - pd.Series(bb_lower, index=df.index))
            
            # 成交量趋势
            volume_trend = df['volume'].pct_change().rolling(window=5).mean()
            
            # 根据市场条件动态调整阈值
            # 波动性调整
            vol_factor = (0.7 * price_volatility + 0.3 * atr_pct).fillna(0)
            # 趋势强度调整
            trend_factor = (adx / 100).fillna(0)
            
            # 基础阈值
            base_threshold = 0.001  # 0.1% 的基础阈值
            
            # 确保vol_factor和trend_factor不包含NaN或None值
            safe_vol_factor = vol_factor.fillna(0.001)
            safe_trend_factor = trend_factor.fillna(0.001)
            
            # 动态计算买入和卖出阈值
            buy_threshold = base_threshold * (1 + safe_vol_factor * 3) * (1 + safe_trend_factor)
            sell_threshold = -base_threshold * (1 + safe_vol_factor * 3) * (1 + safe_trend_factor)
            
            # 生成标签
            labels = pd.Series(0, index=df.index)  # 默认观望
            
            # 有效数据的掩码 - 确保每个值都是布尔类型
            valid_mask = (
                ~weighted_returns.isna() & 
                ~rsi.isna() & 
                ~adx.isna() & 
                ~bb_position.isna() & 
                ~volume_trend.isna() & 
                ~buy_threshold.isna() & 
                ~sell_threshold.isna()
            )
            
            # 买入条件 - 更严格的条件组合，确保使用正确的布尔操作符
            macd_buy_cond = ((macd_series > macd_signal_series) | 
                            ((macd_hist_series > 0) & (macd_hist_series > macd_hist_series.shift(1))))
            
            buy_condition = (
                (weighted_returns > buy_threshold) &  # 预期收益高于阈值
                (rsi < 70) &                          # RSI不超买
                (bb_position < 0.8) &                 # 价格不过高
                macd_buy_cond &                       # MACD信号良好
                (adx > 20)                            # 有明显趋势
            )
            
            # 卖出条件 - 更严格的条件组合，确保使用正确的布尔操作符
            macd_sell_cond = ((macd_series < macd_signal_series) | 
                             ((macd_hist_series < 0) & (macd_hist_series < macd_hist_series.shift(1))))
            
            sell_condition = (
                (weighted_returns < sell_threshold) &  # 预期收益低于阈值
                (rsi > 30) &                           # RSI不超卖
                (bb_position > 0.2) &                  # 价格不过低
                macd_sell_cond &                       # MACD信号良好
                (adx > 20)                             # 有明显趋势
            )
            
            # 应用条件生成标签
            labels[buy_condition & valid_mask] = 1    # 买入
            labels[sell_condition & valid_mask] = -1  # 卖出
            
            # 应用有效数据掩码
            labels = labels[valid_mask]
            
            return labels
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None
    
    def train_model(self, klines):
        """训练模型 - 增强训练过程，提高准确性和稳定性"""
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
            
            if len(features) < 100:  # 确保有足够的训练数据
                self.logger.error(f"训练数据不足，需要至少100个样本，当前: {len(features)}")
                return False
            
            # 检查类别分布
            label_counts = labels.value_counts()
            self.logger.info(f"标签分布: {label_counts.to_dict()}")
            
            # 如果任一类别样本太少，放弃训练
            min_samples_per_class = 15  # 降低最小样本数要求
            if any(count < min_samples_per_class for count in label_counts.values):
                self.logger.warning(f"某个类别的样本数量少于{min_samples_per_class}，暂不训练")
                return False
            
            # 特征选择优化
            if features.shape[1] > 40:  # 如果特征数量大于40
                # 先使用一个简单的随机森林来估计特征重要性
                quick_model = RandomForestClassifier(n_estimators=100, random_state=42)
                quick_model.fit(features, labels)
                feature_importances = pd.Series(quick_model.feature_importances_, index=features.columns)
                
                # 选择最重要的40个特征
                top_features = feature_importances.nlargest(40).index
                features = features[top_features]
                # 保存所选特征列表，用于后续预测
                self.selected_features = list(top_features)
                self.logger.info(f"通过特征选择，将特征数量从{features.shape[1]}降低到40个")
            else:
                # 特征已经少于40个，不需要选择
                self.selected_features = list(features.columns)
                self.logger.info(f"特征数量为{features.shape[1]}个，低于阈值40，保留所有特征")
            
            # 标准化特征
            try:
                # 使用有效的特征名称进行标准化
                feature_names = features.columns.tolist()
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(features)
                
                # 创建一个新的DataFrame保留特征名称
                X_with_names = pd.DataFrame(X, index=features.index, columns=feature_names)
                
                # 保存特征名称列表以后使用
                self.feature_names = feature_names
                
                # 在日志中记录特征名称保存情况
                self.logger.info(f"已保存{len(feature_names)}个特征名称用于预测")
                
                # 使用保留了特征名称的DataFrame进行后续操作
                X = X_with_names
                y = labels
                
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                return False
            
            # 动态调整类权重 - 更精细的权重设置
            buy_count = label_counts.get(1, 0)  # 买入类别样本数
            sell_count = label_counts.get(-1, 0)  # 卖出类别样本数
            hold_count = label_counts.get(0, 0)  # 观望类别样本数
            
            # 针对类别不平衡问题，设置不同的权重策略
            if hold_count > 5 * (buy_count + sell_count):  # 观望类样本明显过多
                # 给予交易信号更高的权重
                class_weight = {
                    -1: hold_count / sell_count * 1.2,  # 卖出信号额外提高20%权重
                    0: 1.0,
                    1: hold_count / buy_count * 1.2  # 买入信号额外提高20%权重
                }
                self.logger.info(f"观望信号比例过高，提高交易信号权重: {class_weight}")
            elif buy_count > 3 * sell_count:  # 买入信号明显多于卖出信号
                class_weight = {
                    -1: buy_count / sell_count * 1.3,  # 提高卖出信号权重
                    0: 1.0,
                    1: 1.0
                }
                self.logger.info(f"买入信号明显多于卖出信号，平衡权重: {class_weight}")
            elif sell_count > 3 * buy_count:  # 卖出信号明显多于买入信号
                class_weight = {
                    -1: 1.0,
                    0: 1.0,
                    1: sell_count / buy_count * 1.3  # 提高买入信号权重
                }
                self.logger.info(f"卖出信号明显多于买入信号，平衡权重: {class_weight}")
            else:
                class_weight = 'balanced'  # 使用scikit-learn的自动平衡
            
            # 根据当前市场状态和数据量调整随机森林参数
            total_samples = len(X)
            volatility_level = 'medium'  # 默认波动性级别
            
            if self.market_volatility is not None:
                if self.market_volatility > 0.003:
                    volatility_level = 'very_high'
                elif self.market_volatility > 0.002:
                    volatility_level = 'high'
                elif self.market_volatility < 0.001:
                    volatility_level = 'low'
            
            # 根据波动性和样本量设置模型参数
            param_configs = {
                'very_high': {
                    'n_estimators': min(500, total_samples // 2),
                    'max_depth': 12,
                    'min_samples_split': 15,
                    'min_samples_leaf': 8
                },
                'high': {
                    'n_estimators': min(400, total_samples // 3),
                    'max_depth': 10,
                    'min_samples_split': 15,
                    'min_samples_leaf': 8
                },
                'medium': {
                    'n_estimators': min(300, total_samples // 3),
                    'max_depth': 9,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10
                },
                'low': {
                    'n_estimators': min(200, total_samples // 4),
                    'max_depth': 8,
                    'min_samples_split': 25,
                    'min_samples_leaf': 12
                }
            }
            
            selected_params = param_configs[volatility_level]
            self.logger.info(f"市场波动性级别: {volatility_level}，调整模型参数: {selected_params}")
            
            # 更新模型参数
            self.model.set_params(
                n_estimators=selected_params['n_estimators'],
                max_depth=selected_params['max_depth'],
                min_samples_split=selected_params['min_samples_split'],
                min_samples_leaf=selected_params['min_samples_leaf'],
                class_weight=class_weight
            )
            
            # 训练模型
            self.model.fit(X, y)
            
            # 评估训练集性能
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # 输出评估指标
            self.logger.info(f"模型训练完成 - 评估指标:")
            self.logger.info(f"准确率: {accuracy:.4f}")
            self.logger.info(f"精确率: {precision:.4f}")
            self.logger.info(f"召回率: {recall:.4f}")
            self.logger.info(f"F1分数: {f1:.4f}")
            
            # 输出混淆矩阵
            cm = confusion_matrix(y, y_pred)
            self.logger.info(f"混淆矩阵:")
            self.logger.info(f"{cm}")
            
            # 计算特征重要性
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # 输出前10个重要特征
            top_features = self.feature_importance.head(10)
            self.logger.info("Top 10 重要特征:")
            for feature, importance in top_features.items():
                self.logger.info(f"{feature}: {importance:.4f}")
            
            # 更新上次训练时间
            self.last_training_time = time.time()
            
            # 根据模型质量动态调整信号阈值
            if accuracy > 0.9:  # 准确率非常高
                self.confidence_threshold = 0.55
                self.min_signal_quality = 0.60
                self.logger.info(f"模型准确度极高({accuracy:.4f})，大幅降低阈值: 置信度>{self.confidence_threshold}, 信号质量>{self.min_signal_quality}")
            elif accuracy > 0.8:  # 准确率高
                self.confidence_threshold = 0.58
                self.min_signal_quality = 0.65
                self.logger.info(f"模型准确度很高({accuracy:.4f})，降低阈值: 置信度>{self.confidence_threshold}, 信号质量>{self.min_signal_quality}")
            elif accuracy > 0.7:  # 准确率中高
                self.confidence_threshold = 0.6
                self.min_signal_quality = 0.7
                self.logger.info(f"模型准确度较高({accuracy:.4f})，使用标准阈值: 置信度>{self.confidence_threshold}, 信号质量>{self.min_signal_quality}")
            else:  # 准确率较低
                self.confidence_threshold = 0.65
                self.min_signal_quality = 0.75
                self.logger.info(f"模型准确度较低({accuracy:.4f})，提高阈值: 置信度>{self.confidence_threshold}, 信号质量>{self.min_signal_quality}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
            
    def evaluate_signal_quality(self, probabilities):
        """评估信号质量"""
        try:
            # 获取最大概率
            max_prob = max(probabilities)
            # 计算第二大概率
            sorted_probs = sorted(probabilities, reverse=True)
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
            
            # 计算概率差异
            prob_diff = max_prob - second_prob
            
            # 计算信号质量得分 (0-1)
            quality_score = 0.7 * max_prob + 0.3 * prob_diff
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"评估信号质量失败: {str(e)}")
            return 0.0
            
    def generate_signal(self, klines=None):
        """生成交易信号 - 优化的信号生成流程，提高准确率"""
        try:
            # 检查是否需要重新训练
            if self.should_retrain():
                self.logger.info("准备重新训练模型...")
                # 如果未提供K线数据，获取新的K线数据
                if klines is None or len(klines) < self.training_lookback:
                    training_klines = self.trader.get_klines(
                        symbol=self.trader.symbol,
                        interval=self.kline_interval,
                        limit=self.training_lookback
                    )
                else:
                    training_klines = klines[-self.training_lookback:]
                
                if self.train_model(training_klines):
                    self.logger.info("模型重新训练成功")
            
            # 确保有足够的K线数据
            if klines is None or len(klines) < 100:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=500
                )
                
            if not klines or len(klines) < 100:
                self.logger.error("获取K线数据失败或数据不足")
                return 0
            
            # 获取当前价格和持仓信息
            current_price = float(klines[-1][4])  # 当前收盘价
            position = self.trader.get_position()
            position_amount = float(position['info'].get('positionAmt', 0)) if position and 'info' in position else 0
            
            # 准备特征
            features = self.prepare_features(klines)
            if features is None or len(features) == 0:
                self.logger.error("无法生成有效特征")
                return 0
            
            # 标准化特征
            try:
                if self.feature_names is not None:
                    # 确保特征顺序一致
                    if set(self.feature_names) <= set(features.columns):
                        # 按照训练时的特征顺序排列
                        aligned_features = features[self.feature_names]
                        features_scaled = self.scaler.transform(aligned_features.values[-1:])
                    else:
                        self.logger.warning("特征集不匹配，将使用所有特征重新训练")
                        # 强制重新训练
                        if self.train_model(klines):
                            self.logger.info("因特征不匹配，模型已强制重新训练")
                            # 重新使用新的特征名称
                            aligned_features = features[self.feature_names]
                            features_scaled = self.scaler.transform(aligned_features.values[-1:])
                        else:
                            return 0
                else:
                    # 如果没有保存特征名称，使用所有特征
                    features_scaled = self.scaler.transform(features.values[-1:])
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}，尝试重新训练")
                # 尝试重新训练模型
                if self.train_model(klines):
                    self.logger.info("因特征标准化失败，模型已重新训练")
                    # 重新尝试标准化
                    try:
                        aligned_features = features[self.feature_names]
                        features_scaled = self.scaler.transform(aligned_features.values[-1:])
                    except Exception as e2:
                        self.logger.error(f"再次特征标准化失败: {str(e2)}")
                        return 0
                else:
                    return 0
            
            # 获取实时市场数据
            market_data = self.get_real_time_market_data()
            
            # 模型预测
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 评估信号质量
            signal_quality = self.evaluate_signal_quality(probabilities)
            
            # 输出预测概率和信号质量
            self.logger.info(f"预测概率: 卖出={probabilities[0]:.4f}, 观望={probabilities[1]:.4f}, 买入={probabilities[2]:.4f}")
            self.logger.info(f"信号质量: {signal_quality:.4f}")
            
            # 获取预测类别
            prediction = self.model.predict(features_scaled)[0]
            
            # 优化: 对于高概率和高质量信号，考虑当前持仓情况进行特殊处理
            is_in_position = abs(position_amount) > 0
            position_direction = 'long' if position_amount > 0 else 'short' if position_amount < 0 else None
            
            # 根据预测结果映射信号值
            signal_mapping = {0: -1, 1: 0, 2: 1}  # 映射到 -1(卖出), 0(观望), 1(买入)
            signal = signal_mapping[prediction]
            
            # 如果已有持仓，只在以下情况下考虑反向信号：
            # 1. 质量非常高的反向信号
            # 2. 当前持仓已经处于亏损状态
            if is_in_position:
                entry_price = float(position['info'].get('entryPrice', 0))
                # 计算当前盈亏
                if position_direction == 'long':
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
                
                # 如果当前持仓盈利，对反向信号要求更高
                if profit_pct > 0:
                    # 持有多仓且有卖出信号，或持有空仓且有买入信号
                    if (position_direction == 'long' and signal == -1) or (position_direction == 'short' and signal == 1):
                        # 对反向平仓信号提高要求
                        required_quality = self.min_signal_quality + 0.1  # 提高10%
                        required_confidence = self.confidence_threshold + 0.1  # 提高10%
                        
                        max_index = np.argmax(probabilities)
                        max_prob = probabilities[max_index]
                        
                        if signal_quality < required_quality or max_prob < required_confidence:
                            self.logger.info(f"持仓盈利中，反向信号质量不足够高(质量:{signal_quality:.4f}<{required_quality:.4f}或置信度:{max_prob:.4f}<{required_confidence:.4f})，保持原仓位")
                            return 0  # 返回观望信号，保持当前持仓
            
            # 标准质量检查
            if signal_quality < self.min_signal_quality:
                self.logger.info(f"信号质量({signal_quality:.4f})低于阈值({self.min_signal_quality})，返回观望信号")
                return 0
                
            if prediction != 0:  # 不是观望信号
                max_index = np.argmax(probabilities)
                max_prob = probabilities[max_index]
                if max_prob < self.confidence_threshold:
                    self.logger.info(f"置信度({max_prob:.4f})低于阈值({self.confidence_threshold})，返回观望信号")
                    return 0
            
            # 使用市场过滤器进一步验证信号
            if signal != 0 and not self.validate_signal_with_market_filters(signal, klines, market_data):
                self.logger.info("信号未通过市场过滤器验证，返回观望信号")
                return 0
            
            # 对已有持仓的保护性检查 - 如果当前信号与持仓方向一致，确认是否要增加仓位
            if is_in_position and ((position_direction == 'long' and signal == 1) or (position_direction == 'short' and signal == -1)):
                # 使用更严格的过滤，避免在同一方向过度加仓
                entry_price = float(position['info'].get('entryPrice', 0))
                # 检查当前价格与开仓价格的偏差
                price_deviation = abs(current_price - entry_price) / entry_price
                
                # 如果价格偏差较大，就不增加仓位
                if price_deviation > 0.01:  # 偏差超过1%
                    self.logger.info(f"已有{position_direction}仓位，当前价格偏离开仓价过大({price_deviation:.2%})，不增加仓位")
                    return 0  # 返回观望信号
            
            # 应用交易频率限制
            if not self.check_trading_frequency(signal):
                self.logger.info("超过交易频率限制，返回观望信号")
                return 0
            
            # 记录信号历史
            self.record_signal(signal, current_price, signal_quality)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0  # 错误时返回观望信号
    
    def get_real_time_market_data(self):
        """获取实时市场数据，用于信号验证"""
        try:
            current_price = self.trader.get_market_price()
            
            # 这里可以添加其他实时市场数据的获取，如买卖盘深度、最新成交等
            
            return {
                'current_price': current_price,
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            self.logger.error(f"获取实时市场数据失败: {str(e)}")
            return None
    
    def validate_signal_with_market_filters(self, signal, klines, market_data):
        """使用市场过滤器进一步验证信号，提高准确性"""
        try:
            if market_data is None:
                return False
                
            current_price = market_data['current_price']
            
            # 创建DataFrame用于技术分析
            df = pd.DataFrame(klines[-50:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 1. 成交量分析 - 使用多种指标检查
            # 使用中位数而不是平均值，避免极端值的影响
            median_volume = df['volume'].median()
            current_volume = float(klines[-1][5])
            
            # 成交量比较
            volume_ratio = current_volume / median_volume
            
            # 优化：对于不同方向的信号使用不同的成交量要求
            # 买入信号需要较高成交量，卖出信号对成交量要求可以放宽
            if signal == 1:  # 买入信号
                min_volume_ratio = 0.5  # 要求至少达到中位数的50%
            else:  # 卖出信号
                min_volume_ratio = 0.3  # 卖出信号只需要30%的中位数成交量
                
            if volume_ratio < min_volume_ratio:
                self.logger.info(f"当前成交量({current_volume:.2f})过低，仅为中位数成交量({median_volume:.2f})的{volume_ratio:.2%}，低于要求的{min_volume_ratio:.0%}，拒绝{signal==1 and '买入' or '卖出'}信号")
                return False
            
            # 2. 技术指标综合分析
            # 计算ATR - 波动性指标
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            atr_pct = atr / current_price
            
            # 计算RSI - 超买超卖指标
            rsi = talib.RSI(df['close'].values, timeperiod=14)[-1]
            
            # 计算MACD - 趋势指标
            macd, signal_line, hist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # 计算布林带 - 价格通道指标
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_width = (upper[-1] - lower[-1]) / middle[-1]  # 布林带宽度
            bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])  # 价格在带中的位置
            
            # 3. 信号验证策略
            # 买入信号验证
            if signal == 1:
                # 极端行情检查
                if rsi > 80:
                    self.logger.info(f"RSI极度超买({rsi:.2f})，拒绝买入信号")
                    return False
                
                # 趋势方向检查 - 允许小幅下跌中买入，但不要在暴跌中买入
                price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                if price_change_5m < -0.03:  # 5分钟内下跌超过3%
                    self.logger.info(f"价格处于暴跌趋势({price_change_5m:.2%})，拒绝买入信号")
                    return False
                
                # 布林带位置检查 - 避免在顶部买入
                if bb_position > 0.9 and rsi > 70:
                    self.logger.info(f"价格处于布林带顶部({bb_position:.2f})且RSI较高({rsi:.2f})，拒绝买入信号")
                    return False
                
                # MACD趋势检查 - 优先在MACD转向时买入
                if hist[-1] < 0 and not (hist[-1] > hist[-2] and hist[-2] > hist[-3]):
                    # 当MACD柱状图为负且没有连续3根上升时，考虑其他因素
                    if rsi < 40 or bb_position < 0.3:
                        # 如果RSI低或价格在布林带底部，则允许买入
                        self.logger.info(f"MACD为负但价格可能在底部区域(RSI={rsi:.2f}, BB位置={bb_position:.2f})，允许买入")
                    else:
                        # 否则拒绝买入
                        self.logger.info(f"MACD柱状图为负且没有明确上升趋势，拒绝买入")
                        return False
            
            # 卖出信号验证
            elif signal == -1:
                # 极端行情检查
                if rsi < 15:
                    self.logger.info(f"RSI极度超卖({rsi:.2f})，拒绝卖出信号")
                    return False
                
                # 趋势方向检查 - 允许小幅上涨中卖出，但不要在暴涨中卖出
                price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                if price_change_5m > 0.03:  # 5分钟内上涨超过3%
                    self.logger.info(f"价格处于暴涨趋势({price_change_5m:.2%})，拒绝卖出信号")
                    return False
                
                # 布林带位置检查 - 避免在底部卖出
                if bb_position < 0.1 and rsi < 30:
                    self.logger.info(f"价格处于布林带底部({bb_position:.2f})且RSI较低({rsi:.2f})，拒绝卖出信号")
                    return False
                
                # MACD趋势检查 - 对已有持仓放宽要求
                has_position = False
                position_direction = None
                position = self.trader.get_position()
                if position and 'info' in position:
                    position_amt = float(position['info'].get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        has_position = True
                        position_direction = 'long' if position_amt > 0 else 'short'
                
                if has_position and position_direction == 'short':
                    # 已有空仓，对卖出信号放宽MACD要求
                    self.logger.info(f"已有空仓，放宽MACD检查条件")
                    return True
                    
                if hist[-1] > 0 and not (hist[-1] < hist[-2] and hist[-2] < hist[-3]):
                    # 当MACD柱状图为正且没有连续3根下降时，考虑其他因素
                    if rsi > 60 or bb_position > 0.7:
                        # 如果RSI高或价格在布林带顶部，则允许卖出
                        self.logger.info(f"MACD为正但价格可能在顶部区域(RSI={rsi:.2f}, BB位置={bb_position:.2f})，允许卖出")
                    else:
                        # 否则拒绝卖出
                        self.logger.info(f"MACD柱状图为正且没有明确下降趋势，拒绝卖出")
                        return False
            
            # 4. 波动率过滤 - 动态调整要求
            # 对于买入和卖出信号有不同要求
            if signal == 1:  # 买入信号
                if atr_pct < 0.0004:  # ATR低于当前价格的0.04%
                    self.logger.info(f"市场波动性过低(ATR={atr_pct:.4%})，拒绝买入信号")
                    return False
            else:  # 卖出信号
                if atr_pct < 0.0003:  # 卖出对波动率要求更低
                    self.logger.info(f"市场波动性极低(ATR={atr_pct:.4%})，拒绝卖出信号")
                    return False
            
            # 5. 时间过滤器 - 避开某些时间段的交易
            timestamp = market_data['timestamp']
            trade_time = datetime.fromtimestamp(timestamp)
            hour = trade_time.hour
            minute = trade_time.minute
            
            # 避免在整点和半点前后1分钟内交易，因为这些时间点常有价格异常波动
            if (minute >= 59 or minute <= 1) or (minute >= 29 and minute <= 31):
                self.logger.info(f"避免在整点/半点附近({hour}:{minute})交易，波动可能不稳定")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证信号失败: {str(e)}")
            return False
    
    def check_trading_frequency(self, signal):
        """检查交易频率限制"""
        # 获取当前小时
        current_hour = datetime.now().hour
        
        # 如果是新的小时，重置计数
        if self.last_trade_hour != current_hour:
            self.trade_count_hour = 0
            self.last_trade_hour = current_hour
        
        # 检查是否超过每小时最大交易次数
        if signal != 0 and self.trade_count_hour >= self.max_trades_per_hour:
            return False
        
        # 如果是交易信号，增加计数
        if signal != 0:
            self.trade_count_hour += 1
            
        return True
    
    def record_signal(self, signal, price, quality):
        """记录信号历史用于性能跟踪"""
        self.signal_history.append({
            'timestamp': datetime.now().timestamp(),
            'signal': signal,
            'price': price,
            'quality': quality
        })
        
        # 仅保留最近100个信号
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
    
    def verify_past_signals(self, current_price):
        """验证过去的信号，更新预测准确度统计"""
        if not self.signal_history:
            return
            
        # 获取至少15分钟前的信号
        fifteen_mins_ago = datetime.now().timestamp() - 900
        old_signals = [s for s in self.signal_history if s['timestamp'] < fifteen_mins_ago]
        
        for old_signal in old_signals:
            signal_type = old_signal['signal']
            entry_price = old_signal['price']
            
            # 跳过观望信号
            if signal_type == 0:
                continue
                
            # 计算价格变化百分比
            price_change_pct = (current_price - entry_price) / entry_price
            
            # 判断预测是否正确
            correct_prediction = False
            if signal_type == 1 and price_change_pct > 0.002:  # 买入信号，价格上涨
                correct_prediction = True
            elif signal_type == -1 and price_change_pct < -0.002:  # 卖出信号，价格下跌
                correct_prediction = True
                
            # 更新统计
            self.prediction_accuracy['total'] += 1
            if correct_prediction:
                self.prediction_accuracy['correct'] += 1
                
            # 从历史记录中移除已验证的信号
            self.signal_history.remove(old_signal)
            
        # 计算准确率并记录
        if self.prediction_accuracy['total'] > 0:
            accuracy = self.prediction_accuracy['correct'] / self.prediction_accuracy['total']
            self.logger.info(f"信号预测准确率: {accuracy:.4f} ({self.prediction_accuracy['correct']}/{self.prediction_accuracy['total']})")
            
            # 根据准确率动态调整参数
            self.adjust_parameters_based_on_accuracy(accuracy)
    
    def adjust_parameters_based_on_accuracy(self, accuracy):
        """根据预测准确率动态调整策略参数"""
        # 至少有10次预测才进行调整
        if self.prediction_accuracy['total'] < 10:
            return
            
        if accuracy < 0.4:  # 准确率很低
            # 提高信号质量要求和置信度阈值
            self.min_signal_quality = min(0.9, self.min_signal_quality + 0.05)
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
            self.logger.info(f"预测准确率低({accuracy:.4f})，提高要求: 信号质量>{self.min_signal_quality}, 置信度>{self.confidence_threshold}")
        elif accuracy > 0.6:  # 准确率较高
            # 可以适当降低阈值
            self.min_signal_quality = max(0.65, self.min_signal_quality - 0.02)
            self.confidence_threshold = max(0.6, self.confidence_threshold - 0.02)
            self.logger.info(f"预测准确率高({accuracy:.4f})，降低要求: 信号质量>{self.min_signal_quality}, 置信度>{self.confidence_threshold}")
    
    def monitor_position(self):
        """监控当前持仓，实现智能止盈止损和动态仓位管理"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 记录初始资金（如果尚未记录）
            if self.initial_balance is None:
                balance = self.trader.get_balance()
                if balance:
                    self.initial_balance = float(balance['total'])
                    self.logger.info(f"记录初始资金: {self.initial_balance:.4f} USDT")
            
            if position is None or 'info' not in position:
                # 无持仓，继续检查是否有新信号
                klines = self.trader.get_klines(interval=self.kline_interval, limit=300)
                signal = self.generate_signal(klines)
                
                # 根据信号执行交易
                self.execute_trade_based_on_signal(signal)
                return
                
            # 有持仓，获取持仓信息
            position_amt = float(position['info'].get('positionAmt', 0))
            if abs(position_amt) <= 0:
                return
                
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_direction = 'long' if position_amt > 0 else 'short'
            
            # 计算持有时间
            entry_time = float(position['info'].get('updateTime', 0)) / 1000  # 转换为秒
            current_time = time.time()
            hold_time_minutes = (current_time - entry_time) / 60
            
            # 计算盈亏
            if position_direction == 'long':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price
                
            # 输出持仓信息
            self.logger.info(f"当前{position_direction}持仓: 数量={abs(position_amt)}, 开仓价={entry_price}, 当前价={current_price}")
            self.logger.info(f"持仓时间: {hold_time_minutes:.1f}分钟, 盈亏: {profit_pct:.2%}")
            
            # 获取账户总资金，跟踪回撤情况
            balance = self.trader.get_balance()
            total_balance = float(balance['total'])
            
            # 计算账户回撤
            if self.initial_balance and total_balance < self.initial_balance:
                drawdown_pct = (self.initial_balance - total_balance) / self.initial_balance
                if drawdown_pct > self.max_drawdown_pct and self.adv_stop_loss_enabled:
                    self.logger.warning(f"账户回撤({drawdown_pct:.2%})超过最大允许值({self.max_drawdown_pct:.2%})，进入风险保护模式")
                    # 平仓并暂停交易一段时间
                    self.trader.close_position()
                    self.logger.info(f"风险保护模式: 已平仓，暂停交易1小时")
                    time.sleep(3600)  # 暂停1小时
                    return
            
            # 止盈止损检查
            if self.should_close_position(position_direction, profit_pct, hold_time_minutes, current_price):
                # 平仓
                result = self.trader.close_position()
                self.logger.info(f"触发平仓条件，已平仓: {result}")
                
                # 更新交易表现统计
                self.trade_performance.append({
                    'direction': position_direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'hold_time': hold_time_minutes,
                    'profit_pct': profit_pct,
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 更新连续亏损计数
                if profit_pct < 0:
                    self.max_consecutive_losses += 1
                    if self.max_consecutive_losses >= 3 and self.adv_position_management:
                        self.logger.warning(f"已连续{self.max_consecutive_losses}次亏损，降低下次交易仓位")
                else:
                    self.max_consecutive_losses = 0
                
                return
                
            # 如果不需要平仓，检查是否有相反信号
            klines = self.trader.get_klines(interval=self.kline_interval, limit=300)
            signal = self.generate_signal(klines)
            
            # 验证反向信号的可靠性
            if (position_direction == 'long' and signal == -1) or (position_direction == 'short' and signal == 1):
                # 反向信号，需要验证信号质量
                recent_signals = self.signal_history[-5:] if len(self.signal_history) > 0 else []
                avg_quality = sum(s['quality'] for s in recent_signals) / len(recent_signals) if recent_signals else 0
                
                if avg_quality > 0.75:  # 高质量信号时才平仓
                    # 强烈反向信号，平仓
                    result = self.trader.close_position()
                    self.logger.info(f"收到高质量反向信号({signal})，已平仓: {result}")
                    
                    # 等待一段时间后根据信号开新仓
                    time.sleep(5)  # 等待5秒确保平仓完成
                    self.execute_trade_based_on_signal(signal)
                else:
                    self.logger.info(f"收到反向信号({signal})但质量不够高(avg_quality={avg_quality:.2f})，继续持仓")
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            if position is None or 'info' not in position:
                # 无持仓，继续检查是否有新信号
                klines = self.trader.get_klines(interval=self.kline_interval, limit=300)
                signal = self.generate_signal(klines)
                
                # 根据信号执行交易
                self.execute_trade_based_on_signal(signal)
                return
                
            # 有持仓，获取持仓信息
            position_amt = float(position['info'].get('positionAmt', 0))
            if abs(position_amt) <= 0:
                return
                
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_direction = 'long' if position_amt > 0 else 'short'
            
            # 计算持有时间
            entry_time = float(position['info'].get('updateTime', 0)) / 1000  # 转换为秒
            current_time = time.time()
            hold_time_minutes = (current_time - entry_time) / 60
            
            # 计算盈亏
            if position_direction == 'long':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price
                
            # 输出持仓信息
            self.logger.info(f"当前{position_direction}持仓: 数量={abs(position_amt)}, 开仓价={entry_price}, 当前价={current_price}")
            self.logger.info(f"持仓时间: {hold_time_minutes:.1f}分钟, 盈亏: {profit_pct:.2%}")
            
            # 止盈止损检查
            if self.should_close_position(position_direction, profit_pct, hold_time_minutes, current_price):
                # 平仓
                self.trader.close_position()
                self.logger.info(f"触发平仓条件，已平仓")
                return
                
            # 如果不需要平仓，检查是否有相反信号
            klines = self.trader.get_klines(interval=self.kline_interval, limit=300)
            signal = self.generate_signal(klines)
            
            if (position_direction == 'long' and signal == -1) or (position_direction == 'short' and signal == 1):
                # 反向信号，平仓
                self.trader.close_position()
                self.logger.info(f"收到反向信号({signal})，已平仓")
                
                # 等待一段时间后根据信号开新仓
                time.sleep(5)  # 等待5秒确保平仓完成
                self.execute_trade_based_on_signal(signal)
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
    
    def should_close_position(self, position_direction, profit_pct, hold_time_minutes, current_price):
        """判断是否应该平仓 - 优化版"""
        # 止盈检查 - 分级止盈策略
        if profit_pct >= self.profit_target_pct:
            self.logger.info(f"达到止盈条件: {profit_pct:.2%} >= {self.profit_target_pct:.2%}")
            return True
            
        # 止损检查 - 考虑持仓时间因素，时间越长，越应容忍更大的回撤
        max_loss = -self.stop_loss_pct
        if hold_time_minutes > 30:  # 持仓超过30分钟
            # 允许更大的亏损空间 
            time_factor = min(3.0, hold_time_minutes / 30)  # 最多放宽到3倍止损
            adjusted_max_loss = max_loss * time_factor
            
            if profit_pct <= adjusted_max_loss:
                self.logger.info(f"达到动态止损条件: {profit_pct:.2%} <= {adjusted_max_loss:.2%} (持仓{hold_time_minutes:.1f}分钟)")
                return True
        elif profit_pct <= max_loss:  # 持仓不足30分钟，使用标准止损
            self.logger.info(f"达到止损条件: {profit_pct:.2%} <= {max_loss:.2%}")
            return True
        
        # 特殊情况：连续亏损超过10分钟且亏损率超过1.5%
        if hold_time_minutes > 10 and profit_pct < -0.015:
            # 检查最近的价格走势
            klines = self.trader.get_klines(interval='1m', limit=5)
            if klines and len(klines) >= 5:
                movement_against_position = False
                if position_direction == 'long':
                    # 多仓且最近价格持续下跌
                    latest_closes = [float(k[4]) for k in klines[-3:]]
                    if latest_closes[0] > latest_closes[1] > latest_closes[2]:
                        movement_against_position = True
                else:  # 空仓
                    # 空仓且最近价格持续上涨
                    latest_closes = [float(k[4]) for k in klines[-3:]]
                    if latest_closes[0] < latest_closes[1] < latest_closes[2]:
                        movement_against_position = True
                
                if movement_against_position:
                    self.logger.info(f"持仓已亏损{hold_time_minutes:.1f}分钟，亏损率{profit_pct:.2%}，且价格持续不利移动，强制止损")
                    return True
            
        # 最大持仓时间检查 - 根据情况提前结束
        max_hold_time = self.max_position_hold_time
        if profit_pct < 0:  # 如果处于亏损状态，减少最大持仓时间
            max_hold_time = max(15, int(self.max_position_hold_time * 0.7))  # 至少15分钟，最多降低30%
            
        if hold_time_minutes >= max_hold_time:
            self.logger.info(f"达到{profit_pct < 0 and '调整后的' or ''}最大持仓时间: {hold_time_minutes:.1f}分钟 >= {max_hold_time}分钟")
            return True
            
        # 改进的获利保护策略 - 根据持仓时间动态调整
        profit_threshold = self.profit_target_pct * 0.5  # 只需达到目标利润的50%
        
        # 持仓时间越长，获利保护阈值越低
        if hold_time_minutes > 15:
            profit_threshold = profit_threshold * (1 - min(0.5, (hold_time_minutes - 15) / 60))
            
        if profit_pct > profit_threshold:
            # 获取更多K线以做出更准确的判断
            klines = self.trader.get_klines(interval='1m', limit=10)
            if klines and len(klines) >= 5:
                # 使用最近5根K线来判断趋势反转
                latest_prices = [float(k[4]) for k in klines[-5:]]
                price_direction = np.polyfit(range(len(latest_prices)), latest_prices, 1)[0]  # 线性回归斜率
                
                # 根据持仓方向和价格趋势决定是否平仓
                if (position_direction == 'long' and price_direction < 0) or \
                   (position_direction == 'short' and price_direction > 0):
                    self.logger.info(f"利润({profit_pct:.2%})已超过动态阈值({profit_threshold:.2%})且价格趋势反转，提前获利了结")
                    return True
        
        # 增加动态止损策略 - 如果亏损加速，提前止损
        if profit_pct < 0:
            # 查看最近的价格走势
            klines = self.trader.get_klines(interval='1m', limit=5)
            if klines and len(klines) >= 5:
                latest_prices = [float(k[4]) for k in klines[-5:]]
                # 计算最近价格变动的加速度
                if position_direction == 'long':
                    # 多仓且价格快速下跌
                    if latest_prices[-1] < latest_prices[-3] < latest_prices[-5]:
                        acceleration = (latest_prices[-5] - latest_prices[-3]) - (latest_prices[-3] - latest_prices[-1])
                        if acceleration > 0 and profit_pct < -0.005:  # 加速下跌且已亏损超过0.5%
                            self.logger.info(f"多仓亏损({profit_pct:.2%})且价格加速下跌，提前止损")
                            return True
                else:  # 空仓
                    # 空仓且价格快速上涨
                    if latest_prices[-1] > latest_prices[-3] > latest_prices[-5]:
                        acceleration = (latest_prices[-1] - latest_prices[-3]) - (latest_prices[-3] - latest_prices[-5])
                        if acceleration > 0 and profit_pct < -0.005:  # 加速上涨且已亏损超过0.5%
                            self.logger.info(f"空仓亏损({profit_pct:.2%})且价格加速上涨，提前止损")
                            return True
        
        return False
    
    def execute_trade_based_on_signal(self, signal):
        """根据信号执行交易 - 增强版，包含风险管理和动态仓位计算"""
        if signal == 0:  # 观望信号
            return
            
        try:
            # 获取账户余额
            balance = self.trader.get_balance()
            available_balance = float(balance['free'])
            total_balance = float(balance['total'])
            
            # 获取当前价格和市场数据
            current_price = self.trader.get_market_price()
            
            # 获取当前市场波动性和信号质量信息
            market_volatility = self.market_volatility or 0.002  # 默认值
            recent_signals = self.signal_history[-10:] if len(self.signal_history) > 0 else []
            avg_signal_quality = sum(s['quality'] for s in recent_signals) / len(recent_signals) if recent_signals else 0.7
            
            # 根据市场状态动态计算仓位大小
            # 1. 基础仓位百分比
            symbol_config = self.trader.symbol_config
            base_trade_percent = symbol_config.get('trade_amount_percent', 50)
            
            # 2. 根据波动性调整 - 高波动性降低仓位
            volatility_factor = 1.0
            if market_volatility > 0.003:  # 高波动
                volatility_factor = 0.7  # 降低30%
            elif market_volatility > 0.002:  # 中高波动
                volatility_factor = 0.85  # 降低15%
            elif market_volatility < 0.001:  # 低波动
                volatility_factor = 1.2  # 增加20%
            
            # 3. 根据信号质量调整 - 高质量信号增加仓位
            quality_factor = 1.0
            if avg_signal_quality > 0.85:  # 高质量信号
                quality_factor = 1.2  # 增加20%
            elif avg_signal_quality < 0.7:  # 低质量信号
                quality_factor = 0.8  # 降低20%
            
            # 4. 考虑当前账户盈亏状态 - 如果已经亏损，降低仓位
            account_factor = 1.0
            if hasattr(self, 'initial_balance') and self.initial_balance > 0:
                profit_pct = (total_balance - self.initial_balance) / self.initial_balance
                if profit_pct < -0.05:  # 亏损超过5%
                    account_factor = 0.7  # 大幅降低仓位
                elif profit_pct < -0.02:  # 亏损超过2%
                    account_factor = 0.85  # 适当降低仓位
                elif profit_pct > 0.1:  # 盈利超过10%
                    account_factor = 1.15  # 适当增加仓位
            else:
                # 记录初始资金
                self.initial_balance = total_balance
            
            # 5. 计算最终使用的仓位百分比
            adjusted_percent = base_trade_percent * volatility_factor * quality_factor * account_factor
            
            # 限制最大和最小百分比
            adjusted_percent = max(10, min(100, adjusted_percent))
            
            # 计算交易数量
            trade_amount = (available_balance * adjusted_percent / 100) / current_price
            
            # 记录仓位调整过程
            self.logger.info(f"仓位调整: 基础={base_trade_percent}%, 波动因子={volatility_factor:.2f}, " +
                           f"质量因子={quality_factor:.2f}, 账户因子={account_factor:.2f}, 最终={adjusted_percent:.2f}%")
            
            # 执行交易
            if signal == 1:  # 买入信号
                self.logger.info(f"执行买入: 价格={current_price}, 数量={trade_amount:.4f}, 金额约={trade_amount*current_price:.2f}USDT")
                result = self.trader.open_long(amount=trade_amount)
                self.logger.info(f"买入结果: {result is not None and '成功' or '失败'}")
            elif signal == -1:  # 卖出信号
                self.logger.info(f"执行卖出: 价格={current_price}, 数量={trade_amount:.4f}, 金额约={trade_amount*current_price:.2f}USDT")
                result = self.trader.open_short(amount=trade_amount)
                self.logger.info(f"卖出结果: {result is not None and '成功' or '失败'}")
                
        except Exception as e:
            self.logger.error(f"执行交易失败: {str(e)}")
            
        # 检查是否需要设置止盈止损
        try:
            # 此处可以添加设置止盈止损的代码
            # 注意：当前BinanceTrader实现中，止盈止损逻辑是在monitor_position中实现的
            pass
        except Exception as e:
            self.logger.error(f"设置止盈止损失败: {str(e)}")
            
    def run(self):
        """运行策略"""
        try:
            self.logger.info(f"启动高精度随机森林策略 v{self.VERSION}")
            
            while True:
                try:
                    self.monitor_position()
                    
                    # 获取当前价格，验证过去信号
                    current_price = self.trader.get_market_price()
                    self.verify_past_signals(current_price)
                    
                except Exception as e:
                    self.logger.error(f"策略执行出错: {str(e)}")
                    
                time.sleep(self.check_interval)
                
        except Exception as e:
            self.logger.error(f"策略运行失败: {str(e)}")
            raise