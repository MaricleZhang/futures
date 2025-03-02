import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class TransformStrategy(BaseRFStrategy):
    """TransformStrategy - 中期转型交易策略
    
    基于3分钟K线数据的随机森林策略，专注于捕捉市场结构转变点。
    结合多个技术指标和市场结构特征，识别趋势转变和市场模式切换。
    
    特点:
    1. 结构转变点识别: 使用3分钟K线识别市场结构变化
    2. 动态波动率适应: 通过多周期ATR分析适应不同市场环境
    3. 成交量确认: 使用成交量指标确认价格结构转变
    4. 多周期分析: 结合短中期多个时间框架分析
    5. 定期训练: 每15分钟重新训练模型适应市场变化
    """
    
    MODEL_NAME = "Transform"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化转型策略"""
        super().__init__(trader)
        
        # 随机森林参数调整
        self.n_estimators = 180        # 树的数量
        self.max_depth = 7             # 控制复杂度避免过拟合
        self.min_samples_split = 18    # 分裂所需最小样本数
        self.min_samples_leaf = 9      # 叶节点最小样本数
        self.confidence_threshold = 0.60  # 提高信号置信度要求
        self.prob_diff_threshold = 0.15   # 概率差异阈值
        
        # K线设置
        self.kline_interval = '3m'      # 3分钟K线
        self.training_lookback = 600    # 训练数据回看周期
        self.retraining_interval = 900  # 15分钟重新训练一次
        self.check_interval = 180       # 检查信号间隔(秒)
        
        # 风险控制参数
        self.max_position_hold_time = 30  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.004    # 目标利润率 0.4%
        self.stop_loss_pct = 0.002        # 止损率 0.2%
        self.max_trades_per_hour = 4      # 每小时最大交易次数
        self.min_vol_percentile = 40      # 最小成交量百分位
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
        self.last_training_time = time.time()
        
    def prepare_features(self, klines):
        """准备特征数据 - 优化版本减少DataFrame碎片化"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 先计算所有指标，存储在字典中，最后一次性创建DataFrame
            features_dict = {}
            
            # 价格特征
            features_dict['close'] = df['close']
            features_dict['open'] = df['open']
            features_dict['high'] = df['high']
            features_dict['low'] = df['low']
            
            # 价格结构特征
            features_dict['close_to_open'] = df['close'] / df['open']
            features_dict['high_to_low'] = df['high'] / df['low']
            features_dict['body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
            features_dict['upper_shadow'] = (df['high'] - df['close'].clip(lower=df['open'])) / (df['high'] - df['low'] + 1e-8)
            features_dict['lower_shadow'] = (df['close'].clip(upper=df['open']) - df['low']) / (df['high'] - df['low'] + 1e-8)
            
            # 1. 多周期移动平均线
            features_dict['sma_10'] = df['close'].rolling(window=10).mean()
            features_dict['sma_20'] = df['close'].rolling(window=20).mean()
            features_dict['sma_50'] = df['close'].rolling(window=50).mean()
            features_dict['sma_100'] = df['close'].rolling(window=100).mean()
            
            features_dict['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            features_dict['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            features_dict['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            features_dict['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
            
            # MA交叉特征
            features_dict['sma_10_20_cross'] = features_dict['sma_10'] - features_dict['sma_20']
            features_dict['sma_20_50_cross'] = features_dict['sma_20'] - features_dict['sma_50']
            features_dict['sma_50_100_cross'] = features_dict['sma_50'] - features_dict['sma_100']
            
            features_dict['ema_5_10_cross'] = features_dict['ema_5'] - features_dict['ema_10']
            features_dict['ema_10_21_cross'] = features_dict['ema_10'] - features_dict['ema_21']
            features_dict['ema_21_55_cross'] = features_dict['ema_21'] - features_dict['ema_55']
            
            # MA距离百分比
            features_dict['close_to_sma_20'] = (df['close'] - features_dict['sma_20']) / features_dict['sma_20'] * 100
            features_dict['close_to_sma_50'] = (df['close'] - features_dict['sma_50']) / features_dict['sma_50'] * 100
            features_dict['close_to_ema_21'] = (df['close'] - features_dict['ema_21']) / features_dict['ema_21'] * 100
            
            # 2. 多周期动量指标
            features_dict['roc_3'] = ((df['close'] / df['close'].shift(3)) - 1) * 100
            features_dict['roc_6'] = ((df['close'] / df['close'].shift(6)) - 1) * 100
            features_dict['roc_12'] = ((df['close'] / df['close'].shift(12)) - 1) * 100
            features_dict['roc_24'] = ((df['close'] / df['close'].shift(24)) - 1) * 100
            
            # 3. 多周期RSI
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            features_dict['rsi_6'] = pd.Series(talib.RSI(close, timeperiod=6), index=df.index)
            features_dict['rsi_14'] = pd.Series(talib.RSI(close, timeperiod=14), index=df.index)
            features_dict['rsi_24'] = pd.Series(talib.RSI(close, timeperiod=24), index=df.index)
            
            # RSI变化率和状态
            features_dict['rsi_6_change'] = features_dict['rsi_6'].diff()
            features_dict['rsi_14_change'] = features_dict['rsi_14'].diff()
            features_dict['rsi_24_change'] = features_dict['rsi_24'].diff()
            
            # RSI发散指标
            features_dict['rsi_6_14_div'] = features_dict['rsi_6'] - features_dict['rsi_14']
            features_dict['rsi_6_14_div_change'] = features_dict['rsi_6_14_div'].diff()
            
            # 4. 多周期MACD
            for period in [(8, 17, 9), (12, 26, 9), (16, 32, 9)]:
                fast, slow, signal = period
                exp_fast = df['close'].ewm(span=fast, adjust=False).mean()
                exp_slow = df['close'].ewm(span=slow, adjust=False).mean()
                macd = exp_fast - exp_slow
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                hist = macd - signal_line
                
                features_dict[f'macd_{fast}_{slow}'] = macd
                features_dict[f'macd_signal_{fast}_{slow}'] = signal_line
                features_dict[f'macd_hist_{fast}_{slow}'] = hist
                features_dict[f'macd_hist_change_{fast}_{slow}'] = hist.diff()
            
            # 5. 多周期布林带
            for window in [20, 50]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features_dict[f'bb_upper_{window}'] = bb_middle + 2 * bb_std
                features_dict[f'bb_lower_{window}'] = bb_middle - 2 * bb_std
                features_dict[f'bb_width_{window}'] = (features_dict[f'bb_upper_{window}'] - features_dict[f'bb_lower_{window}']) / bb_middle
                features_dict[f'bb_position_{window}'] = (df['close'] - features_dict[f'bb_lower_{window}']) / (features_dict[f'bb_upper_{window}'] - features_dict[f'bb_lower_{window}'] + 1e-8)
                features_dict[f'bb_squeeze_{window}'] = features_dict[f'bb_width_{window}'].rolling(window=20).min() / features_dict[f'bb_width_{window}']
            
            # 6. 多周期ATR - 波动性评估
            features_dict['atr_7'] = pd.Series(talib.ATR(high, low, close, timeperiod=7), index=df.index)
            features_dict['atr_14'] = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
            features_dict['atr_28'] = pd.Series(talib.ATR(high, low, close, timeperiod=28), index=df.index)
            
            # 相对ATR
            features_dict['atr_7_pct'] = features_dict['atr_7'] / df['close'] * 100
            features_dict['atr_14_pct'] = features_dict['atr_14'] / df['close'] * 100
            features_dict['atr_28_pct'] = features_dict['atr_28'] / df['close'] * 100
            
            # ATR比率 - 波动率变化
            features_dict['atr_7_14_ratio'] = features_dict['atr_7'] / features_dict['atr_14']
            features_dict['atr_7_28_ratio'] = features_dict['atr_7'] / features_dict['atr_28']
            
            # 7. 成交量指标
            features_dict['volume'] = df['volume']
            features_dict['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            features_dict['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            features_dict['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            
            # 相对成交量和变化率
            features_dict['volume_ratio_5'] = df['volume'] / features_dict['volume_ma_5']
            features_dict['volume_ratio_10'] = df['volume'] / features_dict['volume_ma_10']
            features_dict['volume_ratio_20'] = df['volume'] / features_dict['volume_ma_20']
            features_dict['volume_change'] = df['volume'].pct_change() * 100
            features_dict['volume_trend'] = df['volume'].diff(5) / features_dict['volume_ma_5'] * 100
            
            # 价格和成交量关系
            features_dict['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) * df['volume']).rolling(window=5).sum()
            features_dict['pvt_ratio'] = features_dict['price_volume_trend'] / features_dict['price_volume_trend'].rolling(window=20).mean()
            
            # 8. 市场结构特征
            # 趋势和波动模式识别
            for window in [10, 20, 40]:
                # 高点低点趋势
                features_dict[f'higher_highs_{window}'] = df['high'].rolling(window=window).max() > df['high'].rolling(window=window).max().shift(window//2)
                features_dict[f'lower_lows_{window}'] = df['low'].rolling(window=window).min() < df['low'].rolling(window=window).min().shift(window//2)
                
                # 价格区间宽度变化
                features_dict[f'range_width_{window}'] = (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()) / df['close'] * 100
                features_dict[f'range_width_change_{window}'] = features_dict[f'range_width_{window}'].pct_change() * 100
                
                # 趋势强度
                close_diff = df['close'].diff(window)
                features_dict[f'trend_strength_{window}'] = close_diff.abs() / (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()) * 100
            
            # 9. 模式识别 - 关键反转形态
            # 使用talib的蜡烛图形态识别
            doji = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['doji'] = pd.Series(doji, index=df.index) / 100
            
            engulfing = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['engulfing'] = pd.Series(engulfing, index=df.index) / 100
            
            hammer = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['hammer'] = pd.Series(hammer, index=df.index) / 100
            
            shooting_star = talib.CDLSHOOTINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['shooting_star'] = pd.Series(shooting_star, index=df.index) / 100
            
            morning_star = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['morning_star'] = pd.Series(morning_star, index=df.index) / 100
            
            evening_star = talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features_dict['evening_star'] = pd.Series(evening_star, index=df.index) / 100
            
            # 10. 动量发散检测
            # 计算价格和各种指标间的背离
            rsi_14_ma = features_dict['rsi_14'].rolling(window=5).mean()
            price_ma = df['close'].rolling(window=5).mean()
            
            # 牛市背离（价格创新高但RSI未创新高）
            features_dict['bearish_divergence'] = (
                (df['close'] > df['close'].shift(3)) & 
                (features_dict['rsi_14'] < features_dict['rsi_14'].shift(3))
            ).astype(int)
            
            # 熊市背离（价格创新低但RSI未创新低）
            features_dict['bullish_divergence'] = (
                (df['close'] < df['close'].shift(3)) & 
                (features_dict['rsi_14'] > features_dict['rsi_14'].shift(3))
            ).astype(int)
            
            # 11. 超级趋势指标
            atr_period = 10
            atr_multiplier = 3.0
            
            hl2 = (df['high'] + df['low']) / 2
            atr = pd.Series(talib.ATR(high, low, close, timeperiod=atr_period), index=df.index)
            
            upper_band = hl2 + (atr_multiplier * atr)
            lower_band = hl2 - (atr_multiplier * atr)
            
            # 初始化超级趋势
            supertrend = pd.Series(0.0, index=df.index)
            trend = pd.Series(1, index=df.index)  # 1:上升趋势, -1:下降趋势
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper_band.iloc[i-1]:
                    trend.iloc[i] = 1
                elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = trend.iloc[i-1]
                    
                    if trend.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                        lower_band.iloc[i] = lower_band.iloc[i-1]
                    if trend.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                        upper_band.iloc[i] = upper_band.iloc[i-1]
                    
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
            
            features_dict['supertrend'] = supertrend
            features_dict['supertrend_direction'] = trend
            features_dict['supertrend_distance'] = (df['close'] - supertrend) / df['close'] * 100
            
            # 12. Ichimoku云指标
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52
            
            tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + df['low'].rolling(window=tenkan_period).min()) / 2
            kijun_sen = (df['high'].rolling(window=kijun_period).max() + df['low'].rolling(window=kijun_period).min()) / 2
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            senkou_span_b = (df['high'].rolling(window=senkou_span_b_period).max() + df['low'].rolling(window=senkou_span_b_period).min()) / 2
            
            features_dict['tenkan_sen'] = tenkan_sen
            features_dict['kijun_sen'] = kijun_sen
            features_dict['tenkan_kijun_cross'] = tenkan_sen - kijun_sen
            features_dict['price_above_cloud'] = ((df['close'] > senkou_span_a) & (df['close'] > senkou_span_b)).astype(int)
            features_dict['price_below_cloud'] = ((df['close'] < senkou_span_a) & (df['close'] < senkou_span_b)).astype(int)
            features_dict['cloud_thickness'] = abs(senkou_span_a - senkou_span_b) / df['close'] * 100
            
            # 一次性创建DataFrame，避免逐列插入导致的碎片化
            features = pd.DataFrame(features_dict)
            
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
            if not isinstance(klines, list) or len(klines) < 50:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来收益
            future_returns = []
            # 使用多个未来时间点，从近期到中期
            lookback_windows = [1, 3, 5, 10]  # 对应3分钟、9分钟、15分钟和30分钟
            weights = [0.25, 0.3, 0.3, 0.15]  # 权重优化，中间时间段权重更高
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权未来收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算技术指标
            # 1. 多周期RSI
            features = pd.DataFrame(index=df.index)
            features['rsi_6'] = pd.Series(talib.RSI(df['close'].values, timeperiod=6), index=df.index)
            features['rsi_14'] = pd.Series(talib.RSI(df['close'].values, timeperiod=14), index=df.index)
            features['rsi_24'] = pd.Series(talib.RSI(df['close'].values, timeperiod=24), index=df.index)
            
            # 2. 布林带
            for window in [20, 50]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'bb_upper_{window}'] = bb_middle + 2 * bb_std
                features[f'bb_lower_{window}'] = bb_middle - 2 * bb_std
                features[f'bb_position_{window}'] = (df['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'] + 1e-8)
                features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / bb_middle
            
            # 3. ATR波动率
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            features['atr_7'] = pd.Series(talib.ATR(high, low, close, timeperiod=7), index=df.index)
            features['atr_14'] = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
            features['atr_28'] = pd.Series(talib.ATR(high, low, close, timeperiod=28), index=df.index)
            
            # 相对ATR
            features['atr_7_pct'] = features['atr_7'] / df['close'] * 100
            features['atr_14_pct'] = features['atr_14'] / df['close'] * 100
            features['atr_28_pct'] = features['atr_28'] / df['close'] * 100
            
            # 4. 成交量和价格关系
            features['volume'] = df['volume']
            features['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma_10']
            features['volume_change'] = df['volume'].pct_change() * 100
            
            # 5. 趋势强度
            for window in [10, 20]:
                close_diff = df['close'].diff(window)
                features[f'trend_strength_{window}'] = close_diff.abs() / (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min() + 1e-8) * 100
            
            # 市场结构评分 - 综合多指标
            features['market_structure'] = (
                0.3 * (features['rsi_14'] - 50) / 25 +  # RSI偏离中线
                0.2 * (2 * features['bb_position_20'] - 1) +  # 布林带位置(-1到1)
                0.2 * (features['volume_ratio'] - 1) +  # 成交量异常
                0.15 * features['trend_strength_10'] / 50 +  # 短期趋势强度
                0.15 * features['trend_strength_20'] / 50   # 中期趋势强度
            )
            
            # 波动率评分
            features['volatility_score'] = (
                0.5 * features['atr_7_pct'] / features['atr_7_pct'].rolling(window=20).mean() +
                0.3 * features['atr_14_pct'] / features['atr_14_pct'].rolling(window=20).mean() +
                0.2 * features['bb_width_20'] / features['bb_width_20'].rolling(window=20).mean()
            ) - 1  # 标准化，使得正常波动率为0
            
            # 市场条件调整系数
            features['market_adjustment'] = 0.5 * features['market_structure'] + 0.5 * features['volatility_score']
            
            # 动态阈值 - 基于ATR的自适应阈值
            atr_scale_factor = 1.5  # ATR的缩放因子
            base_threshold = 0.002  # 基础阈值 (0.2%)
            
            # 根据ATR和市场结构调整阈值
            features['dynamic_threshold'] = base_threshold * (1 + features['atr_14_pct'] / features['atr_14_pct'].rolling(window=30).mean())
            
            # 调整买入卖出阈值
            features['buy_threshold'] = features['dynamic_threshold'] * (1 + 0.5 * features['market_adjustment'])
            features['sell_threshold'] = -features['dynamic_threshold'] * (1 - 0.5 * features['market_adjustment'])
            
            # 调整加权收益，考虑技术指标
            features['adjusted_returns'] = weighted_returns + 0.2 * features['market_structure']
            
            # 移除含NaN的行
            features = features.dropna()
            
            # 确保所有特征和标签索引对齐
            common_index = weighted_returns.dropna().index.intersection(features.index)
            
            # 选择有效数据
            valid_features = features.loc[common_index]
            valid_returns = weighted_returns.loc[common_index]
            
            # 标签初始化为0(观望)
            labels = pd.Series(0, index=common_index)
            
            # 生成买入信号
            buy_condition = (
                (valid_features['adjusted_returns'] > valid_features['buy_threshold']) &  # 预期收益高于阈值
                (valid_features['rsi_14'] < 75) &  # RSI不严重超买
                (valid_features['rsi_14'] > valid_features['rsi_14'].shift(1)) &  # RSI向上
                (valid_features['bb_position_20'] < 0.85) &  # 价格不在布林带顶部
                (valid_features['volume_ratio'] > 0.8)  # 成交量合理
            )
            
            # 生成卖出信号
            sell_condition = (
                (valid_features['adjusted_returns'] < valid_features['sell_threshold']) &  # 预期收益低于阈值
                (valid_features['rsi_14'] > 25) &  # RSI不严重超卖
                (valid_features['rsi_14'] < valid_features['rsi_14'].shift(1)) &  # RSI向下
                (valid_features['bb_position_20'] > 0.15) &  # 价格不在布林带底部
                (valid_features['volume_ratio'] > 0.8)  # 成交量合理
            )
            
            # 设置标签
            labels[buy_condition] = 1    # 买入
            labels[sell_condition] = -1  # 卖出
            
            return labels
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
                
                # 是否需要重新训练模型
                if self.should_retrain():
                    if self.train_model(klines):
                        self.logger.info("模型重新训练完成")
                        self.last_training_time = time.time()
                    else:
                        self.logger.error("模型重新训练失败")
                
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
                    self.logger.info(f"开多仓 - 数量: {trade_amount}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    
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
                    self.logger.info(f"开空仓 - 数量: {trade_amount}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
            
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
                
                # A. 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # B. 检查止盈
                if profit_rate >= self.profit_target_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # C. 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # D. 获取最新K线检查动态止盈止损条件
                klines = self.trader.get_klines(interval=self.kline_interval, limit=50)
                
                if len(klines) < 20:
                    self.logger.warning("K线数据不足，无法计算技术指标")
                    return
                
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # E. 计算关键指标
                # RSI
                close = df['close'].values
                rsi_14 = talib.RSI(close, timeperiod=14)
                
                # 波动率
                high = df['high'].values
                low = df['low'].values
                atr = talib.ATR(high, low, close, timeperiod=14)
                atr_pct = atr[-1] / close[-1] * 100
                
                # 布林带
                sma_20 = df['close'].rolling(window=20).mean()
                std_20 = df['close'].rolling(window=20).std()
                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20
                
                # F. 动态止盈止损条件
                # 多仓平仓条件
                if position_side == "多":
                    # 1. RSI超买区域且开始回落
                    if rsi_14[-1] > 70 and rsi_14[-1] < rsi_14[-2]:
                        self.logger.info(f"RSI超买并回落 (RSI={rsi_14[-1]:.2f})，平多仓")
                        self.trader.close_position()
                        return
                    
                    # 2. 价格触及布林带上轨
                    if df['close'].iloc[-1] > bb_upper.iloc[-1] and df['close'].iloc[-2] <= bb_upper.iloc[-2]:
                        self.logger.info(f"价格触及布林带上轨，平多仓")
                        self.trader.close_position()
                        return
                    
                    # 3. 当前利润大于1/2目标利润且近期价格开始回落
                    if (profit_rate > self.profit_target_pct / 2 and 
                        df['close'].iloc[-1] < df['close'].iloc[-2] and 
                        df['close'].iloc[-2] < df['close'].iloc[-3]):
                        self.logger.info(f"已达一半目标利润且价格开始回落，保护利润平多仓")
                        self.trader.close_position()
                        return
                
                # 空仓平仓条件
                else:
                    # 1. RSI超卖区域且开始回升
                    if rsi_14[-1] < 30 and rsi_14[-1] > rsi_14[-2]:
                        self.logger.info(f"RSI超卖并回升 (RSI={rsi_14[-1]:.2f})，平空仓")
                        self.trader.close_position()
                        return
                    
                    # 2. 价格触及布林带下轨
                    if df['close'].iloc[-1] < bb_lower.iloc[-1] and df['close'].iloc[-2] >= bb_lower.iloc[-2]:
                        self.logger.info(f"价格触及布林带下轨，平空仓")
                        self.trader.close_position()
                        return
                    
                    # 3. 当前利润大于1/2目标利润且近期价格开始反弹
                    if (profit_rate > self.profit_target_pct / 2 and 
                        df['close'].iloc[-1] > df['close'].iloc[-2] and 
                        df['close'].iloc[-2] > df['close'].iloc[-3]):
                        self.logger.info(f"已达一半目标利润且价格开始反弹，保护利润平空仓")
                        self.trader.close_position()
                        return
                
                # G. 市场结构变化检测
                # 指数移动平均线
                ema_10 = df['close'].ewm(span=10, adjust=False).mean()
                ema_21 = df['close'].ewm(span=21, adjust=False).mean()
                
                # 趋势转变信号
                if position_side == "多" and ema_10.iloc[-2] > ema_21.iloc[-2] and ema_10.iloc[-1] < ema_21.iloc[-1]:
                    self.logger.info("EMA交叉向下，平多仓")
                    self.trader.close_position()
                    return
                    
                if position_side == "空" and ema_10.iloc[-2] < ema_21.iloc[-2] and ema_10.iloc[-1] > ema_21.iloc[-1]:
                    self.logger.info("EMA交叉向上，平空仓")
                    self.trader.close_position()
                    return
                
                # H. 止损保护 - 如果行情快速变化，使用更紧的止损
                # 如果波动率高于平均值的1.5倍，启用更紧的止损
                avg_atr_pct = df['close'].pct_change().rolling(window=14).std().iloc[-1] * 100
                if atr_pct > avg_atr_pct * 1.5:
                    tighter_stop_loss = self.stop_loss_pct * 0.7  # 更紧的止损比例
                    if profit_rate <= -tighter_stop_loss:
                        self.logger.info(f"高波动环境下触发紧急止损，亏损率: {profit_rate:.4%}，平仓")
                        self.trader.close_position()
                        return
                
                # I. 止盈保护 - 如果已有不错利润，但市场气氛开始变化，提前止盈
                if profit_rate > self.profit_target_pct * 0.7:
                    # 计算成交量变化
                    vol_change = df['volume'].iloc[-1] / df['volume'].iloc[-5:].mean()
                    
                    # 如果成交量突然增大 (大于平均的1.5倍)，可能是市场情绪突变
                    if vol_change > 1.5:
                        self.logger.info(f"成交量突变且已有良好利润，提前止盈，利润率: {profit_rate:.4%}")
                        self.trader.close_position()
                        return
                        
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            
    def generate_signal(self, klines):
        """生成交易信号
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            if self.model is None:
                self.logger.error("模型未初始化")
                return 0
                
            # 准备特征
            features = self.prepare_features(klines)
            if features is None or len(features) == 0:
                return 0
            
            # 标准化特征
            features_scaled = self.scaler.transform(features.iloc[-1:])
            
            # 模型预测
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 获取预测结果 (0=卖出, 1=观望, 2=买入)
            if len(probabilities) == 3:
                sell_prob, hold_prob, buy_prob = probabilities
            else:
                self.logger.warning(f"预测概率维度异常: {probabilities}")
                return 0
            
            # 输出预测概率
            self.logger.info(f"预测概率: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
            
            # 获取最高概率对应的预测
            prediction = np.argmax(probabilities)
            max_prob = max(probabilities)
            
            # 检查置信度
            if prediction != 1 and max_prob < self.confidence_threshold:  # 1是观望
                self.logger.info(f"信号置信度({max_prob:.4f})低于阈值({self.confidence_threshold})")
                return 0  # 不够确定时保持观望
            
            # 概率差异检查 - 确保买入/卖出信号与观望有足够差距
            if prediction == 0:  # 卖出信号
                if sell_prob - hold_prob < self.prob_diff_threshold:
                    self.logger.info(f"卖出信号与观望概率差异({sell_prob - hold_prob:.4f})不足")
                    return 0
            elif prediction == 2:  # 买入信号
                if buy_prob - hold_prob < self.prob_diff_threshold:
                    self.logger.info(f"买入信号与观望概率差异({buy_prob - hold_prob:.4f})不足")
                    return 0
            
            # 检查交易次数限制
            current_hour = time.localtime().tm_hour
            if current_hour != self.last_trade_hour:
                self.last_trade_hour = current_hour
                self.trade_count_hour = 0
            
            if prediction != 1:  # 不是观望信号
                if self.trade_count_hour >= self.max_trades_per_hour:
                    self.logger.info(f"已达到每小时最大交易次数({self.max_trades_per_hour})")
                    return 0
                self.trade_count_hour += 1
            
            # 将预测值从[0,1,2]映射到[-1,0,1]
            signal_mapping = {0: -1, 1: 0, 2: 1}
            
            signal = signal_mapping[prediction]
            signal_name = "卖出" if signal == -1 else "买入" if signal == 1 else "观望"
            self.logger.info(f"生成交易信号: {signal_name}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0  # 发生错误时返回观望信号