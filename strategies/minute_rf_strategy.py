import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class MinuteRFStrategy(BaseRFStrategy):
    """MinuteRFStrategy - 分钟级随机森林交易策略
    
    基于1分钟K线数据的随机森林策略，专注于捕捉市场微小波动。
    结合多个技术指标和波动率特征，以快速响应短期价格变化。
    
    特点:
    1. 超短线交易: 使用1分钟K线进行高频决策
    2. 波动率过滤: 通过ATR和波动率分析过滤噪音
    3. 成交量确认: 使用成交量指标确认价格趋势
    4. 动态止损: 基于ATR的自适应止损策略
    5. 快速训练: 每5分钟重新训练模型适应市场变化
    """
    
    MODEL_NAME = "MinuteRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化分钟级随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数调整
        self.n_estimators = 150        # 减少树的数量提高训练速度
        self.max_depth = 6             # 控制复杂度避免过拟合
        self.min_samples_split = 15    # 分裂所需最小样本数
        self.min_samples_leaf = 8      # 叶节点最小样本数
        self.confidence_threshold = 0.58  # 提高信号置信度要求
        self.prob_diff_threshold = 0.12   # 概率差异阈值
        
        # K线设置
        self.kline_interval = '1m'      # 1分钟K线
        self.training_lookback = 400    # 训练数据回看周期
        self.retraining_interval = 300  # 5分钟重新训练一次
        self.check_interval = 60        # 检查信号间隔(秒)
        
        # 风险控制参数
        self.max_position_hold_time = 15  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.002    # 目标利润率 0.2%
        self.stop_loss_pct = 0.0015       # 止损率 0.15%
        self.max_trades_per_hour = 6      # 每小时最大交易次数
        self.min_vol_percentile = 40      # 最小成交量百分位
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
        self.last_training_time = time.time()
        
    def prepare_features(self, klines):
        """准备特征数据"""
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
            
            # 价格特征
            features['close'] = df['close']
            features['open'] = df['open']
            features['high'] = df['high']
            features['low'] = df['low']
            
            # 基础价格比率特征
            features['close_to_open'] = df['close'] / df['open']
            features['high_to_low'] = df['high'] / df['low']
            features['body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
            # 1. 指数移动平均线 - 短期敏感
            features['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
            features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            features['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
            features['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
            features['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # EMA交叉和距离
            features['ema_3_5_cross'] = features['ema_3'] - features['ema_5']
            features['ema_5_8_cross'] = features['ema_5'] - features['ema_8']
            features['ema_8_13_cross'] = features['ema_8'] - features['ema_13']
            features['ema_13_21_cross'] = features['ema_13'] - features['ema_21']
            
            # EMA距离百分比
            features['ema_3_5_dist'] = (features['ema_3'] - features['ema_5']) / features['ema_5'] * 100
            features['ema_5_8_dist'] = (features['ema_5'] - features['ema_8']) / features['ema_8'] * 100
            features['ema_8_13_dist'] = (features['ema_8'] - features['ema_13']) / features['ema_13'] * 100
            
            # 价格与EMA的关系
            features['close_over_ema5'] = (df['close'] - features['ema_5']) / features['ema_5'] * 100
            features['close_over_ema8'] = (df['close'] - features['ema_8']) / features['ema_8'] * 100
            features['close_over_ema13'] = (df['close'] - features['ema_13']) / features['ema_13'] * 100
            
            # 2. 动量指标 - 适合分钟级别
            features['roc_1'] = ((df['close'] / df['close'].shift(1)) - 1) * 100
            features['roc_3'] = ((df['close'] / df['close'].shift(3)) - 1) * 100
            features['roc_5'] = ((df['close'] / df['close'].shift(5)) - 1) * 100
            
            # RSI指标 - 短周期
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # 直接赋值，不使用pd.Series包装
            features['rsi_7'] = talib.RSI(close, timeperiod=7)
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            
            # RSI变化率
            features['rsi_7_change'] = features['rsi_7'].diff()
            features['rsi_14_change'] = features['rsi_14'].diff()
            
            # RSI交叉信号
            features['rsi_cross'] = features['rsi_7'] - features['rsi_14']
            
            # 3. MACD - 短周期适配
            exp12 = df['close'].ewm(span=8, adjust=False).mean()
            exp26 = df['close'].ewm(span=17, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=5, adjust=False).mean()
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = macd - signal
            features['macd_hist_change'] = features['macd_hist'].diff()
            
            # 4. 布林带 - 波动率检测
            for window in [10, 20]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'bb_upper_{window}'] = bb_middle + 2 * bb_std
                features[f'bb_lower_{window}'] = bb_middle - 2 * bb_std
                features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / bb_middle
                features[f'bb_position_{window}'] = (df['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])
            
            # 5. ATR - 波动性评估
            features['atr_7'] = talib.ATR(high, low, close, timeperiod=7)
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            
            # 相对ATR
            features['atr_7_pct'] = features['atr_7'] / df['close'] * 100
            features['atr_14_pct'] = features['atr_14'] / df['close'] * 100
            features['atr_ratio'] = features['atr_7'] / features['atr_14']
            
            # 6. 成交量指标
            features['volume'] = df['volume']
            features['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            features['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            
            # 相对成交量
            features['volume_ratio_5'] = df['volume'] / features['volume_ma_5']
            features['volume_ratio_10'] = df['volume'] / features['volume_ma_10']
            
            # 7. OBV - 能量指标
            features['obv'] = talib.OBV(close, df['volume'].values)
            features['obv_ma_5'] = features['obv'].rolling(window=5).mean()
            features['obv_ratio'] = features['obv'] / features['obv_ma_5']
            
            # 8. Stochastic - 随机指标
            slowk, slowd = talib.STOCH(high, low, close, 
                                    fastk_period=5, 
                                    slowk_period=3, 
                                    slowk_matype=0, 
                                    slowd_period=3, 
                                    slowd_matype=0)
            features['slowk'] = slowk
            features['slowd'] = slowd
                
            features['stoch_cross'] = features['slowk'] - features['slowd']
            
            # 9. 蜡烛图形态识别 - 确保索引长度匹配
            # 使用numpy数组避免索引问题
            hammer_vals = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features['hammer'] = hammer_vals / 100  # 直接赋值，无需创建Series
            
            doji_vals = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features['doji'] = doji_vals / 100
            
            engulfing_vals = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features['engulfing'] = engulfing_vals / 100
            
            star_vals = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features['star'] = star_vals / 100
            
            # 10. 移动平均线交叉 - Ichimoku云指标简化版
            features['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
            features['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
            features['tenkan_kijun_cross'] = features['tenkan_sen'] - features['kijun_sen']
            
            # 11. 价格突破特征
            for window in [5, 10, 20]:
                features[f'breakout_high_{window}'] = df['high'] > df['high'].rolling(window=window).max().shift(1)
                features[f'breakout_low_{window}'] = df['low'] < df['low'].rolling(window=window).min().shift(1)
            
            # 12. 价格加速度
            features['price_accel'] = df['close'].diff().diff()
            features['price_accel_ma'] = features['price_accel'].rolling(window=3).mean()
            
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
            if not isinstance(klines, list) or len(klines) < 30:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来短期收益
            future_returns = []
            lookback_windows = [1, 2, 3, 5]  # 1-5分钟的未来价格变化
            weights = [0.4, 0.3, 0.2, 0.1]   # 权重偏向更近的未来
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权未来收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 创建特征DataFrame存储所有计算的指标
            features = pd.DataFrame(index=df.index)
            
            # 1. RSI指标 - 直接使用pandas计算
            features['rsi'] = df['close'].rolling(window=7).apply(
                lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0).sum()))))
            )
            
            # 2. MACD指标
            features['ema12'] = df['close'].ewm(span=8, adjust=False).mean()
            features['ema26'] = df['close'].ewm(span=17, adjust=False).mean()
            features['macd'] = features['ema12'] - features['ema26']
            features['signal'] = features['macd'].ewm(span=5, adjust=False).mean()
            features['macd_hist'] = features['macd'] - features['signal']
            
            # 3. 布林带
            features['bb_middle'] = df['close'].rolling(window=20).mean()
            features['bb_std'] = df['close'].rolling(window=20).std()
            features['bb_upper'] = features['bb_middle'] + 2 * features['bb_std']
            features['bb_lower'] = features['bb_middle'] - 2 * features['bb_std']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # 4. 成交量指标
            features['volume_ma'] = df['volume'].rolling(window=10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
            
            # 5. ATR波动率指标 - 手动计算ATR
            features['tr'] = pd.DataFrame({
                'hl': df['high'] - df['low'],
                'hc': (df['high'] - df['close'].shift(1)).abs(),
                'lc': (df['low'] - df['close'].shift(1)).abs()
            }).max(axis=1)
            features['atr'] = features['tr'].rolling(window=14).mean()
            features['atr_pct'] = features['atr'] / df['close']
            
            # 市场条件得分 (用于调整阈值)
            features['trend_strength'] = (
                0.4 * features['macd_hist'] / df['close'] +           # MACD柱状图强度
                0.3 * (features['rsi'] - 50) / 50 +                   # RSI偏离中位值
                0.2 * (2 * features['bb_position'] - 1) +             # 布林带位置
                0.1 * (features['volume_ratio'] - 1)                  # 成交量异常
            )
            
            # 动态阈值 - 基于ATR的自适应阈值
            atr_scale_factor = 2.0  # ATR的缩放因子
            base_threshold = 0.0015  # 基础阈值 (0.15%)
            
            # 根据ATR调整阈值
            features['dynamic_threshold'] = base_threshold + atr_scale_factor * features['atr_pct']
            
            # 考虑趋势强度，调整买入卖出阈值
            features['trend_adjustment'] = features['trend_strength'].rolling(window=3).mean()
            
            features['buy_threshold'] = features['dynamic_threshold'] * (1 + 0.5 * features['trend_adjustment'])
            features['sell_threshold'] = -features['dynamic_threshold'] * (1 - 0.5 * features['trend_adjustment'])
            
            # 调整加权收益，考虑技术指标
            features['adjusted_returns'] = weighted_returns + 0.3 * features['trend_strength']
            
            # 标签初始化为0(观望)
            labels = pd.Series(0, index=df.index)
            
            # 移除含NaN的行
            features = features.dropna()
            
            # 确保所有特征和标签索引对齐
            common_index = weighted_returns.dropna().index.intersection(features.index)
            
            # 选择有效数据
            valid_features = features.loc[common_index]
            valid_returns = weighted_returns.loc[common_index]
            
            # 生成买入信号
            buy_condition = (
                (valid_features['adjusted_returns'] > valid_features['buy_threshold']) &  # 预期收益高于阈值
                (valid_features['rsi'] < 70) &                                            # RSI不超买
                (valid_features['bb_position'] < 0.85) &                                  # 价格不在布林带顶部
                (valid_features['volume_ratio'] > 0.8)                                    # 成交量合理
            )
            
            # 生成卖出信号
            sell_condition = (
                (valid_features['adjusted_returns'] < valid_features['sell_threshold']) &  # 预期收益低于阈值
                (valid_features['rsi'] > 30) &                                             # RSI不超卖
                (valid_features['bb_position'] > 0.15) &                                   # 价格不在布林带底部
                (valid_features['volume_ratio'] > 0.8)                                     # 成交量合理
            )
            
            # 设置标签
            valid_labels = pd.Series(0, index=common_index)
            valid_labels[buy_condition] = 1    # 买入
            valid_labels[sell_condition] = -1  # 卖出
            
            return valid_labels
            
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
                
                # 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # 检查止盈
                if profit_rate >= self.profit_target_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 获取最新K线，检查是否有反转信号
                klines = self.trader.get_klines(interval=self.kline_interval, limit=50)  # 只需要少量K线来检查反转
                
                # 计算短期技术指标
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 简单计算短期RSI
                rsi = talib.RSI(df['close'].values, timeperiod=7)
                
                # 多仓反转条件
                if position_side == "多" and rsi[-1] > 75:
                    self.logger.info(f"多仓RSI超买({rsi[-1]:.2f})，平仓")
                    self.trader.close_position()
                
                # 空仓反转条件
                if position_side == "空" and rsi[-1] < 25:
                    self.logger.info(f"空仓RSI超卖({rsi[-1]:.2f})，平仓")
                    self.trader.close_position()
                
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
