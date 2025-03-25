"""
稳定随机森林策略 - 15分钟时间框架
基于随机森林机器学习算法的稳定交易决策模型
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
from strategies.base_rf_strategy import BaseRFStrategy


class StableRFStrategy15m(BaseRFStrategy):
    """StableRFStrategy15m - 稳定随机森林策略
    
    基于15分钟K线的稳定随机森林预测策略，通过多重特征分析和稳定性机制，
    提供高置信度的交易信号，减少假突破和错误信号。
    
    特点:
    1. 多样化特征集：整合价格动量、波动率、趋势和交易量等多维度特征
    2. 信号稳定性机制：连续分析多个时间点信号，确保趋势的稳定性
    3. 概率阈值过滤：仅在预测置信度高于阈值时生成信号
    4. 自适应特征选择：基于市场条件动态调整特征重要性
    5. 历史模式识别：识别类似的历史价格模式作为决策参考
    """
    
    def __init__(self, trader):
        """初始化稳定随机森林策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '15m'       # 15分钟K线
        self.training_lookback = 1000     # 训练数据回看周期
        self.retraining_interval = 7200   # 重新训练间隔(秒)，约2小时
        self.check_interval = 300         # 检查信号间隔(秒)
        
        # 随机森林参数调整
        self.n_estimators = 200           # 树的数量
        self.max_depth = 10               # 树的最大深度
        self.min_samples_split = 20       # 分裂所需最小样本数
        self.min_samples_leaf = 10        # 叶节点最小样本数
        self.confidence_threshold = 0.3  # 信号置信度阈值
        self.prob_diff_threshold = 0.01    # 概率差异阈值
        
        # 信号稳定性参数
        self.signal_confirmation_count = 3 # 需要连续多少个周期确认信号
        self.consistent_signals = []       # 存储最近的信号
        self.last_signal_time = None       # 上次产生信号的时间
        self.min_signal_interval = 3600    # 最小信号间隔(秒)
        
        # 风控参数
        self.stop_loss_pct = 0.015        # 止损比例 (1.5%)
        self.take_profit_pct = 0.035      # 止盈比例 (3.5%)
        self.max_position_hold_time = 480 # 最大持仓时间(分钟)
        self.trailing_stop = True         # 是否启用追踪止损
        self.trailing_stop_activation = 0.02  # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.01    # 追踪止损距离百分比
        
        # 初始化模型
        self.initialize_model()
        
        # 特征名称列表 - 需要和prepare_features中的特征对应
        self.feature_names = [
            # 价格特征
            'close_change_1', 'close_change_3', 'close_change_5', 
            'close_change_10', 'close_change_20',
            
            # 移动平均线特征
            'ema_9_dist', 'ema_21_dist', 'ema_55_dist', 'ema_9_21_cross', 
            'ema_21_55_cross', 'ema_9_slope', 'ema_21_slope',
            
            # 动量指标
            'rsi_14', 'rsi_14_change', 'rsi_divergence',
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
            'cci_20', 'cci_change',
            
            # 波动性指标
            'atr_14', 'atr_change', 'atr_percent',
            'bb_width', 'bb_position', 'bb_width_change',
            
            # 交易量指标
            'volume_change', 'volume_ma_ratio', 'obv_change',
            'adosc', 'mfi',
            
            # 趋势指标
            'adx', 'di_diff', 'adx_change', 'aroon_up', 'aroon_down',
            
            # 波浪特征
            'close_to_min_5', 'close_to_max_5',
            'close_to_min_10', 'close_to_max_10',
            'close_to_min_20', 'close_to_max_20',
            
            # 价格模式特征
            'close_ma_pattern', 'candle_pattern_bullish', 'candle_pattern_bearish'
        ]
        
    def prepare_features(self, klines):
        """
        从K线数据准备特征
        
        Args:
            klines (list): K线数据 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: 特征数据框
        """
        try:
            # 转换K线为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 确保数值类型正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加日期时间
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 确保数据足够
            if len(df) < 60:
                self.logger.error("K线数据不足，无法计算特征")
                return None
                
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 1. 价格变化特征
            # 计算不同周期的价格变化百分比
            for period in [1, 3, 5, 10, 20]:
                features[f'close_change_{period}'] = df['close'].pct_change(period) * 100
                
            # 2. 移动平均线特征
            # 计算EMA
            df['ema_9'] = talib.EMA(df['close'].values, timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
            df['ema_55'] = talib.EMA(df['close'].values, timeperiod=55)
            
            # 计算价格与EMA的距离（百分比）
            features['ema_9_dist'] = (df['close'] / df['ema_9'] - 1) * 100
            features['ema_21_dist'] = (df['close'] / df['ema_21'] - 1) * 100
            features['ema_55_dist'] = (df['close'] / df['ema_55'] - 1) * 100
            
            # 计算EMA交叉
            features['ema_9_21_cross'] = df['ema_9'] - df['ema_21']
            features['ema_21_55_cross'] = df['ema_21'] - df['ema_55']
            
            # 计算EMA斜率（3周期变化百分比）
            features['ema_9_slope'] = df['ema_9'].pct_change(3) * 100
            features['ema_21_slope'] = df['ema_21'].pct_change(3) * 100
            
            # 3. 动量指标
            # RSI
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            features['rsi_14'] = df['rsi_14']
            features['rsi_14_change'] = df['rsi_14'].diff(3)
            
            # RSI背离（价格创新高但RSI没有，或价格创新低但RSI没有）
            price_higher_high = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
            rsi_not_higher = (df['rsi_14'] <= df['rsi_14'].shift(1))
            price_lower_low = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
            rsi_not_lower = (df['rsi_14'] >= df['rsi_14'].shift(1))
            
            features['rsi_divergence'] = ((price_higher_high & rsi_not_higher) * -1) + ((price_lower_low & rsi_not_lower) * 1)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            features['macd_hist_change'] = np.diff(macd_hist, prepend=np.nan)
            
            # CCI
            df['cci_20'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
            features['cci_20'] = df['cci_20']
            features['cci_change'] = df['cci_20'].diff(3)
            
            # 4. 波动性指标
            # ATR
            df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['atr_14'] = df['atr_14']
            features['atr_change'] = df['atr_14'].pct_change(3) * 100
            features['atr_percent'] = (df['atr_14'] / df['close']) * 100
            
            # 布林带
            upper, middle, lower = talib.BBANDS(
                df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            features['bb_width'] = df['bb_width']
            features['bb_position'] = df['bb_position']
            features['bb_width_change'] = df['bb_width'].pct_change(3) * 100
            
            # 5. 交易量指标
            features['volume_change'] = df['volume'].pct_change(3) * 100
            df['volume_ma'] = df['volume'].rolling(10).mean()
            features['volume_ma_ratio'] = df['volume'] / df['volume_ma']
            
            # OBV
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            features['obv_change'] = df['obv'].pct_change(3) * 100
            
            # Chaikin Money Flow
            features['adosc'] = talib.ADOSC(
                df['high'].values, df['low'].values, 
                df['close'].values, df['volume'].values,
                fastperiod=3, slowperiod=10
            )
            
            # Money Flow Index
            features['mfi'] = talib.MFI(
                df['high'].values, df['low'].values, 
                df['close'].values, df['volume'].values,
                timeperiod=14
            )
            
            # 6. 趋势指标
            # ADX
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            features['adx'] = df['adx']
            features['di_diff'] = df['plus_di'] - df['minus_di']
            features['adx_change'] = df['adx'].diff(3)
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(df['high'].values, df['low'].values, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            
            # 7. 波浪特征
            # 计算当前价格在不同周期内的相对位置
            for period in [5, 10, 20]:
                min_price = df['low'].rolling(period).min()
                max_price = df['high'].rolling(period).max()
                features[f'close_to_min_{period}'] = (df['close'] - min_price) / (max_price - min_price) * 100
                features[f'close_to_max_{period}'] = (max_price - df['close']) / (max_price - min_price) * 100
            
            # 8. 价格模式特征
            # 价格相对于MA的模式：上涨、下跌或盘整
            ma_20 = talib.SMA(df['close'].values, timeperiod=20)
            price_above_ma = (df['close'] > ma_20).astype(int)
            features['close_ma_pattern'] = price_above_ma.diff(1) * 10 + price_above_ma  # 创建一个模式的数值表示
            
            # 蜡烛图模式 - 使用几个常见的蜡烛图模式
            # 看涨模式
            features['candle_pattern_bullish'] = 0
            # 锤子线
            hammer = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 牛市吞没
            engulfing_bullish = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 穿刺线
            piercing = talib.CDLPIERCING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 晨星
            morning_star = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            
            features['candle_pattern_bullish'] = (hammer > 0) | (engulfing_bullish > 0) | (piercing > 0) | (morning_star > 0)
            features['candle_pattern_bullish'] = features['candle_pattern_bullish'].astype(int)
            
            # 看跌模式
            features['candle_pattern_bearish'] = 0
            # 吊死线
            hanging_man = talib.CDLHANGINGMAN(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 熊市吞没
            engulfing_bearish = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 乌云盖顶
            dark_cloud = talib.CDLDARKCLOUDCOVER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            # 黄昏星
            evening_star = talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            
            features['candle_pattern_bearish'] = (hanging_man < 0) | (engulfing_bearish < 0) | (dark_cloud < 0) | (evening_star < 0)
            features['candle_pattern_bearish'] = features['candle_pattern_bearish'].astype(int)
            
            # 处理缺失值
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)  # 用0填充缺失值
            
            # 确保所有特征已准备好
            for feature in self.feature_names:
                if feature not in features.columns:
                    self.logger.warning(f"特征 {feature} 未创建，添加为0")
                    features[feature] = 0
                    
            # 确保特征列顺序与feature_names一致
            features = features[self.feature_names]
            
            return features
            
        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def generate_labels(self, klines):
        """
        从K线数据生成训练标签
        
        Args:
            klines (list): K线数据
            
        Returns:
            pandas.Series: 训练标签 (-1: 卖出, 0: 持有, 1: 买入)
        """
        try:
            # 转换K线为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 确保数值类型正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加日期时间
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 确保数据足够
            if len(df) < 30:
                self.logger.error("K线数据不足，无法生成标签")
                return None
                
            # 计算未来N个周期的价格变化百分比（预测目标）
            future_periods = 5  # 未来5个15分钟K线，即75分钟
            future_change = df['close'].shift(-future_periods) / df['close'] - 1
            
            # 根据未来价格变化确定标签
            # 卖出阈值和买入阈值（可调整）
            sell_threshold = -0.005  # -0.5%
            buy_threshold = 0.005    # 0.5%
            
            # 生成标签：-1(卖出)，0(持有)，1(买入)
            labels = pd.Series(0, index=df.index)
            labels[future_change <= sell_threshold] = -1  # 卖出信号
            labels[future_change >= buy_threshold] = 1    # 买入信号
            
            # 不包括最后几个没有未来数据的K线
            labels = labels[:-future_periods]
            
            # 统计各类标签数量
            sell_count = (labels == -1).sum()
            hold_count = (labels == 0).sum()
            buy_count = (labels == 1).sum()
            
            total = len(labels)
            self.logger.info(f"标签生成完成: 总计{total}个, 买入:{buy_count}({buy_count/total*100:.1f}%), " +
                          f"持有:{hold_count}({hold_count/total*100:.1f}%), 卖出:{sell_count}({sell_count/total*100:.1f}%)")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def generate_signal(self, klines):
        """
        生成交易信号，包含信号稳定性检查
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 检查模型是否初始化
            if self.model is None:
                self.logger.info("模型未初始化，执行初始训练")
                self._initial_training()
                
            # 检查是否需要重新训练
            if self.should_retrain():
                self.logger.info("模型需要重新训练")
                if self.train_model(klines):
                    self.last_training_time = time.time()
                    self.logger.info("模型重新训练完成")
                    
            # 准备特征
            features = self.prepare_features(klines)
            if features is None:
                return 0
                
            # 获取最后一行特征用于预测
            last_features = features.iloc[-1:].copy()
                
            # 标准化特征
            last_features_scaled = self.scaler.transform(last_features)
                
            # 获取预测概率
            probabilities = self.model.predict_proba(last_features_scaled)[0]
                
            # 使用自定义标签-1, 0, 1对应的索引
            sell_prob, hold_prob, buy_prob = probabilities
                
            # 输出预测概率
            self.logger.info(f"预测概率: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
                
            # 获取当前时间
            current_time = time.time()
                
            # 检查最小信号间隔
            # if self.last_signal_time is not None and current_time - self.last_signal_time < self.min_signal_interval:
            #     self.logger.info(f"距离上次信号时间不足{self.min_signal_interval/60}分钟，保持观望")
            #     return 0
                
            # 确定初始信号
            signal = 0
            # 根据概率差异生成信号
            if buy_prob - sell_prob > self.prob_diff_threshold and buy_prob > self.confidence_threshold:
                signal = 1  # 买入信号
            elif sell_prob - buy_prob > self.prob_diff_threshold and sell_prob > self.confidence_threshold:
                signal = -1  # 卖出信号
                
            # 信号稳定性检查
            if signal != 0:
                # 更新信号历史
                self.consistent_signals.append(signal)
                if len(self.consistent_signals) > self.signal_confirmation_count:
                    self.consistent_signals.pop(0)
                    
                # 检查是否达到连续确认的阈值
                if len(self.consistent_signals) >= self.signal_confirmation_count:
                    # 判断是否所有信号一致
                    if all(s == signal for s in self.consistent_signals):
                        self.logger.info(f"信号稳定性检查通过: 连续{len(self.consistent_signals)}个周期确认{'买入' if signal == 1 else '卖出'}信号")
                        self.last_signal_time = current_time
                        return signal
                    else:
                        self.logger.info("信号稳定性检查未通过: 信号不一致")
                else:
                    self.logger.info(f"需要更多信号确认: 当前{len(self.consistent_signals)}/{self.signal_confirmation_count}")
            else:
                # 无信号时清空历史
                self.consistent_signals = []
                
            return 0  # 默认返回观望信号
                
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
            
    def monitor_position(self):
        """监控当前持仓，决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
                
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
                    self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    
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
                    self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
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
                
                # 更新最大利润记录
                if profit_rate > self.max_profit_reached:
                    self.max_profit_reached = profit_rate
                
                # 检查止盈
                if profit_rate >= self.take_profit_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查追踪止损
                if self.trailing_stop and profit_rate >= self.trailing_stop_activation:
                    # 计算回撤比例
                    drawdown = self.max_profit_reached - profit_rate
                    
                    # 如果回撤超过追踪止损距离，平仓止盈
                    if drawdown >= self.trailing_stop_distance:
                        self.logger.info(f"触发追踪止损，最大利润: {self.max_profit_reached:.4%}, " +
                                      f"当前利润: {profit_rate:.4%}, 回撤: {drawdown:.4%}")
                        self.trader.close_position()
                        return
                
                # 获取最新预测，决定是否反转平仓
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
                # 使用基础预测，不应用稳定性检查
                features = self.prepare_features(klines)
                if features is not None:
                    last_features = features.iloc[-1:].copy()
                    last_features_scaled = self.scaler.transform(last_features)
                    probabilities = self.model.predict_proba(last_features_scaled)[0]
                    sell_prob, hold_prob, buy_prob = probabilities
                    
                    # 如果信号明显反转，考虑平仓
                    if position_side == "多" and sell_prob > 0.7:
                        self.logger.info(f"预测信号明显反转 (卖出概率: {sell_prob:.4f})，平多仓")
                        self.trader.close_position()
                        return
                    elif position_side == "空" and buy_prob > 0.7:
                        self.logger.info(f"预测信号明显反转 (买入概率: {buy_prob:.4f})，平空仓")
                        self.trader.close_position()
                        return
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
