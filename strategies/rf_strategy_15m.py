import numpy as np
import pandas as pd
import talib
import time
from datetime import datetime
import logging
from strategies.base_rf_strategy import BaseRFStrategy

class RF15mStrategy(BaseRFStrategy):
    """
    RF15mStrategy - 15分钟随机森林交易策略
    
    基于15分钟K线数据的机器学习策略，使用随机森林算法预测市场走势，
    结合技术指标和统计特征，识别潜在的交易机会。
    
    特点:
    1. 15分钟时间框架: 捕捉中短期市场波动和趋势
    2. 随机森林模型: 集成学习算法，具有较高的鲁棒性和准确性
    3. 多指标融合: 结合动量、趋势和波动率指标，多维度分析市场
    4. 自适应训练: 定期重新训练模型，适应市场变化
    5. 止盈止损: 设置动态止盈止损，保护资金安全
    """
    
    MODEL_NAME = "RF15m"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化15分钟随机森林策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '15m'       # 15分钟K线
        self.training_lookback = 1500     # 训练数据回看期(K线数量)
        self.retraining_interval = 3600   # 模型重新训练间隔(秒)，约1小时
        self.check_interval = 300         # 交易检查间隔(秒)，约5分钟
        self.last_training_time = 0       # 上次训练时间
        
        # 随机森林参数
        self.n_estimators = 300           # 树的数量，增加以提高稳定性和精度
        self.max_depth = 10               # 树的最大深度，适当增加以捕捉更复杂的模式
        self.min_samples_split = 15       # 分裂所需最小样本数
        self.min_samples_leaf = 8         # 叶节点最小样本数
        self.confidence_threshold = 0.43  # 信号置信度阈值
        self.prob_diff_threshold = 0   # 概率差异阈值
        
        # 风险控制参数
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)，约8小时
        self.profit_target_pct = 0.02     # 目标利润率 2%
        self.stop_loss_pct = 0.01         # 止损率 1%
        self.risk_reward_ratio = 2.0      # 风险回报比
        
        # 趋势过滤参数
        self.trend_period = 50            # 趋势判断周期
        self.min_trend_strength = 25      # 最小趋势强度(ADX)
        self.volatility_filter = True     # 是否启用波动率过滤
        
        # 交易状态变量
        self.last_trend_check = 0         # 上次趋势检查时间
        self.trend_direction = 0          # 当前趋势方向
        self.trend_strength = 0           # 当前趋势强度
        
        # 初始化模型
        self.initialize_model()
        self._initial_training()
        
    def prepare_features(self, klines):
        """
        准备特征数据，计算技术指标作为随机森林的输入特征
        
        Args:
            klines (list): K线数据列表
            
        Returns:
            pandas.DataFrame: 特征数据
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据不足，无法计算特征")
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 设置时间索引
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 1. 价格特征
            df['returns'] = df['close'].pct_change()
            df['returns_1'] = df['returns'].shift(1)
            df['returns_2'] = df['returns'].shift(2)
            df['returns_3'] = df['returns'].shift(3)
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 价格与均线关系
            df['ema_9'] = talib.EMA(df['close'].values, timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
            df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
            
            df['price_to_ema9'] = df['close'] / df['ema_9'] - 1
            df['price_to_ema21'] = df['close'] / df['ema_21'] - 1
            df['price_to_ema50'] = df['close'] / df['ema_50'] - 1
            
            df['ema9_slope'] = (df['ema_9'] - df['ema_9'].shift(3)) / df['ema_9'].shift(3) * 100
            df['ema21_slope'] = (df['ema_21'] - df['ema_21'].shift(3)) / df['ema_21'].shift(3) * 100
            
            # 均线交叉
            df['ema_cross_9_21'] = df['ema_9'] - df['ema_21']
            df['ema_cross_21_50'] = df['ema_21'] - df['ema_50']
            
            # 2. 动量指标
            # RSI - 相对强弱指数
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_9'] = talib.RSI(df['close'].values, timeperiod=9)
            df['rsi_diff'] = df['rsi_14'] - df['rsi_14'].shift(1)
            
            # Stochastic Oscillator - 随机指标
            df['slowk'], df['slowd'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                                 fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_diff'] = df['slowk'] - df['slowd']
            
            # CCI - 顺势指标
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
            df['cci_diff'] = df['cci'] - df['cci'].shift(1)
            
            # MACD - 指数平滑异同移动平均线
            macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist
            df['macd_diff'] = df['macd'] - df['macdsignal']
            
            # 3. 趋势指标
            # ADX - 平均趋向指数
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['adx_trend'] = df['plus_di'] - df['minus_di']
            
            # 4. 波动率指标
            # ATR - 平均真实波幅
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['atr_percent'] = df['atr'] / df['close'] * 100
            
            # Bollinger Bands - 布林带
            upperband, middleband, lowerband = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = upperband
            df['bb_middle'] = middleband
            df['bb_lower'] = lowerband
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 5. 成交量指标
            df['volume_ema5'] = talib.EMA(df['volume'].values, timeperiod=5)
            df['volume_ema10'] = talib.EMA(df['volume'].values, timeperiod=10)
            df['volume_ratio'] = df['volume'] / df['volume_ema10']
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            df['obv_slope'] = (df['obv'] - df['obv'].shift(3)) / df['obv'].shift(3) * 100
            
            # 6. 蜡烛图特征
            # 实体比例
            df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            # 上影线比例
            df['upper_shadow'] = (df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])) / (df['high'] - df['low'])
            # 下影线比例
            df['lower_shadow'] = (df['close'].where(df['close'] <= df['open'], df['open']) - df['low']) / (df['high'] - df['low'])
            
            # 处理缺失值
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            # 创建特征集
            feature_columns = [
                # 价格特征
                'returns_1', 'returns_2', 'returns_3', 
                'price_to_ema9', 'price_to_ema21', 'price_to_ema50',
                'ema9_slope', 'ema21_slope', 'ema_cross_9_21', 'ema_cross_21_50',
                
                # 动量指标
                'rsi_14', 'rsi_9', 'rsi_diff', 'slowk', 'slowd', 'stoch_diff',
                'cci', 'cci_diff', 'macd', 'macdsignal', 'macdhist', 'macd_diff',
                
                # 趋势指标
                'adx', 'plus_di', 'minus_di', 'adx_trend',
                
                # 波动率指标
                'atr_percent', 'bb_width', 'bb_position',
                
                # 成交量指标
                'volume_ratio', 'obv_slope',
                
                # 蜡烛图特征
                'body_ratio', 'upper_shadow', 'lower_shadow'
            ]
            
            # 返回特征集
            features = df[feature_columns].copy()
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"准备特征数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_labels(self, klines):
        """
        生成训练标签，根据未来价格变动确定标签
        
        标签说明:
        -1: 卖出 (未来价格下跌)
        0: 观望 (未来价格横盘)
        1: 买入 (未来价格上涨)
        
        Args:
            klines (list): K线数据列表
            
        Returns:
            pandas.Series: 训练标签
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据不足，无法生成标签")
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 设置时间索引
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 计算未来N根K线的价格变动
            future_periods = 4  # 未来4根15分钟K线，约1小时
            
            # 未来收盘价变动百分比
            df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
            
            # 计算波动率作为标签阈值
            volatility = df['close'].pct_change().rolling(20).std() * 100  # 20周期标准差
            avg_volatility = volatility.mean()
            
            # 动态阈值，基于波动率设置
            up_threshold = max(0.005, avg_volatility * 0.5)    # 默认0.5%，或波动率的一半
            down_threshold = -max(0.005, avg_volatility * 0.5) # 负值
            
            # 生成标签
            df['label'] = 0  # 默认为观望
            df.loc[df['future_return'] > up_threshold, 'label'] = 1    # 上涨
            df.loc[df['future_return'] < down_threshold, 'label'] = -1  # 下跌
            
            # 降噪：消除剧烈波动中的观望信号
            high_volatility_mask = volatility > avg_volatility * 1.5
            consecutive_ups = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
            consecutive_downs = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
            
            # 在高波动且连续上涨中不发出观望信号
            df.loc[(high_volatility_mask) & (consecutive_ups) & (df['label'] == 0), 'label'] = 1
            # 在高波动且连续下跌中不发出观望信号
            df.loc[(high_volatility_mask) & (consecutive_downs) & (df['label'] == 0), 'label'] = -1
            
            # 查看标签分布
            label_counts = df['label'].value_counts().to_dict()
            bearish_count = label_counts.get(-1, 0)
            neutral_count = label_counts.get(0, 0)
            bullish_count = label_counts.get(1, 0)
            self.logger.info(f"标签分布: 看跌={bearish_count}, 盘整={neutral_count}, 看涨={bullish_count}")
            
            # 返回标签序列
            labels = df['label'].copy()
            labels = labels.dropna()
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def analyze_market_trend(self, klines, lookback=None):
        """
        分析市场趋势，用于过滤交易信号
        
        Args:
            klines (list): K线数据列表
            lookback (int): 回看周期，默认为self.trend_period
            
        Returns:
            tuple: (trend_direction, trend_strength, trend_score)
                trend_direction: 趋势方向，1为上升，-1为下降，0为盘整
                trend_strength: 趋势强度，0-100
                trend_score: 趋势评分，-100到100，越大表示上升趋势越强
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                return 0, 0, 0
                
            # 使用指定回看周期或默认趋势周期
            lookback = lookback or self.trend_period
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 计算趋势指标
            # 1. ADX - 趋势强度
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            plus_di = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            minus_di = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # 2. 移动平均线
            ema10 = talib.EMA(df['close'].values, timeperiod=10)
            ema20 = talib.EMA(df['close'].values, timeperiod=20)
            ema50 = talib.EMA(df['close'].values, timeperiod=50)
            
            # 3. MACD
            macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # 获取最新指标值
            latest_adx = adx[-1]
            latest_plus_di = plus_di[-1]
            latest_minus_di = minus_di[-1]
            
            # 判断趋势强度 (0-100)
            trend_strength = min(100, float(latest_adx))
            
            # 判断趋势方向
            if latest_plus_di > latest_minus_di:
                trend_direction = 1  # 上升趋势
            elif latest_minus_di > latest_plus_di:
                trend_direction = -1  # 下降趋势
            else:
                trend_direction = 0  # 无明显趋势
                
            # 计算趋势评分 (-100 到 100)
            # 1. EMA趋势得分 (-35 到 35)
            ema_score = 0
            if ema10[-1] > ema20[-1]:
                ema_score += 17.5
            else:
                ema_score -= 17.5
                
            if ema20[-1] > ema50[-1]:
                ema_score += 17.5
            else:
                ema_score -= 17.5
                
            # 2. ADX趋势得分 (-30 到 30)
            di_diff = latest_plus_di - latest_minus_di
            adx_factor = min(1, latest_adx / 50)  # 将ADX归一化到0-1
            adx_score = di_diff * adx_factor / 2  # 通常DI在0-50范围，除以2归一化
            
            # 3. MACD趋势得分 (-20 到 20)
            macd_score = macd[-1] * 300  # 将MACD值放大，MACD通常很小
            macd_score = max(-20, min(20, macd_score))  # 限制在-20到20范围内
            
            # 4. 价格趋势得分 (-15 到 15)
            price_change_pct = (df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1) * 100
            price_score = price_change_pct * 1.5  # 放大影响
            price_score = max(-15, min(15, price_score))  # 限制在-15到15范围内
            
            # 综合趋势评分
            trend_score = ema_score + adx_score + macd_score + price_score
            trend_score = max(-100, min(100, trend_score))  # 限制在-100到100范围内
            
            # 输出趋势分析结果
            self.logger.info(f"趋势分析: 方向={trend_direction}, 强度={trend_strength:.1f}, 评分={trend_score:.1f}")
            
            # 更新策略状态
            self.trend_direction = trend_direction
            self.trend_strength = trend_strength
            self.last_trend_check = time.time()
            
            return trend_direction, trend_strength, trend_score
            
        except Exception as e:
            self.logger.error(f"分析市场趋势失败: {str(e)}")
            return 0, 0, 0
    
    def filter_signal(self, signal, trend_direction, trend_strength, trend_score):
        """
        根据趋势过滤信号，避免逆势交易
        
        Args:
            signal (int): 原始交易信号 (-1, 0, 1)
            trend_direction (int): 趋势方向 (-1, 0, 1)
            trend_strength (float): 趋势强度 (0-100)
            trend_score (float): 趋势评分 (-100 到 100)
            
        Returns:
            int: 过滤后的信号
        """
        # 没有信号则直接返回
        if signal == 0:
            return 0
            
        # 1. 强趋势下，抑制逆势信号
        if trend_strength >= self.min_trend_strength:
            # 上升趋势中抑制卖出信号
            if trend_direction > 0 and signal < 0 and trend_score > 30:
                self.logger.info("抑制卖出信号，因为处于上升趋势")
                return 0
                
            # 下降趋势中抑制买入信号
            if trend_direction < 0 and signal > 0 and trend_score < -30:
                self.logger.info("抑制买入信号，因为处于下降趋势")
                return 0
                
        # 2. 极弱趋势下，增强顺势信号
        if trend_strength < 15:
            # 在无明显趋势时保持观望
            self.logger.info(f"趋势强度不足({trend_strength:.1f})，维持观望信号")
            return 0
            
        # 3. 价格反转情况下，增强反转信号
        if abs(trend_score) > 70:  # 极强趋势可能即将耗尽
            if trend_score > 70 and signal < 0:  # 上涨过度，增强卖出信号
                self.logger.info("增强卖出信号，因为上涨过度(可能见顶)")
                return signal
            if trend_score < -70 and signal > 0:  # 下跌过度，增强买入信号
                self.logger.info("增强买入信号，因为下跌过度(可能见底)")
                return signal
                
        # 4. 默认保持原信号
        return signal
    
    def generate_signal(self, klines):
        """
        生成交易信号，结合模型预测和趋势分析
        
        Returns:
            int: 交易信号，-1(卖出)，0(观望)，1(买入)
        """
        try:
            # 检查是否需要重新训练模型
            if self.should_retrain():
                self.logger.info("开始重新训练模型...")
                if self.train_model(klines):
                    self.logger.info("模型重新训练完成")
                    self.last_training_time = time.time()
                else:
                    self.logger.error("模型重新训练失败")
                    
            # 分析市场趋势
            trend_direction, trend_strength, trend_score = self.analyze_market_trend(klines)
            
            # 准备特征
            features = self.prepare_features(klines)
            if features is None:
                return 0
                
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 模型预测
            probabilities = self.model.predict_proba(features_scaled)[-1]
            
            # 根据类标签顺序调整输出
            # 确保顺序是 [-1, 0, 1] 对应 [卖出, 观望, 买入]
            class_order = self.model.classes_
            probability_map = {cls: prob for cls, prob in zip(class_order, probabilities)}
            sell_prob = probability_map.get(-1, 0)
            hold_prob = probability_map.get(0, 0)
            buy_prob = probability_map.get(1, 0)
            
            # 输出预测概率
            self.logger.info(f"模型预测概率: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
            
            # 生成初始信号
            if buy_prob > sell_prob + self.prob_diff_threshold:
                signal = 1  # 买入信号
            elif sell_prob > buy_prob + self.prob_diff_threshold:
                signal = -1  # 卖出信号
            else:
                signal = 0  # 观望信号
                
            # 应用趋势过滤
            filtered_signal = self.filter_signal(signal, trend_direction, trend_strength, trend_score)
            
            # 如果信号被过滤改变，记录日志
            if filtered_signal != signal:
                self.logger.info(f"原始信号 {signal} 被过滤为 {filtered_signal}")
                
            return filtered_signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0  # 发生错误时返回观望信号
            
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
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
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) * self.trader.symbol_config.get('leverage', 5)
                    trade_amount = trade_amount / current_price
                    
                    # 开多仓
                    self.trader.open_long(amount=trade_amount)
                    self.logger.info(f"开多仓 - 数量: {trade_amount:.4f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    
                elif signal == -1:  # 卖出信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) * self.trader.symbol_config.get('leverage', 5)
                    trade_amount = trade_amount / current_price
                    
                    # 开空仓
                    self.trader.open_short(amount=trade_amount)
                    self.logger.info(f"开空仓 - 数量: {trade_amount:.4f}, 价格: {current_price}")
                    
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
                
                # 动态止盈: 随着盈利增加调整止盈点
                dynamic_take_profit = self.profit_target_pct
                if profit_rate > self.profit_target_pct * 0.5:
                    # 盈利超过目标的一半，提高止盈点防止回撤
                    dynamic_take_profit = max(self.profit_target_pct * 0.7, profit_rate * 0.8)
                
                # 检查止盈
                if profit_rate >= dynamic_take_profit:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 获取最新K线和信号，检查趋势是否反转
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                signal = self.generate_signal(klines)
                
                # 如果持有多仓但出现强烈卖出信号，平仓
                if position_side == "多" and signal == -1:
                    self.logger.info("当前持有多仓，但出现强烈卖出信号，平仓")
                    self.trader.close_position()
                    return
                    
                # 如果持有空仓但出现强烈买入信号，平仓
                if position_side == "空" and signal == 1:
                    self.logger.info("当前持有空仓，但出现强烈买入信号，平仓")
                    self.trader.close_position()
                    return
                
                # 如果行情出现回撤，提前止盈
                if profit_rate > self.profit_target_pct * 0.3:  # 至少达到目标利润的30%
                    # 检查近期K线是否出现回撤
                    recent_close_prices = [float(k[4]) for k in klines[-5:]]  # 最近5根K线收盘价
                    
                    if position_side == "多" and recent_close_prices[-1] < max(recent_close_prices[:-1]) * 0.995:
                        self.logger.info(f"多仓盈利中出现价格回撤，提前止盈，利润率: {profit_rate:.4%}")
                        self.trader.close_position()
                        return
                        
                    if position_side == "空" and recent_close_prices[-1] > min(recent_close_prices[:-1]) * 1.005:
                        self.logger.info(f"空仓盈利中出现价格反弹，提前止盈，利润率: {profit_rate:.4%}")
                        self.trader.close_position()
                        return
                        
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
