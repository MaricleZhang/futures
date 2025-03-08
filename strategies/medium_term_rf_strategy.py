import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class MediumTermRFStrategy(BaseRFStrategy):
    """MediumTermRFStrategy - 中期随机森林交易策略
    
    专门针对中期交易优化的随机森林策略模型，使用5分钟K线数据，
    结合多种技术指标、波动率分析和市场结构特征来捕捉中期趋势机会。
    
    特点：
    1. 中期特征提取：使用5到120分钟的中期技术指标
    2. 波动率适应：根据市场波动率动态调整进出场条件
    3. 市场结构分析：通过支撑阻力、趋势线和价格形态分析市场结构
    4. 整合多因子信号：综合考量趋势、动量、反转和波动等多维信号
    5. 动态止盈止损：根据市场环境自适应调整风险参数
    """
    
    MODEL_NAME = "MediumTermRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化中期随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数
        self.n_estimators = 200     # 树的数量
        self.max_depth = 10         # 树的最大深度，增加以捕捉更复杂的模式
        self.min_samples_split = 15 # 分裂所需最小样本数，减少以增加适应性
        self.min_samples_leaf = 8   # 叶节点最小样本数，减少以增加适应性
        self.confidence_threshold = 0.60  # 信号置信度阈值，增加以减少虚假信号
        self.prob_diff_threshold = 0.10   # 概率差异阈值
        
        # K线设置
        self.kline_interval = '5m'  # 5分钟K线
        self.training_lookback = 800  # 训练数据回看周期，增加以捕捉更多的市场周期
        self.retraining_interval = 1800  # 30分钟重新训练一次
        self.check_interval = 300   # 策略检查间隔(秒)
        
        # 风险控制参数
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.1    # 目标利润率 1%
        self.stop_loss_pct = 0.005       # 止损率 0.5%
        self.max_trades_per_hour = 2     # 每小时最大交易次数
        self.min_vol_percentile = 40     # 最小成交量百分位
        
        # 市场状态评估参数
        self.trend_strength_threshold = 0.6  # 趋势强度阈值
        self.market_volatility_lookback = 20  # 波动率计算回看周期
        self.support_resistance_lookback = 50  # 支撑阻力计算回看周期
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
    
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # === 1. 价格特征 ===
            # 价格变化率 - 不同周期
            for period in [1, 3, 6, 12, 24]:  # 5分钟到2小时
                features[f'price_change_{period}'] = df['close'].pct_change(period)
            
            # 价格相对于移动平均线
            for period in [10, 20, 50, 100]:
                ma = df['close'].rolling(window=period).mean()
                features[f'price_to_ma_{period}'] = df['close'] / ma - 1
            
            # 高低价格范围
            features['high_low_range'] = (df['high'] - df['low']) / df['close']
            features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
            
            # 价格波动和趋势
            features['price_volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            features['price_velocity'] = df['close'].diff(5) / 5  # 5根K线的平均变化速度
            features['price_acceleration'] = features['price_velocity'].diff(5) / 5  # 速度的变化率
            
            # === 2. 技术指标 ===
            # RSI - 多周期
            for period in [6, 14, 21]:
                features[f'rsi_{period}'] = pd.Series(talib.RSI(df['close'].values, timeperiod=period), index=df.index)
                # RSI斜率
                features[f'rsi_{period}_slope'] = features[f'rsi_{period}'].diff(3)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            features['macd'] = pd.Series(macd, index=df.index)
            features['macd_signal'] = pd.Series(macd_signal, index=df.index)
            features['macd_hist'] = pd.Series(macd_hist, index=df.index)
            features['macd_hist_slope'] = pd.Series(macd_hist, index=df.index).diff(3)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features['bb_upper'] = pd.Series(upper, index=df.index)
            features['bb_middle'] = pd.Series(middle, index=df.index)
            features['bb_lower'] = pd.Series(lower, index=df.index)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # ATR - 波动率
            features['atr'] = pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            features['atr_percent'] = features['atr'] / df['close']
            
            # 随机指标
            features['slowk'], features['slowd'] = talib.STOCH(
                df['high'].values, df['low'].values, df['close'].values, 
                fastk_period=14, slowk_period=3, slowk_matype=0, 
                slowd_period=3, slowd_matype=0
            )
            features['slowk'] = pd.Series(features['slowk'], index=df.index)
            features['slowd'] = pd.Series(features['slowd'], index=df.index)
            
            # 趋势强度指标 - ADX
            features['adx'] = pd.Series(talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            features['plus_di'] = pd.Series(talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            features['minus_di'] = pd.Series(talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            
            # 动量指标
            features['roc'] = pd.Series(talib.ROC(df['close'].values, timeperiod=10), index=df.index)
            features['cci'] = pd.Series(talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            features['mfi'] = pd.Series(talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14), index=df.index)
            
            # === 3. 成交量特征 ===
            # 成交量变化率
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma'] = df['volume'].rolling(window=20).mean()
            features['relative_volume'] = df['volume'] / features['volume_ma']
            
            # 成交量趋势
            features['volume_trend'] = df['volume'].diff(5).rolling(window=5).mean()
            
            # OBV - 能量潮指标
            features['obv'] = pd.Series(talib.OBV(df['close'].values, df['volume'].values), index=df.index)
            features['obv_ma'] = features['obv'].rolling(window=20).mean()
            features['obv_slope'] = (features['obv'] - features['obv'].shift(5)) / 5
            
            # 价量相关性
            features['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
            
            # === 4. 市场结构特征 ===
            # 支撑阻力水平
            roll_high = df['high'].rolling(window=self.support_resistance_lookback).max()
            roll_low = df['low'].rolling(window=self.support_resistance_lookback).min()
            
            features['dist_to_resistance'] = (roll_high - df['close']) / df['close']
            features['dist_to_support'] = (df['close'] - roll_low) / df['close']
            
            # 价格与趋势线的关系
            sma50 = df['close'].rolling(window=50).mean()
            sma200 = df['close'].rolling(window=200).mean()
            features['sma_cross'] = sma50 - sma200
            features['sma_cross_slope'] = features['sma_cross'].diff(5)
            
            # 通道宽度和位置
            for period in [20, 50]:
                roll_high_period = df['high'].rolling(window=period).max()
                roll_low_period = df['low'].rolling(window=period).min()
                channel_width = (roll_high_period - roll_low_period) / df['close']
                channel_position = (df['close'] - roll_low_period) / (roll_high_period - roll_low_period)
                
                features[f'channel_width_{period}'] = channel_width
                features[f'channel_position_{period}'] = channel_position
            
            # === 5. 综合趋势评分 ===
            # 趋势组合信号
            trend_signals = pd.Series(0.0, index=df.index)
            
            # 添加均线趋势信号
            for period in [10, 20, 50]:
                ma = df['close'].rolling(window=period).mean()
                trend_signals += np.where(df['close'] > ma, 1/3, -1/3)
            
            # 添加MACD信号
            trend_signals += np.where(features['macd_hist'] > 0, 0.5, -0.5)
            
            # 添加RSI信号
            rsi14 = features['rsi_14']
            trend_signals += np.where(rsi14 > 50, 0.3, -0.3)
            
            # 添加ADX信号
            adx = features['adx']
            plus_di = features['plus_di']
            minus_di = features['minus_di']
            trend_signals += np.where((adx > 25) & (plus_di > minus_di), 0.4, np.where((adx > 25) & (plus_di < minus_di), -0.4, 0))
            
            # 归一化趋势信号到[-1, 1]范围
            features['trend_score'] = trend_signals.clip(-1, 1)
            
            # === 6. 异常模式检测 ===
            # 价格缺口检测
            gap_up = df['low'] > df['high'].shift(1)
            gap_down = df['high'] < df['low'].shift(1)
            features['gap_up'] = gap_up.astype(int)
            features['gap_down'] = gap_down.astype(int)
            
            # 极端波动检测
            avg_range = (df['high'] - df['low']).rolling(window=20).mean()
            features['extreme_volatility'] = ((df['high'] - df['low']) > 2 * avg_range).astype(int)
            
            # 成交量异常
            features['volume_spike'] = (df['volume'] > 2 * features['volume_ma']).astype(int)
            
            # 删除NaN值
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_labels(self, klines):
        """生成训练标签"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据不足，无法生成标签")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来多个时间窗口的收益率
            future_returns = []
            lookback_windows = [6, 12, 24, 48]  # 30分钟到4小时的预测窗口
            weights = [0.4, 0.3, 0.2, 0.1]  # 权重偏向近期
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算技术指标作为标签的补充条件
            rsi = pd.Series(talib.RSI(df['close'].values, timeperiod=14), index=df.index)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            macd_hist = pd.Series(macd_hist, index=df.index)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(
                df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            bb_position = (df['close'] - pd.Series(lower, index=df.index)) / (pd.Series(upper, index=df.index) - pd.Series(lower, index=df.index))
            
            # 趋势强度 - 基于多个指标的综合评分
            adx = pd.Series(talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            plus_di = pd.Series(talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            minus_di = pd.Series(talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            
            trend_strength = (
                0.3 * ((plus_di - minus_di) / adx) + 
                0.3 * ((rsi - 50) / 25) + 
                0.3 * (macd_hist / (df['close'] * 0.01)) +
                0.1 * (2 * bb_position - 1)
            )
            
            # 波动率
            atr = pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            volatility = atr / df['close']
            
            # 动态阈值 - 根据当前市场环境调整阈值
            base_threshold = 0.005  # 基础收益率阈值 0.5%
            
            # 根据波动率调整阈值
            vol_factor = volatility / volatility.rolling(window=50).mean()
            threshold = base_threshold * vol_factor
            
            # 根据趋势强度调整阈值
            trend_adjustment = abs(trend_strength).clip(0.5, 1.5)
            buy_threshold = threshold * trend_adjustment
            sell_threshold = -threshold * trend_adjustment
            
            # 防止阈值过大或过小
            buy_threshold = buy_threshold.clip(0.002, 0.01)  # 限制在0.2%到1%之间
            sell_threshold = sell_threshold.clip(-0.01, -0.002)  # 限制在-1%到-0.2%之间
            
            # 生成标签
            labels = pd.Series(0, index=df.index)  # 默认观望
            
            # 过滤有效样本
            valid_mask = ~(weighted_returns.isna() | trend_strength.isna() | 
                         buy_threshold.isna() | sell_threshold.isna() | 
                         rsi.isna() | bb_position.isna())
            
            # 买入条件
            buy_condition = (
                (weighted_returns > buy_threshold) &  # 预期收益率超过阈值
                (rsi < 70) &  # RSI未超买
                (bb_position < 0.8) &  # 价格未接近布林带上轨
                (trend_strength > 0)  # 趋势方向向上
            )
            
            # 卖出条件
            sell_condition = (
                (weighted_returns < sell_threshold) &  # 预期亏损率超过阈值
                (rsi > 30) &  # RSI未超卖
                (bb_position > 0.2) &  # 价格未接近布林带下轨
                (trend_strength < 0)  # 趋势方向向下
            )
            
            # 分配标签
            labels[buy_condition & valid_mask] = 1    # 买入
            labels[sell_condition & valid_mask] = -1  # 卖出
            
            # 保留有效样本
            labels = labels[valid_mask]
            
            return labels
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def generate_signal(self, klines):
        """生成交易信号
        
        根据模型预测和市场条件，生成交易信号
        
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            # 准备特征数据
            features = self.prepare_features(klines)
            if features is None:
                return 0
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 模型预测
            probabilities = self.model.predict_proba(features_scaled)[-1]
            
            # 获取主要的市场指标
            adx = features['adx'].iloc[-1]
            rsi = features['rsi_14'].iloc[-1]
            trend_score = features['trend_score'].iloc[-1]
            atr_percent = features['atr_percent'].iloc[-1]
            
            # 记录指标值和预测结果
            self.logger.info(f"市场指标: ADX={adx:.1f}, RSI={rsi:.1f}, 趋势分数={trend_score:.2f}, ATR百分比={atr_percent*100:.2f}%")
            self.logger.info(f"预测概率: 卖出={probabilities[0]:.4f}, 观望={probabilities[1]:.4f}, 买入={probabilities[2]:.4f}")
            
            # 信号映射: 0 -> 卖出(-1), 1 -> 观望(0), 2 -> 买入(1)
            raw_prediction = np.argmax(probabilities)
            
            # 检查置信度是否达到阈值
            max_prob = probabilities[raw_prediction]
            if raw_prediction != 1 and max_prob < self.confidence_threshold:
                self.logger.info(f"信号置信度({max_prob:.4f})低于阈值({self.confidence_threshold})，切换为观望信号")
                return 0
            
            # 检查市场条件是否支持信号
            # 如果是买入信号，但市场指标不支持，则降级为观望
            if raw_prediction == 2:  # 买入信号
                if rsi > 75:
                    self.logger.info(f"RSI过高({rsi:.1f})，降级买入信号为观望")
                    return 0
                if trend_score < -0.5:
                    self.logger.info(f"趋势分数过低({trend_score:.2f})，降级买入信号为观望")
                    return 0
                if adx < 20:
                    self.logger.info(f"ADX过低({adx:.1f})，趋势不明确，降级买入信号为观望")
                    return 0
            
            # 如果是卖出信号，但市场指标不支持，则降级为观望
            if raw_prediction == 0:  # 卖出信号
                if rsi < 25:
                    self.logger.info(f"RSI过低({rsi:.1f})，降级卖出信号为观望")
                    return 0
                if trend_score > 0.5:
                    self.logger.info(f"趋势分数过高({trend_score:.2f})，降级卖出信号为观望")
                    return 0
                if adx < 20:
                    self.logger.info(f"ADX过低({adx:.1f})，趋势不明确，降级卖出信号为观望")
                    return 0
            
            # 检查波动率是否适合交易
            avg_atr_percent = features['atr_percent'].iloc[-20:].mean()
            if atr_percent < avg_atr_percent * 0.5 and raw_prediction != 1:
                self.logger.info(f"当前波动率过低，降级信号为观望")
                return 0
            
            # 将预测结果转换为交易信号
            signal_mapping = {0: -1, 1: 0, 2: 1}
            signal = signal_mapping[raw_prediction]
            
            # 记录最终信号
            signal_name = "卖出" if signal == -1 else ("观望" if signal == 0 else "买入")
            self.logger.info(f"生成{signal_name}信号，置信度: {max_prob:.4f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0  # 出错时返回观望信号
    
    def monitor_position(self):
        """监控当前持仓，根据策略决定是否平仓或重新训练模型"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 获取最新K线数据
            klines = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if len(klines) < self.training_lookback:
                self.logger.error(f"K线数据不足，需要至少 {self.training_lookback} 根")
                return
            
            # 获取当前市场价格
            current_price = self.trader.get_market_price()
            
            # 检查是否需要重新训练模型
            if self.should_retrain():
                self.logger.info("开始重新训练模型...")
                training_klines = self.trader.get_klines(
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                if self.train_model(training_klines):
                    self.logger.info("模型重新训练完成")
                    self.last_training_time = time.time()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 生成交易信号
                signal = self.generate_signal(klines)
                
                # 获取当前小时
                current_hour = time.localtime().tm_hour
                
                # 检查交易次数限制
                if self.last_trade_hour != current_hour:
                    self.last_trade_hour = current_hour
                    self.trade_count_hour = 0
                
                if self.trade_count_hour >= self.max_trades_per_hour:
                    self.logger.info(f"已达到每小时最大交易次数({self.max_trades_per_hour})，本小时不再开新仓")
                    return
                
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
                    
                    # 执行开多仓操作
                    order = self.trader.open_long(amount=trade_amount)
                    if order:
                        self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        
                        # 更新交易计数
                        self.trade_count_hour += 1
                
                elif signal == -1:  # 卖出信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 执行开空仓操作
                    order = self.trader.open_short(amount=trade_amount)
                    if order:
                        self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        
                        # 更新交易计数
                        self.trade_count_hour += 1
            
            # 如果有持仓，检查是否需要平仓
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                position_side = "多" if position_amount > 0 else "空"
                
                # 计算持仓时间
                current_time = time.time()
                holding_time_minutes = 0
                
                if self.position_entry_time is not None:
                    holding_time_minutes = (current_time - self.position_entry_time) / 60
                    
                    # 检查最大持仓时间
                    if holding_time_minutes >= self.max_position_hold_time:
                        self.logger.info(f"持仓时间超过{self.max_position_hold_time}分钟，执行平仓")
                        self.trader.close_position()
                        return
                
                # 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # 计算动态止盈止损阈值
                features = self.prepare_features(klines)
                if features is not None:
                    # 获取最新的ATR百分比作为波动率指标
                    atr_percent = features['atr_percent'].iloc[-1]
                    avg_atr_percent = features['atr_percent'].iloc[-20:].mean()
                    
                    # 计算动态止盈阈值 - 波动率高时提高止盈目标
                    dynamic_profit_target = self.profit_target_pct
                    if atr_percent > avg_atr_percent * 1.5:  # 如果波动率高于平均的1.5倍
                        dynamic_profit_target = self.profit_target_pct * 1.5
                    
                    # 计算动态止损阈值 - 波动率高时放宽止损
                    dynamic_stop_loss = self.stop_loss_pct
                    if atr_percent > avg_atr_percent * 1.5:  # 如果波动率高于平均的1.5倍
                        dynamic_stop_loss = self.stop_loss_pct * 1.5
                    
                    # 根据持仓时间调整止盈止损 - 持仓时间越长，止盈要求越低，止损越严格
                    time_factor = min(holding_time_minutes / (self.max_position_hold_time / 2), 1.0)
                    dynamic_profit_target *= (1.0 - time_factor * 0.5)  # 随时间降低止盈目标
                    dynamic_stop_loss *= (1.0 - time_factor * 0.3)     # 随时间收紧止损
                    
                    # 获取市场趋势信息以调整止盈止损
                    trend_score = features['trend_score'].iloc[-1]
                    if (position_side == "多" and trend_score < -0.5) or (position_side == "空" and trend_score > 0.5):
                        # 如果持仓方向与趋势方向相反，收紧止损
                        dynamic_stop_loss *= 0.7
                    
                    # 检查止盈
                    if profit_rate >= dynamic_profit_target:
                        self.logger.info(f"达到动态止盈条件，利润率: {profit_rate:.4%} >= {dynamic_profit_target:.4%}，执行平仓")
                        self.trader.close_position()
                        return
                    
                    # 检查止损
                    if profit_rate <= -dynamic_stop_loss:
                        self.logger.info(f"达到动态止损条件，亏损率: {profit_rate:.4%} <= -{dynamic_stop_loss:.4%}，执行平仓")
                        self.trader.close_position()
                        return
                else:
                    # 如果无法获取特征数据，使用基本的止盈止损策略
                    # 检查止盈
                    if profit_rate >= self.profit_target_pct:
                        self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，执行平仓")
                        self.trader.close_position()
                        return
                    
                    # 检查止损
                    if profit_rate <= -self.stop_loss_pct:
                        self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，执行平仓")
                        self.trader.close_position()
                        return
                
                # 生成新信号来检查是否需要平仓
                signal = self.generate_signal(klines)
                
                # 如果信号与当前持仓方向相反，则平仓
                if (position_side == "多" and signal == -1) or (position_side == "空" and signal == 1):
                    self.logger.info(f"根据策略信号({signal})与当前持仓方向({position_side})相反，执行平仓")
                    self.trader.close_position()
                    return
                
                # 特殊情况：信号强度很高且与持仓方向相反时立即平仓
                features_scaled = self.scaler.transform(features)
                probabilities = self.model.predict_proba(features_scaled)[-1]
                
                if position_side == "多" and probabilities[0] > 0.75:  # 有很强的卖出信号
                    self.logger.info(f"检测到强烈的卖出信号(概率:{probabilities[0]:.4f})，执行平仓")
                    self.trader.close_position()
                    return
                
                if position_side == "空" and probabilities[2] > 0.75:  # 有很强的买入信号
                    self.logger.info(f"检测到强烈的买入信号(概率:{probabilities[2]:.4f})，执行平仓")
                    self.trader.close_position()
                    return
                
                # 趋势反转检查 - 基于技术指标
                if features is not None:
                    adx = features['adx'].iloc[-1]
                    plus_di = features['plus_di'].iloc[-1]
                    minus_di = features['minus_di'].iloc[-1]
                    rsi14 = features['rsi_14'].iloc[-1]
                    macd_hist = features['macd_hist'].iloc[-1]
                    
                    # 多仓反转信号
                    if position_side == "多" and adx > 25:
                        if minus_di > plus_di and macd_hist < 0 and rsi14 < 45:
                            self.logger.info(f"检测到多头趋势反转信号，执行平仓")
                            self.trader.close_position()
                            return
                    
                    # 空仓反转信号
                    if position_side == "空" and adx > 25:
                        if plus_di > minus_di and macd_hist > 0 and rsi14 > 55:
                            self.logger.info(f"检测到空头趋势反转信号，执行平仓")
                            self.trader.close_position()
                            return
        
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def calculate_market_conditions(self, klines):
        """分析市场状况，用于优化交易决策"""
        try:
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算基本指标
            rsi14 = pd.Series(talib.RSI(df['close'].values, timeperiod=14), index=df.index)
            adx = pd.Series(talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            atr = pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14), index=df.index)
            atr_percent = atr / df['close']
            
            # 计算均线
            ema50 = df['close'].ewm(span=50, adjust=False).mean()
            ema200 = df['close'].ewm(span=200, adjust=False).mean()
            
            # 获取最新值
            current_rsi = rsi14.iloc[-1]
            current_adx = adx.iloc[-1]
            current_atr_pct = atr_percent.iloc[-1]
            avg_atr_pct = atr_percent.iloc[-20:].mean()
            ema50_last = ema50.iloc[-1]
            ema200_last = ema200.iloc[-1]
            
            # 布林带宽度 - 判断震荡/趋势
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            bb_width = (pd.Series(upper, index=df.index) - pd.Series(lower, index=df.index)) / pd.Series(middle, index=df.index)
            current_bb_width = bb_width.iloc[-1]
            avg_bb_width = bb_width.iloc[-50:].mean()
            
            # 市场类型分析
            market_type = "unknown"
            
            # 判断市场类型：震荡市/趋势市
            if current_adx < 20:
                market_type = "震荡市"
            elif current_adx >= 20:
                # 判断趋势方向
                if df['close'].iloc[-1] > ema50_last and ema50_last > ema200_last:
                    market_type = "上升趋势市"
                elif df['close'].iloc[-1] < ema50_last and ema50_last < ema200_last:
                    market_type = "下降趋势市"
                else:
                    market_type = "趋势转换中"
            
            # 判断波动性
            volatility = "正常"
            if current_atr_pct > avg_atr_pct * 1.5:
                volatility = "高波动"
            elif current_atr_pct < avg_atr_pct * 0.5:
                volatility = "低波动"
            
            # 市场强度
            market_strength = "中等"
            if current_adx > 40:
                market_strength = "强劲"
            elif current_adx < 15:
                market_strength = "弱势"
            
            # 过热/过冷分析
            market_heat = "正常"
            if current_rsi > 70:
                market_heat = "过热"
            elif current_rsi < 30:
                market_heat = "过冷"
            
            # 布林带状态
            bb_state = "正常"
            if current_bb_width > avg_bb_width * 1.5:
                bb_state = "扩张"
            elif current_bb_width < avg_bb_width * 0.5:
                bb_state = "收缩"
            
            market_conditions = {
                "market_type": market_type,
                "volatility": volatility,
                "market_strength": market_strength,
                "market_heat": market_heat,
                "bb_state": bb_state,
                "rsi": current_rsi,
                "adx": current_adx,
                "atr_percent": current_atr_pct * 100,  # 转换为百分比
                "bb_width": current_bb_width
            }
            
            self.logger.info(f"市场状况分析: {market_type}, {volatility}波动, {market_strength}趋势, {market_heat}, 布林带{bb_state}")
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"市场状况分析失败: {str(e)}")
            return None
            
    def optimize_rf_parameters(self, klines):
        """根据市场状况动态优化随机森林模型参数"""
        try:
            market_conditions = self.calculate_market_conditions(klines)
            if not market_conditions:
                return
                
            # 获取市场状态
            market_type = market_conditions["market_type"]
            volatility = market_conditions["volatility"]
            market_strength = market_conditions["market_strength"]
            
            # 基础参数
            base_n_estimators = 200
            base_max_depth = 10
            base_min_samples_split = 15
            base_min_samples_leaf = 8
            base_confidence_threshold = 0.60
            
            # 根据市场状态调整参数
            # 1. 震荡市场 - 增加树的数量和降低置信度阈值以捕捉更多交易机会
            if market_type == "震荡市":
                self.n_estimators = int(base_n_estimators * 1.2)  # 增加树的数量
                self.max_depth = base_max_depth - 1  # 减小树的深度避免过拟合
                self.min_samples_split = base_min_samples_split + 5  # 增加分裂所需的样本数
                self.min_samples_leaf = base_min_samples_leaf + 2  # 增加叶节点最小样本数
                self.confidence_threshold = base_confidence_threshold - 0.05  # 降低置信度阈值
            
            # 2. 趋势市场 - 减少树的数量并提高置信度阈值以减少虚假信号
            elif "趋势市" in market_type:
                self.n_estimators = base_n_estimators
                self.max_depth = base_max_depth + 1  # 增加树的深度以捕捉更复杂的模式
                self.min_samples_split = base_min_samples_split - 3  # 减少分裂所需的样本数
                self.min_samples_leaf = base_min_samples_leaf - 2  # 减少叶节点最小样本数
                self.confidence_threshold = base_confidence_threshold + 0.05  # 提高置信度阈值
            
            # 3. 高波动市场 - 增加树的数量和提高置信度阈值以过滤噪音
            if volatility == "高波动":
                self.n_estimators = int(self.n_estimators * 1.1)  # 进一步增加树的数量
                self.confidence_threshold += 0.05  # 提高置信度阈值
            
            # 4. 低波动市场 - 减少树的数量和降低置信度阈值以捕捉微小信号
            elif volatility == "低波动":
                self.n_estimators = int(self.n_estimators * 0.9)  # 减少树的数量
                self.confidence_threshold -= 0.05  # 降低置信度阈值
            
            # 5. 强劲趋势 - 提高置信度阈值，专注于高质量信号
            if market_strength == "强劲":
                self.confidence_threshold += 0.05  # 提高置信度阈值
            
            # 6. 弱势趋势 - 降低置信度阈值，增加交易机会
            elif market_strength == "弱势":
                self.confidence_threshold -= 0.05  # 降低置信度阈值
            
            # 确保参数在合理范围内
            self.n_estimators = max(50, min(400, self.n_estimators))
            self.max_depth = max(5, min(15, self.max_depth))
            self.min_samples_split = max(5, min(30, self.min_samples_split))
            self.min_samples_leaf = max(2, min(15, self.min_samples_leaf))
            self.confidence_threshold = max(0.4, min(0.8, self.confidence_threshold))
            
            self.logger.info(f"优化后的随机森林参数: n_estimators={self.n_estimators}, max_depth={self.max_depth}, " +
                           f"min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}, " +
                           f"confidence_threshold={self.confidence_threshold:.2f}")
            
        except Exception as e:
            self.logger.error(f"优化模型参数失败: {str(e)}")
            
    def run(self):
        """运行策略"""
        try:
            self.logger.info(f"启动中期随机森林交易策略 (5分钟K线)")
            
            while True:
                try:
                    # 监控持仓并根据策略生成交易信号
                    self.monitor_position()
                except Exception as e:
                    self.logger.error(f"策略执行错误: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                
                # 等待下一个检查周期
                time.sleep(self.check_interval)
                
        except Exception as e:
            self.logger.error(f"策略运行失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise