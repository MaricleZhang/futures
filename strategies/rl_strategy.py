"""
强化学习交易策略 - 30分钟时间框架
基于深度Q网络(DQN)的交易决策系统
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
import os
import pickle
import random
from collections import deque
from strategies.base_strategy import BaseStrategy
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


class RLTradingStrategy30m(BaseStrategy):
    """RLTradingStrategy30m - 强化学习交易策略
    
    基于30分钟K线的强化学习策略，使用深度Q网络(DQN)自动学习
    最优交易决策，通过与市场交互不断优化策略。
    
    特点:
    1. 自主学习: 通过与环境交互自动学习最优决策策略
    2. 探索与利用: 平衡探索新策略和利用已知有效策略
    3. 延迟奖励: 能够学习长期回报而非仅关注短期收益
    4. 经验回放: 从历史交易中高效学习
    5. 实时适应: 随着市场条件变化持续调整策略
    6. 风险管理: 内置风险控制机制避免过度亏损
    """
    
    def __init__(self, trader):
        """初始化强化学习策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '30m'       # 30分钟K线
        self.check_interval = 300         # 检查信号间隔(秒)
        self.lookback_period = 600        # 计算指标所需的K线数量
        self.training_lookback = 2000     # 训练模型所需的K线数量
        
        # 强化学习参数
        self.state_size = 20              # 状态空间维度(特征数量)
        self.action_size = 3              # 动作空间大小: 0(观望), 1(买入), 2(卖出)
        self.model_path = 'models/rl_dqn_model_{}.h5'.format(trader.symbol.replace('/', '_'))
        self.scaler_path = 'models/rl_scaler_{}.pkl'.format(trader.symbol.replace('/', '_'))
        self.memory_capacity = 10000      # 经验回放容量
        self.memory = deque(maxlen=self.memory_capacity)
        
        # DQN超参数
        self.gamma = 0.95                 # 折扣因子
        self.epsilon = 1.0                # 探索率(初始值)
        self.epsilon_min = 0.01           # 最小探索率
        self.epsilon_decay = 0.995        # 探索率衰减
        self.learning_rate = 0.001        # 学习率
        self.batch_size = 32              # 批处理大小
        self.tau = 0.01                   # 目标网络更新率
        self.update_target_every = 5      # 目标网络更新频率(训练次数)
        
        # 交易参数
        self.min_confidence = 0.6         # 最小置信度阈值
        self.sequence_length = 15         # 状态序列长度
        self.reward_scaling = 100         # 奖励缩放因子
        self.position_penalty_factor = 0.2  # 持仓惩罚因子
        self.max_position_hold_time = 12 * 60  # 最大持仓时间(分钟)
        
        # 风险控制参数
        self.stop_loss_pct = 0.03         # 止损比例
        self.take_profit_pct = 0.06       # 止盈比例
        self.trailing_stop = True         # 是否启用追踪止损
        self.trailing_stop_activation = 0.02   # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.01     # 追踪止损距离百分比
        self.max_drawdown_pct = 0.05      # 最大回撤比例
        self.cooldown_period = 60         # 冷却期(分钟)
        
        # 模型和内部状态
        self.model = None                 # DQN主网络
        self.target_model = None          # DQN目标网络
        self.scaler = None                # 特征缩放器
        self.last_state = None            # 上一个状态
        self.last_action = None           # 上一个动作
        self.last_train_time = 0          # 上次训练时间
        self.total_train_steps = 0        # 总训练步数
        self.train_interval = 3600        # 训练间隔(秒)
        self.is_training_mode = True      # 是否处于训练模式
        self.performance_history = []     # 策略表现历史
        
        # 持仓状态
        self.position_entry_time = None   # 开仓时间
        self.position_entry_price = None  # 开仓价格
        self.max_profit_reached = 0       # 达到的最大利润
        self.in_cooldown = False          # 是否处于冷却期
        self.cooldown_end_time = None     # 冷却期结束时间
        
        # 初始化
        self._init_model_and_scaler()
        
    def _init_model_and_scaler(self):
        """初始化模型和缩放器"""
        # 创建模型目录
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # 初始化特征缩放器
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"加载特征缩放器: {self.scaler_path}")
        else:
            self.scaler = StandardScaler()
            self.logger.info("创建新的特征缩放器")
        
        # 初始化或加载模型
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.logger.info(f"加载现有模型: {self.model_path}")
                
                # 创建目标网络(与主网络权重相同)
                self.target_model = load_model(self.model_path)
                self.logger.info("目标网络创建成功")
                
                # 加载完模型后降低探索率，更多地利用已学习的策略
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)
                self.logger.info(f"降低探索率至 {self.epsilon:.4f}")
                
            except Exception as e:
                self.logger.error(f"加载模型失败: {str(e)}")
                self._build_model()
        else:
            self._build_model()
    
    def _build_model(self):
        """构建DQN模型"""
        try:
            # 主网络
            self.model = Sequential()
            # LSTM层用于处理序列数据
            self.model.add(LSTM(128, 
                              return_sequences=True, 
                              input_shape=(self.sequence_length, self.state_size),
                              activation='tanh'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            
            self.model.add(LSTM(64, return_sequences=False))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            
            self.model.add(Dense(32, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            
            # 输出层 - Q值预测
            self.model.add(Dense(self.action_size, activation='linear'))
            
            # 编译模型
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            self.logger.info("主网络构建完成")
            
            # 创建目标网络(与主网络结构相同)
            self.target_model = Sequential.from_config(self.model.get_config())
            self.target_model.set_weights(self.model.get_weights())
            self.logger.info("目标网络构建完成")
            
            return True
        except Exception as e:
            self.logger.error(f"构建模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _prepare_features(self, klines):
        """
        准备特征数据
        
        Args:
            klines (list): K线数据
            
        Returns:
            pandas.DataFrame: 特征数据
        """
        try:
            # 转换K线数据到DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算技术指标特征
            
            # 1. 价格特征
            df['close_diff'] = df['close'].pct_change()
            df['high_diff'] = df['high'].pct_change()
            df['low_diff'] = df['low'].pct_change()
            df['close_diff_5'] = df['close'].pct_change(periods=5)
            df['close_diff_10'] = df['close'].pct_change(periods=10)
            
            # 2. 移动平均线
            df['ma5'] = talib.MA(df['close'].values, timeperiod=5)
            df['ma10'] = talib.MA(df['close'].values, timeperiod=10)
            df['ma20'] = talib.MA(df['close'].values, timeperiod=20)
            df['ma50'] = talib.MA(df['close'].values, timeperiod=50)
            
            # 移动平均线的相对位置
            df['ma5_10'] = df['ma5'] / df['ma10'] - 1
            df['ma10_20'] = df['ma10'] / df['ma20'] - 1
            df['ma20_50'] = df['ma20'] / df['ma50'] - 1
            
            # 3. MACD
            macd, macdsignal, macdhist = talib.MACD(
                df['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist
            
            # 4. RSI
            df['rsi6'] = talib.RSI(df['close'].values, timeperiod=6)
            df['rsi14'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # 5. 布林带
            upperband, middleband, lowerband = talib.BBANDS(
                df['close'].values, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            df['bb_upper'] = upperband
            df['bb_middle'] = middleband
            df['bb_lower'] = lowerband
            df['bb_width'] = (upperband - lowerband) / middleband
            df['bb_pos'] = (df['close'] - lowerband) / (upperband - lowerband)
            
            # 6. ATR - 波动率
            df['atr'] = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=14
            )
            df['atr_ratio'] = df['atr'] / df['close']
            
            # 7. 成交量特征
            df['volume_diff'] = df['volume'].pct_change()
            df['volume_ma5'] = talib.MA(df['volume'].values, timeperiod=5)
            df['volume_ma10'] = talib.MA(df['volume'].values, timeperiod=10)
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
            
            # 8. ADX - 趋势强度
            df['adx'] = talib.ADX(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=14
            )
            
            # 9. OBV - 能量潮
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            df['obv_diff'] = df['obv'].pct_change()
            
            # 10. 随机指标
            df['slowk'], df['slowd'] = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
            # 删除NaN值
            df = df.dropna()
            
            return df
        
        except Exception as e:
            self.logger.error(f"准备特征数据失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_state(self, df, add_position=True):
        """
        获取当前状态表示
        
        Args:
            df (pandas.DataFrame): 特征数据
            add_position (bool): 是否添加持仓信息到状态
            
        Returns:
            numpy.ndarray: 状态向量
        """
        try:
            if len(df) < self.sequence_length:
                self.logger.error(f"数据长度不足: {len(df)} < {self.sequence_length}")
                return None
            
            # 选择最近的数据
            df_recent = df.iloc[-self.sequence_length:].copy()
            
            # 选择特征列
            features = [
                'close_diff', 'close_diff_5', 'close_diff_10',
                'ma5_10', 'ma10_20', 'ma20_50',
                'macd', 'macdhist',
                'rsi6', 'rsi14',
                'bb_width', 'bb_pos',
                'atr_ratio',
                'volume_ratio', 'obv_diff',
                'adx',
                'slowk', 'slowd'
            ]
            
            # 检查特征是否存在
            missing_features = [f for f in features if f not in df_recent.columns]
            if missing_features:
                self.logger.error(f"缺少特征: {missing_features}")
                return None
            
            # 创建特征矩阵
            X = df_recent[features].values
            
            # 添加持仓信息
            if add_position:
                try:
                    position = self.trader.get_position()
                    if position and 'info' in position:
                        position_amt = float(position['info'].get('positionAmt', 0))
                        # 归一化持仓数量 (-1 到 1)
                        norm_pos = np.tanh(position_amt / 10)  # 使用tanh限制在-1到1之间
                        
                        # 扩展特征矩阵添加持仓信息
                        position_feature = np.ones((len(X), 2)) * np.array([norm_pos, int(position_amt != 0)])
                        X = np.hstack((X, position_feature))
                    else:
                        # 如果没有持仓，添加零向量
                        X = np.hstack((X, np.zeros((len(X), 2))))
                except Exception as e:
                    self.logger.error(f"获取持仓信息失败: {str(e)}")
                    # 添加零向量
                    X = np.hstack((X, np.zeros((len(X), 2))))
            
            # 特征缩放
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(X)
                
                # 保存缩放器
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                # 检查scaler是否已经被fit过
                try:
                    # 尝试获取均值属性，如果不存在则说明未fit
                    getattr(self.scaler, 'mean_')
                    
                    # 检查特征数量是否匹配
                    if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != X.shape[1]:
                        self.logger.warning(f"特征数量不匹配: 当前{X.shape[1]}个特征, scaler期望{self.scaler.n_features_in_}个特征, 重新训练scaler")
                        self.scaler = StandardScaler()
                        self.scaler.fit(X)
                        
                        # 保存更新后的缩放器
                        with open(self.scaler_path, 'wb') as f:
                            pickle.dump(self.scaler, f)
                except AttributeError:
                    # 如果未fit，则进行fit
                    self.logger.info("Scaler未经过训练，正在进行fit操作")
                    self.scaler.fit(X)
                    
                    # 保存更新后的缩放器
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    
            X_scaled = self.scaler.transform(X)
            
            return X_scaled
        
        except Exception as e:
            self.logger.error(f"获取状态失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_reward(self, action, current_price, next_price, position=None):
        """
        计算强化学习奖励
        
        Args:
            action (int): 执行的动作
            current_price (float): 当前价格
            next_price (float): 下一时刻价格
            position (dict): 持仓信息
            
        Returns:
            float: 奖励值
        """
        try:
            # 获取持仓信息
            if position is None:
                position = self.trader.get_position()
                
            # 初始化奖励
            reward = 0
            
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                
                # 计算价格变化
                price_change = (next_price - current_price) / current_price
                
                # 计算持仓的即时回报
                if position_amt > 0:  # 多头持仓
                    # 持仓盈亏变化
                    pnl_change = price_change
                    
                    # 根据动作调整奖励
                    if action == 1:  # 继续买入/持有多头
                        reward = pnl_change * self.reward_scaling
                    elif action == 2:  # 卖出(平仓)
                        # 如果卖出时实际盈利，给予额外奖励
                        actual_pnl = (next_price - entry_price) / entry_price
                        if actual_pnl > 0:
                            reward = actual_pnl * self.reward_scaling * 1.5  # 额外奖励
                        else:
                            reward = actual_pnl * self.reward_scaling
                    else:  # 观望
                        reward = pnl_change * self.reward_scaling * 0.5  # 减少奖励
                        
                elif position_amt < 0:  # 空头持仓
                    # 空头持仓盈亏是反向的
                    pnl_change = -price_change
                    
                    # 根据动作调整奖励
                    if action == 2:  # 继续卖出/持有空头
                        reward = pnl_change * self.reward_scaling
                    elif action == 1:  # 买入(平仓)
                        # 如果买入时实际盈利，给予额外奖励
                        actual_pnl = (entry_price - next_price) / entry_price
                        if actual_pnl > 0:
                            reward = actual_pnl * self.reward_scaling * 1.5  # 额外奖励
                        else:
                            reward = actual_pnl * self.reward_scaling
                    else:  # 观望
                        reward = pnl_change * self.reward_scaling * 0.5  # 减少奖励
                
                # 持仓时间惩罚 - 持仓时间越长，惩罚越大
                if self.position_entry_time is not None:
                    holding_time_minutes = (time.time() - self.position_entry_time) / 60
                    time_penalty = min(0.5, holding_time_minutes / self.max_position_hold_time) * self.position_penalty_factor
                    reward -= time_penalty
                
            else:  # 无持仓
                # 无持仓时，根据动作和价格变化给予奖励
                price_change = (next_price - current_price) / current_price
                
                if action == 1:  # 买入
                    reward = price_change * self.reward_scaling  # 如果价格上涨则为正奖励
                elif action == 2:  # 卖出
                    reward = -price_change * self.reward_scaling  # 如果价格下跌则为正奖励
                else:  # 观望
                    # 无持仓观望的奖励较小
                    reward = 0.01  # 小的正奖励，鼓励在不确定时观望
            
            # 限制奖励范围，避免极端值
            reward = np.clip(reward, -10, 10)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"计算奖励失败: {str(e)}")
            return 0
    
    def _remember(self, state, action, reward, next_state, done):
        """
        将经验添加到回放内存
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        if state is not None and next_state is not None:
            self.memory.append((state, action, reward, next_state, done))
    
    def _replay(self, batch_size=None):
        """
        从经验回放中学习
        
        Args:
            batch_size: 批量大小
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # 如果内存中的样本不足，不进行训练
        if len(self.memory) < batch_size:
            return
        
        try:
            # 从记忆中随机抽取批次样本
            minibatch = random.sample(self.memory, batch_size)
            
            for state, action, reward, next_state, done in minibatch:
                # 当前Q值预测
                target = self.model.predict(state, verbose=0)[0]
                
                if done:
                    # 如果是结束状态，只考虑立即奖励
                    target[action] = reward
                else:
                    # 使用目标网络预测下一状态的最大Q值
                    t = self.target_model.predict(next_state, verbose=0)[0]
                    # Q值更新公式: Q(s,a) = r + γ * max(Q(s',a'))
                    target[action] = reward + self.gamma * np.amax(t)
                
                # 使用单样本进行训练
                self.model.fit(state, np.array([target]), epochs=1, verbose=0)
            
            # 更新目标网络
            self.total_train_steps += 1
            if self.total_train_steps % self.update_target_every == 0:
                # 使用软更新策略
                target_weights = self.target_model.get_weights()
                model_weights = self.model.get_weights()
                updated_weights = []
                
                for i in range(len(target_weights)):
                    updated_weights.append(
                        (1 - self.tau) * target_weights[i] + self.tau * model_weights[i]
                    )
                    
                self.target_model.set_weights(updated_weights)
                self.logger.info(f"目标网络更新完成 (步骤 {self.total_train_steps})")
        
        except Exception as e:
            self.logger.error(f"经验回放训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _save_model(self):
        """保存模型和缩放器"""
        if self.model is not None:
            try:
                self.model.save(self.model_path)
                self.logger.info(f"模型保存成功: {self.model_path}")
                
                # 保存缩放器
                if self.scaler is not None:
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    self.logger.info(f"缩放器保存成功: {self.scaler_path}")
                    
                return True
            except Exception as e:
                self.logger.error(f"保存模型失败: {str(e)}")
                return False
        return False
    
    def _act(self, state, explore=True):
        """
        根据状态选择动作
        
        Args:
            state: 当前状态
            explore: 是否启用探索
            
        Returns:
            tuple: (选择的动作, Q值, 是否是探索行为)
        """
        is_explore = False
        
        # 检查是否处于冷却期
        if self.in_cooldown:
            current_time = time.time()
            if current_time < self.cooldown_end_time:
                self.logger.info("处于冷却期，选择观望")
                return 0, [0, 0, 0], False
            else:
                self.in_cooldown = False
                self.logger.info("冷却期结束")
        
        # 探索: 随机选择动作
        if explore and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            is_explore = True
            q_values = [0, 0, 0]  # 随机探索时没有实际Q值
        else:
            # 利用: 选择Q值最高的动作
            q_values = self.model.predict(state, verbose=0)[0]
            action = np.argmax(q_values)
            
            # 检查是否有足够的置信度
            if max(q_values) < self.min_confidence and explore:
                # 如果置信度不足且允许探索，选择观望
                action = 0
                self.logger.info(f"置信度不足 ({max(q_values):.4f} < {self.min_confidence}), 选择观望")
        
        return action, q_values, is_explore
    
    def _update_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, klines=None):
        """
        训练模型
        
        Args:
            klines: K线数据，如果为None则获取历史数据
            
        Returns:
            bool: 训练是否成功
        """
        try:
            # 检查模型是否初始化
            if self.model is None:
                self._build_model()
                
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    interval=self.kline_interval,
                    limit=self.training_lookback
                )
                
            if klines is None or len(klines) < 100:
                self.logger.error("训练数据不足")
                return False
            
            # 准备特征数据
            df = self._prepare_features(klines)
            if df is None:
                return False
            
            # 获取历史数据的状态表示
            states = []
            for i in range(self.sequence_length, len(df) - 1):
                # 截取当前时间窗口的数据
                window_df = df.iloc[i-self.sequence_length:i]
                state = self._get_state(window_df, add_position=False)
                
                if state is not None:
                    # 扩展维度以适应模型输入
                    state = np.reshape(state, [1, self.sequence_length, state.shape[1]])
                    states.append(state)
            
            # 没有足够的状态样本
            if len(states) < 100:
                self.logger.error("状态样本不足，无法训练")
                return False
            
            # 模拟历史交易进行训练
            self.logger.info(f"开始模拟历史交易训练，样本数量: {len(states) - 1}")
            
            for i in range(len(states) - 1):
                state = states[i]
                next_state = states[i+1]
                
                # 使用当前策略选择动作
                action, _, is_explore = self._act(state, explore=True)
                
                # 获取当前和下一时刻的价格
                current_price = df.iloc[i + self.sequence_length]['close']
                next_price = df.iloc[i + 1 + self.sequence_length]['close']
                
                # 计算奖励
                reward = self._get_reward(action, current_price, next_price)
                
                # 判断是否是结束状态
                done = (i == len(states) - 2)
                
                # 记录经验
                self._remember(state, action, reward, next_state, done)
                
                # 如果记忆足够多，进行批量训练
                if len(self.memory) >= self.batch_size:
                    self._replay()
                
                # 更新探索率
                self._update_epsilon()
            
            # 训练完成后保存模型
            self._save_model()
            
            # 更新训练时间
            self.last_train_time = time.time()
            
            self.logger.info(f"模型训练完成，当前探索率: {self.epsilon:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def should_train(self):
        """检查是否应该进行训练"""
        current_time = time.time()
        if current_time - self.last_train_time > self.train_interval:
            return True
        return False
    
    def generate_signal(self, klines):
        """
        生成交易信号
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 检查模型是否初始化
            if self.model is None or self.target_model is None:
                self.logger.error("模型未初始化")
                return 0
            
            # 检查是否需要训练
            if self.should_train() and self.is_training_mode:
                self.logger.info("开始定期训练...")
                self.train(klines)
            
            # 准备特征数据
            df = self._prepare_features(klines)
            if df is None:
                return 0
            
            # 获取当前状态
            state_matrix = self._get_state(df)
            if state_matrix is None:
                return 0
            
            # 重塑状态矩阵以适应模型输入
            state = np.reshape(state_matrix, [1, self.sequence_length, state_matrix.shape[1]])
            
            # 选择动作
            action, q_values, is_explore = self._act(state, explore=self.is_training_mode)
            
            # 记录当前状态和动作
            self.last_state = state
            self.last_action = action
            
            # 将DQN动作映射到交易信号
            signal_mapping = {0: 0, 1: 1, 2: -1}  # 0:观望, 1:买入, 2:卖出
            signal = signal_mapping[action]
            
            # 记录交易决策
            if is_explore:
                self.logger.info(f"探索行为: 执行动作 {action} (信号 {signal})")
            else:
                action_names = ["观望", "买入", "卖出"]
                q_values_str = ", ".join([f"{q:.4f}" for q in q_values])
                self.logger.info(f"交易决策: {action_names[action]} (Q值: [{q_values_str}])")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 在冷却期内不进行交易
            if self.in_cooldown:
                current_time = time.time()
                if current_time < self.cooldown_end_time:
                    return
                else:
                    self.in_cooldown = False
                    self.logger.info("冷却期结束")
            
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
                        
                        # 进入冷却期
                        self.in_cooldown = True
                        self.cooldown_end_time = current_time + self.cooldown_period * 60
                        self.logger.info(f"进入冷却期 {self.cooldown_period} 分钟")
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
                    
                    # 如果是学习模式，记录正向奖励
                    if self.is_training_mode and self.last_state is not None:
                        # 获取最新K线数据
                        klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                        df = self._prepare_features(klines)
                        if df is not None:
                            # 获取新状态
                            next_state_matrix = self._get_state(df)
                            if next_state_matrix is not None:
                                next_state = np.reshape(next_state_matrix, [1, self.sequence_length, next_state_matrix.shape[1]])
                                # 给予高额奖励
                                reward = profit_rate * self.reward_scaling * 2
                                # 记录经验
                                self._remember(self.last_state, self.last_action, reward, next_state, True)
                    
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    
                    # 如果是学习模式，记录负向奖励
                    if self.is_training_mode and self.last_state is not None:
                        # 获取最新K线数据
                        klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                        df = self._prepare_features(klines)
                        if df is not None:
                            # 获取新状态
                            next_state_matrix = self._get_state(df)
                            if next_state_matrix is not None:
                                next_state = np.reshape(next_state_matrix, [1, self.sequence_length, next_state_matrix.shape[1]])
                                # 给予负向奖励
                                reward = -5  # 固定的负向奖励
                                # 记录经验
                                self._remember(self.last_state, self.last_action, reward, next_state, True)
                    
                    # 进入冷却期
                    self.in_cooldown = True
                    self.cooldown_end_time = current_time + self.cooldown_period * 60
                    self.logger.info(f"进入冷却期 {self.cooldown_period} 分钟")
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
                
                # 检查是否需要反向操作
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                signal = self.generate_signal(klines)
                
                # 如果信号与当前持仓方向相反，平仓
                if (position_side == "多" and signal == -1) or (position_side == "空" and signal == 1):
                    self.logger.info(f"信号方向与当前持仓相反，平仓")
                    self.trader.close_position()
                    return
            
            # 如果是学习模式，且存在上一个状态和动作，记录经验
            if self.is_training_mode and self.last_state is not None and self.last_action is not None:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                df = self._prepare_features(klines)
                
                if df is not None and len(df) > self.sequence_length:
                    # 获取当前状态
                    current_price = df.iloc[-1]['close']
                    
                    # 获取上一个状态所对应的价格
                    past_price = df.iloc[-2]['close'] if len(df) > self.sequence_length + 1 else current_price
                    
                    # 计算奖励
                    reward = self._get_reward(self.last_action, past_price, current_price, position)
                    
                    # 获取新状态
                    next_state_matrix = self._get_state(df)
                    if next_state_matrix is not None:
                        next_state = np.reshape(next_state_matrix, [1, self.sequence_length, next_state_matrix.shape[1]])
                        
                        # 记录经验
                        self._remember(self.last_state, self.last_action, reward, next_state, False)
                        
                        # 如果内存中的样本足够多，进行训练
                        if len(self.memory) >= self.batch_size * 4:
                            self._replay()
                            self._update_epsilon()
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def run(self):
        """运行策略"""
        self.logger.info("启动强化学习交易策略")
        
        # 首次训练
        if self.is_training_mode and (self.model is None or self.should_train()):
            self.logger.info("开始初始模型训练...")
            # 获取历史K线数据
            klines = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            self.train(klines)
        
        # 运行父类的run方法
        super().run()
