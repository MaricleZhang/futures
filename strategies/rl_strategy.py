import numpy as np
import pandas as pd
import talib
import time
import logging
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random
import config
from strategies.base_strategy import BaseStrategy

class RLTrendStrategy(BaseStrategy):
    """RLTrendStrategy - 强化学习趋势跟踪策略
    
    基于深度强化学习(DQN)的交易策略，使用3分钟K线数据进行市场分析和交易决策。
    通过状态表示、动作选择和奖励机制来训练代理进行最优交易决策。
    
    特点:
    1. 深度Q网络(DQN): 使用深度学习模型学习价格走势特征
    2. 经验回放: 通过存储和回放历史交易数据提高学习效率
    3. ε-贪婪策略: 在探索新策略和利用已知策略间平衡
    4. 复合奖励函数: 同时考虑收益率、风险控制和交易成本
    5. 动态适应: 在线学习能力，随市场变化而调整
    """
    
    MODEL_NAME = "RLTrend"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化强化学习趋势跟踪策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '3m'      # 3分钟K线
        self.check_interval = 60        # 检查信号间隔(秒)
        self.lookback_period = 100      # 计算指标所需的K线数量
        self.training_lookback = 500    # 训练所需K线数量
        
        # RL模型参数
        self.state_size = 30           # 状态特征维度
        self.action_size = 3           # 动作空间大小: [卖出, 观望, 买入]
        self.batch_size = 32           # 经验回放批次大小
        self.gamma = 0.95              # 奖励折扣因子
        self.epsilon = 1.0             # 探索率初始值
        self.epsilon_min = 0.01        # 最小探索率
        self.epsilon_decay = 0.995     # 探索率衰减速度
        self.learning_rate = 0.001     # 学习率
        self.update_target_freq = 10   # 目标网络更新频率
        self.memory_size = 2000        # 经验回放缓冲区大小
        
        # 风险控制参数
        self.max_position_hold_time = 120  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.5     # 目标利润率 50%
        self.stop_loss_pct = 0.1         # 止损率 10%
        self.max_trades_per_hour = 4       # 每小时最大交易次数
        
        # 交易状态
        self.trade_count_hour = 0       # 当前小时交易次数
        self.last_trade_hour = None     # 上次交易的小时
        self.position_entry_time = None # 开仓时间
        self.position_entry_price = None # 开仓价格
        self.last_action = 1            # 上一次动作 (初始为观望)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=self.memory_size)
        
        # 模型
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.online_model.get_weights())
        self.target_updates = 0
        self.step_count = 0
        
        # 模型存储路径
        self.model_dir = "models"
        self.model_path = f"{self.model_dir}/rl_trend_strategy_{self.trader.symbol}.h5"
        self.metadata_path = f"{self.model_dir}/model_metadata_{self.trader.symbol}.json"
        
        # 性能跟踪
        self.cum_reward = 0
        self.trades_history = []
        self.trade_id = 0
        
        self.last_training_time = time.time()
        self.retraining_interval = 3600  # 1小时重新训练一次

        # 尝试加载模型
        model_loaded = self._load_model()
    
        # 如果没有成功加载模型，则执行初始训练
        if not model_loaded:
            self.logger.info("模型加载失败，执行初始训练...")
            initial_klines = self.trader.get_klines(interval=self.kline_interval, limit=1000)
            if len(initial_klines) > 500:
                for _ in range(10):
                    self.train_model(initial_klines)
                    self._save_model()  # 保存训练好的模型
            else:
               self.logger.info("成功加载保存的模型，可以直接使用")
    
    def _build_model(self):
        """构建深度Q学习网络"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _load_model(self):
        try:
            import os
            import json
            
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
            # 加载模型
            if os.path.exists(self.model_path):
                self.online_model = load_model(self.model_path)
                self.target_model.set_weights(self.online_model.get_weights())
                self.logger.info(f"已加载保存的模型: {self.model_path}")
                
                # 尝试加载元数据
                metadata_path = self.metadata_path
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # 恢复探索率
                    if 'epsilon' in metadata:
                        self.epsilon = metadata['epsilon']
                        self.logger.info(f"已恢复探索率: {self.epsilon:.4f}")
                    else:
                        # 如果没有保存过探索率，使用默认值但略低
                        self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)
                else:
                    self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)
                return True
            else:
                self.logger.info("未找到保存的模型，将使用新模型")
                return False
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    def _save_model(self):
        try:
            # 保存模型
            self.online_model.save(self.model_path)
            
            # 保存探索率和其他超参数
            metadata = {
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'last_update': time.time()
            }
            
            # 保存到单独的文件
            import json
            with open(f"{self.model_dir}/model_metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            self.logger.info(f"模型和元数据已保存，当前探索率: {self.epsilon:.4f}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
    
    def _get_state(self, klines):
        """从K线数据提取当前状态特征"""
        try:
            if not isinstance(klines, list) or len(klines) < 30:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 准备特征
            features = []
            
            # 1. 价格相对特征 (6个，这里增加了一个特征以达到总共30个)
            close = df['close'].values
            close_prev = np.roll(close, 1)
            close_prev[0] = close[0]
            
            # 价格相对于开盘价
            features.append((df['close'].iloc[-1] / df['open'].iloc[-1]) - 1)
            
            # 最近5根K线的价格变化率
            for i in range(1, 6):
                if len(close) > i:
                    features.append((close[-1] / close[-i-1]) - 1)
                else:
                    features.append(0)
            
            # 2. 技术指标 (15个)
            # RSI - 3个不同周期
            for period in [7, 14, 21]:
                rsi = talib.RSI(close, timeperiod=period)
                features.append(rsi[-1] / 100)  # 归一化到0-1范围
            
            # 移动平均 - 3个不同周期
            for period in [5, 10, 20]:
                sma = talib.SMA(close, timeperiod=period)
                features.append(close[-1] / sma[-1] - 1)  # 当前价格与MA的偏离度
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            features.append(macd[-1] / close[-1])  # 归一化MACD
            features.append(macd_signal[-1] / close[-1])  # 归一化信号线
            features.append(macd_hist[-1] / close[-1])  # 归一化直方图
            
            # 布林带
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features.append((close[-1] - lower[-1]) / (upper[-1] - lower[-1]))  # 价格在布林带中的位置
            features.append((upper[-1] - lower[-1]) / middle[-1])  # 布林带宽度
            
            # ATR - 波动率
            atr = talib.ATR(df['high'].values, df['low'].values, close, timeperiod=14)
            features.append(atr[-1] / close[-1])  # 相对ATR
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, close, 
                                      fastk_period=14, slowk_period=3, slowk_matype=0, 
                                      slowd_period=3, slowd_matype=0)
            features.append(slowk[-1] / 100)  # 归一化到0-1范围
            features.append(slowd[-1] / 100)  # 归一化到0-1范围
            
            # 3. 成交量特征 (5个)
            volume = df['volume'].values
            
            # 成交量变化
            vol_ma = talib.SMA(volume, timeperiod=10)
            features.append(volume[-1] / vol_ma[-1])  # 相对于MA的成交量
            
            # 成交量趋势 - 最近几根K线的成交量变化
            for i in range(1, 5):
                if len(volume) > i:
                    features.append(volume[-1] / volume[-i-1]) 
                else:
                    features.append(1.0)
            
            # 4. 形态识别特征 (2个)
            # 使用形态识别模式，检测潜在反转模式
            doji = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, close)
            features.append(1 if doji[-1] != 0 else 0)
            
            hammer = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, close)
            features.append(1 if hammer[-1] != 0 else 0)
            
            # 5. 当前持仓相关特征 (3个)
            # 获取当前持仓状态
            position = self.trader.get_position()
            
            # 持仓方向和数量
            position_amt = 0
            entry_price = 0
            holding_time = 0
            
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    entry_price = float(position['info'].get('entryPrice', 0))
                    # 计算持仓时间比例 (相对于最大持仓时间)
                    if self.position_entry_time:
                        holding_time = (time.time() - self.position_entry_time) / 60  # 转为分钟
                        holding_time = min(holding_time / self.max_position_hold_time, 1.0)  # 归一化
            
            # 添加持仓特征
            features.append(1 if position_amt > 0 else (-1 if position_amt < 0 else 0))  # 持仓方向
            
            # 如果有持仓，计算当前的盈亏率
            if abs(position_amt) > 0:
                if position_amt > 0:  # 多仓
                    profit_pct = (close[-1] - entry_price) / entry_price
                else:  # 空仓
                    profit_pct = (entry_price - close[-1]) / entry_price
                features.append(profit_pct)  # 未实现盈亏率
            else:
                features.append(0)  # 无持仓，盈亏为0
                
            # 持仓时间比例
            features.append(holding_time)
            
            # 确保维度正确
            features = np.array(features)
            
            # 验证特征数量
            if len(features) != self.state_size:
                self.logger.error(f"特征维度不匹配! 预期 {self.state_size}, 实际 {len(features)}")
                return None
                
            return features
            
        except Exception as e:
            self.logger.error(f"提取状态特征失败: {str(e)}")
            return None
    
    def _get_reward(self, action, prev_state, next_state, current_price, next_price):
        """计算给定动作的奖励"""
        try:
            # 获取持仓状态
            position = self.trader.get_position()
            position_amt = 0
            entry_price = 0
            
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    entry_price = float(position['info'].get('entryPrice', 0))
            
            # 1. 基础奖励 - 来自价格变化的收益
            price_change_pct = (next_price - current_price) / current_price
            
            # 2. 动作奖励
            action_reward = 0
            
            # 持仓奖励计算
            if abs(position_amt) > 0:  # 有持仓
                if position_amt > 0:  # 多仓
                    position_reward = price_change_pct * 100  # 放大奖励
                else:  # 空仓
                    position_reward = -price_change_pct * 100  # 价格下跌时有正奖励
                
                # 根据持仓方向和当前动作给予额外奖励/惩罚
                if (position_amt > 0 and action == 2) or (position_amt < 0 and action == 0):
                    # 持有方向正确的仓位
                    action_reward += 0.1
                elif (position_amt > 0 and action == 0) or (position_amt < 0 and action == 2):
                    # 应该平仓或反向操作
                    action_reward -= 0.1
            else:  # 无持仓
                position_reward = 0
                
                # 观望奖励 - 根据价格趋势给予观望的奖励
                if action == 1:  # 观望
                    if abs(price_change_pct) < 0.0005:  # 价格变化很小时，观望是好的
                        action_reward += 0.05
            
            # 3. 风险控制奖励
            risk_reward = 0
            
            # 止盈止损奖励
            if abs(position_amt) > 0:
                current_profit_pct = 0
                
                if position_amt > 0:  # 多仓
                    current_profit_pct = (current_price - entry_price) / entry_price
                else:  # 空仓
                    current_profit_pct = (entry_price - current_price) / entry_price
                
                # 止盈奖励
                if current_profit_pct >= self.profit_target_pct and action != 1:  # 非观望动作
                    risk_reward += 0.5
                
                # 止损惩罚
                if current_profit_pct <= -self.stop_loss_pct and action != 1:  # 非观望动作
                    risk_reward += 0.3
                elif current_profit_pct <= -self.stop_loss_pct * 2 and action == 1:  # 观望动作
                    risk_reward -= 0.5  # 严重亏损还不平仓的惩罚
            
            # 4. 交易频率控制
            freq_reward = 0
            
            # 过度交易惩罚
            if self.last_action != action and action != 1:  # 如果动作改变了且不是观望
                freq_reward -= 0.05  # 小惩罚以减少频繁交易
            
            # 长时间持仓惩罚
            if abs(position_amt) > 0 and self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60  # 转为分钟
                if holding_time > self.max_position_hold_time * 0.8:
                    freq_reward -= 0.1  # 接近最大持仓时间时给予惩罚
            
            # 5. 市场状态匹配度奖励
            state_reward = 0
            
            # 根据RSI指标添加市场状态匹配度奖励
            if prev_state is not None:
                rsi_value = prev_state[2] * 100  # 第3个特征是RSI(14)，还原为0-100范围
                
                if rsi_value > 70 and action == 0:  # 超买时卖出
                    state_reward += 0.2
                elif rsi_value < 30 and action == 2:  # 超卖时买入
                    state_reward += 0.2
                elif rsi_value > 70 and action == 2:  # 超买时买入
                    state_reward -= 0.2
                elif rsi_value < 30 and action == 0:  # 超卖时卖出
                    state_reward -= 0.2
            
            # 综合奖励计算
            total_reward = position_reward + action_reward + risk_reward + freq_reward + state_reward
            
            self.logger.debug(f"""
                奖励组成:
                - 持仓奖励: {position_reward:.4f}
                - 动作奖励: {action_reward:.4f}
                - 风险奖励: {risk_reward:.4f}
                - 频率奖励: {freq_reward:.4f}
                - 状态奖励: {state_reward:.4f}
                = 总奖励: {total_reward:.4f}
            """)
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"计算奖励失败: {str(e)}")
            return 0
    
    def _remember(self, state, action, reward, next_state, done):
        """将经验添加到回放缓冲区"""
        try:
            if state is not None and next_state is not None:
                self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            self.logger.error(f"保存经验失败: {str(e)}")
    
    def _replay(self, batch_size=None):
        """从经验回放缓冲区中训练模型"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        # 更新探索率后记录日志
        if self.epsilon > self.epsilon_min:
            old_epsilon = self.epsilon
            self.epsilon *= self.epsilon_decay
            self.logger.info(f"探索率更新: {old_epsilon:.6f} -> {self.epsilon:.6f}")
            
        try:
            minibatch = random.sample(self.memory, batch_size)
            
            states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])
            
            # 计算目标Q值
            target = self.online_model.predict(states, verbose=0)
            target_next = self.target_model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target[i, actions[i]] = rewards[i]
                else:
                    target[i, actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
            
            # 训练模型
            self.online_model.fit(states, target, epochs=1, verbose=0)
            
            # 更新探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # 增加步数计数器
            self.step_count += 1
            
            # 定期更新目标网络
            if self.step_count % self.update_target_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())
                self.target_updates += 1
                
                # 每更新10次目标网络保存一次模型
                if self.target_updates % 10 == 0:
                    self._save_model()
                    
        except Exception as e:
            self.logger.error(f"经验回放训练失败: {str(e)}")
    
    def generate_signal(self, klines):
        """生成交易信号
        
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            if len(klines) < self.lookback_period:
                self.logger.error(f"K线数据不足，需要至少 {self.lookback_period} 根")
                return 0
                
            # 获取当前状态
            state = self._get_state(klines)
            if state is None:
                return 0
                
            # 获取当前价格
            current_price = self.trader.get_market_price()
            
            # ε-贪婪策略选择动作
            if np.random.rand() <= self.epsilon:
                # 探索 - 随机选择动作
                action = np.random.randint(0, self.action_size)
                self.logger.info(f"探索模式: 随机选择动作 {action} (探索率 ε={self.epsilon:.4f})")
            else:
                # 利用 - 选择Q值最大的动作
                act_values = self.online_model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(act_values[0])
                self.logger.info(f"利用模式: 选择Q值最大的动作 {action} (Q值: {act_values[0]})")
            
            # 映射动作到信号
            # 动作空间: [0=卖出, 1=观望, 2=买入]
            # 信号映射: [0->-1, 1->0, 2->1]
            signal_mapping = {0: -1, 1: 0, 2: 1}
            signal = signal_mapping[action]
            
            # 记录这次选择的动作
            self.last_action = action
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0
    
    def update_model(self, prev_state, action, next_state, reward, done):
        """更新RL模型"""
        try:
            if prev_state is not None and next_state is not None:
                # 记录经验
                self._remember(prev_state, action, reward, next_state, done)
                
                # 进行经验回放训练
                self._replay()
                
                # 累计奖励
                self.cum_reward += reward
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
    
    def should_retrain(self):
        """检查是否需要重新训练"""
        current_time = time.time()
        if current_time - self.last_training_time >= self.retraining_interval:
            self.last_training_time = current_time
            return True
        return False
    
    def train_model(self, klines):
        """使用历史K线数据训练模型"""
        try:
            if not isinstance(klines, list) or len(klines) < 100:
                self.logger.error("K线数据不足，无法训练模型")
                return False
                
            self.logger.info(f"使用 {len(klines)} 根K线数据训练RL模型")
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 获取价格序列
            prices = df['close'].values
            
            # 训练轮次
            batch_size = min(32, len(self.memory) // 2) if len(self.memory) > 0 else 32
            episodes = min(200, len(klines) - self.lookback_period)
            
            self.logger.info(f"开始训练，批次大小: {batch_size}, 训练轮次: {episodes}")
            
            # 开始训练循环
            for episode in range(episodes):
                # 随机选择起始点
                start_idx = np.random.randint(self.lookback_period, len(klines) - 10)
                
                # 获取起始状态
                episode_klines = klines[start_idx - self.lookback_period:start_idx]
                state = self._get_state(episode_klines)
                
                if state is None:
                    continue
                
                # 模拟几个时间步
                for t in range(5):  # 每个episode模拟5个时间步
                    # ε-贪婪策略选择动作
                    if np.random.rand() <= self.epsilon:
                        action = np.random.randint(0, self.action_size)
                    else:
                        act_values = self.online_model.predict(state.reshape(1, -1), verbose=0)
                        action = np.argmax(act_values[0])
                    
                    # 模拟执行动作，获取下一状态
                    next_idx = start_idx + t + 1
                    if next_idx >= len(klines):
                        break
                        
                    next_episode_klines = klines[next_idx - self.lookback_period:next_idx]
                    next_state = self._get_state(next_episode_klines)
                    
                    if next_state is None:
                        break
                    
                    # 计算奖励
                    current_price = float(klines[start_idx + t][4])  # 当前收盘价
                    next_price = float(klines[next_idx][4])          # 下一收盘价
                    
                    # 模拟完成标志
                    done = (t == 4) or (next_idx >= len(klines) - 1)
                    
                    # 计算奖励
                    reward = self._get_reward(action, state, next_state, current_price, next_price)
                    
                    # 保存经验
                    self._remember(state, action, reward, next_state, done)
                    
                    # 更新状态
                    state = next_state
                    
                    if done:
                        break
                
                # 经验回放训练
                if len(self.memory) >= batch_size:
                    self._replay(batch_size)
            
            # 训练后保存模型
            self._save_model()
            
            self.logger.info(f"模型训练完成，当前探索率: {self.epsilon:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            return False
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 获取最新K线数据
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
            if len(klines) < self.lookback_period:
                self.logger.error(f"K线数据不足，需要至少 {self.lookback_period} 根")
                return
            
            # 提取当前状态
            current_state = self._get_state(klines)
            if current_state is None:
                return
            
            # 获取当前市场价格
            current_price = self.trader.get_market_price()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 如果需要重新训练模型
                if self.should_retrain():
                    training_klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
                    if self.train_model(training_klines):
                        self.logger.info("RL模型重新训练完成")
                
                # 生成交易信号
                signal = self.generate_signal(klines)
                
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
                    
                    # 记录开仓前状态
                    prev_state = current_state
                    
                    # 开多仓
                    order = self.trader.open_long(amount=trade_amount)
                    if order:
                        self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        
                        # 记录交易历史
                        self.trade_id += 1
                        self.trades_history.append({
                            'id': self.trade_id,
                            'type': 'LONG',
                            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_price': current_price,
                            'amount': trade_amount
                        })
                        
                        # 下一个状态将在下次循环中获取，这里暂不更新模型
                        # 将在下次调用时使用close_position的结果更新模型
                
                elif signal == -1:  # 卖出信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 记录开仓前状态
                    prev_state = current_state
                    
                    # 开空仓
                    order = self.trader.open_short(amount=trade_amount)
                    if order:
                        self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        
                        # 记录交易历史
                        self.trade_id += 1
                        self.trades_history.append({
                            'id': self.trade_id,
                            'type': 'SHORT',
                            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_price': current_price,
                            'amount': trade_amount
                        })
                        
                        # 下一个状态将在下次循环中获取，这里暂不更新模型
            
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
                        self.logger.info(f"持仓时间超过{self.max_position_hold_time}分钟，平仓")
                        self._handle_position_close(current_state, current_price, "time_limit")
                        return
                
                # 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # 检查止盈
                if profit_rate >= self.profit_target_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self._handle_position_close(current_state, current_price, "take_profit")
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self._handle_position_close(current_state, current_price, "stop_loss")
                    return
                
                # 生成新信号
                signal = self.generate_signal(klines)
                
                # 如果信号与当前持仓方向相反，则平仓
                if (position_side == "多" and signal == -1) or (position_side == "空" and signal == 1):
                    self.logger.info(f"根据RL模型信号({signal})平{position_side}仓")
                    self._handle_position_close(current_state, current_price, "signal_reverse")
                    return
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
    
    def _handle_position_close(self, current_state, current_price, reason):
        """处理平仓操作并更新RL模型"""
        try:
            # 获取平仓前的状态和持仓信息
            position = self.trader.get_position()
            if not position:
                return
                
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            position_side = "多" if position_amount > 0 else "空"
            
            # 计算持仓时间和利润
            entry_time = self.position_entry_time
            holding_time_minutes = 0
            if entry_time:
                holding_time_minutes = (time.time() - entry_time) / 60
            
            # 计算利润率
            if position_side == "多":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price
                
            # 执行平仓
            order = self.trader.close_position()
            if not order:
                return
                
            # 平仓成功，更新交易历史
            trade_record = None
            for trade in reversed(self.trades_history):
                if trade.get('exit_time') is None:
                    trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade['exit_price'] = current_price
                    trade['profit_rate'] = profit_rate
                    trade['profit'] = profit_rate * abs(position_amount) * entry_price
                    trade['holding_time'] = holding_time_minutes
                    trade['exit_reason'] = reason
                    trade_record = trade
                    break
            
            if trade_record:
                self.logger.info(f"交易完成: ID={trade_record['id']}, "
                                f"类型={trade_record['type']}, "
                                f"利润率={trade_record['profit_rate']:.4%}, "
                                f"持仓时间={trade_record['holding_time']:.1f}分钟, "
                                f"平仓原因={trade_record['exit_reason']}")
            
            # 获取新状态(平仓后)
            time.sleep(2)  # 稍等片刻让市场状态更新
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
            next_state = self._get_state(klines)
            
            if next_state is None:
                return
                
            # 根据平仓原因和结果计算奖励
            reward = 0
            
            # 基础奖励 - 实现的利润
            reward += profit_rate * 100  # 放大利润率作为基础奖励
            
            # 附加奖励
            if reason == "take_profit":
                reward += 1.0  # 止盈加分
            elif reason == "stop_loss":
                reward -= 0.5  # 止损轻微减分
            elif reason == "time_limit":
                reward -= 0.2  # 超时轻微减分
            elif reason == "signal_reverse":
                # 如果是信号反转，那么根据利润率调整奖励
                if profit_rate > 0:
                    reward += 0.5  # 有利润的信号反转加分
                else:
                    reward -= 0.3  # 亏损的信号反转减分
            
            # 时间奖励 - 鼓励短期高效交易
            if holding_time_minutes < self.max_position_hold_time * 0.3 and profit_rate > 0:
                reward += 0.3  # 快速盈利加分
            
            # 更新RL模型
            # 假设我们的平仓操作对应动作 1 (观望，因为平仓后就是观望状态)
            action = 1  # 观望动作
            done = True  # 交易完成
            
            self.update_model(current_state, action, next_state, reward, done)
            
            # 重置持仓记录
            self.position_entry_time = None
            self.position_entry_price = None
            
        except Exception as e:
            self.logger.error(f"处理平仓失败: {str(e)}")
    
    def run(self):
        """运行策略"""
        try:
            self.logger.info("启动强化学习趋势跟踪策略")
            
            # 初始训练
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
            if len(klines) >= self.training_lookback:
                if self.train_model(klines):
                    self.logger.info("初始模型训练完成")
            
            # 主循环
            while True:
                try:
                    self.monitor_position()
                except Exception as e:
                    self.logger.error(f"策略执行出错: {str(e)}")
                time.sleep(self.check_interval)
        except Exception as e:
            self.logger.error(f"策略运行失败: {str(e)}")
            raise