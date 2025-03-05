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
import random
import os
import warnings
import config
from strategies.base_strategy import BaseStrategy


class PPOTrendStrategy(BaseStrategy):
    """PPOTrendStrategy - PPO强化学习趋势跟踪策略
    
    基于PPO (Proximal Policy Optimization)强化学习算法的交易策略，使用5分钟K线数据进行市场分析和交易决策。
    PPO算法相比传统DQN有更好的样本效率和稳定性，能够处理连续动作空间并防止策略更新过大。
    
    特点:
    1. 双网络架构: 使用演员-评论家(Actor-Critic)结构进行策略优化和价值评估
    2. 近端策略优化: 通过限制新旧策略差异来提高训练稳定性
    3. 多周期分析: 结合多个时间框架的特征进行决策
    4. 自适应学习: 根据市场波动性调整学习率和更新频率
    5. 风险管理: 综合考虑持仓时间、未实现盈亏和市场波动进行风控
    """
    
    MODEL_NAME = "PPOTrend"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化PPO趋势跟踪策略"""
        super().__init__(trader)

        # 抑制TensorFlow警告
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings('ignore', message='Even though the.*tf.data functions')
        
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '5m'      # 使用5分钟K线
        self.check_interval = 60        # 检查信号间隔(秒)
        self.lookback_period = 120      # 计算指标所需的K线数量
        self.training_lookback = 500    # 训练所需K线数量
        
        # PPO模型参数
        self.state_size = 40            # 状态特征维度
        self.action_size = 3            # 动作空间大小: [卖出, 观望, 买入]
        self.batch_size = 64            # 批次大小
        self.gamma = 0.99               # 奖励折扣因子
        self.clip_ratio = 0.2           # PPO裁剪参数
        self.policy_learning_rate = 0.0001  # 策略网络学习率
        self.value_learning_rate = 0.0005   # 价值网络学习率
        self.gae_lambda = 0.95          # GAE参数Lambda
        self.update_epochs = 4          # 每次训练的epoch数
        self.memory_size = 2000         # 经验回放缓冲区大小
        
        # 风险控制参数
        self.max_position_hold_time = 240  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.5     # 目标利润率 50%
        self.stop_loss_pct = 0.05       # 止损率 5%
        self.max_trades_per_hour = 3    # 每小时最大交易次数
        
        # 交易状态
        self.trade_count_hour = 0       # 当前小时交易次数
        self.last_trade_hour = None     # 上次交易的小时
        self.position_entry_time = None # 开仓时间
        self.position_entry_price = None # 开仓价格
        self.last_action = 1            # 上一次动作 (初始为观望)
        
        # 经验回放缓冲区
        self.memory = []
        
        # PPO网络模型
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        
        # 存储当前回合的数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []  # 存储动作概率以计算旧策略的概率比
        
        # 模型存储路径
        self.model_dir = "models"
        self.actor_model_path = f"{self.model_dir}/ppo_actor_{self.trader.symbol}.h5"
        self.critic_model_path = f"{self.model_dir}/ppo_critic_{self.trader.symbol}.h5"
        self.metadata_path = f"{self.model_dir}/ppo_metadata_{self.trader.symbol}.json"
        
        # 性能跟踪
        self.cum_reward = 0
        self.trades_history = []
        self.trade_id = 0
        
        self.last_training_time = time.time()
        self.retraining_interval = 3600  # 1小时重新训练一次

        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # 尝试加载模型
        model_loaded = self._load_model()
    
        # 如果没有成功加载模型，则执行初始训练
        if not model_loaded:
            self.logger.info("模型加载失败，执行初始训练...")
            initial_klines = self.trader.get_klines(interval=self.kline_interval, limit=1000)
            if len(initial_klines) > 500:
                self.logger.info(f"开始初始训练，使用{len(initial_klines)}根K线数据...")
                for _ in range(5):  # 进行5轮初始训练
                    self.train_model(initial_klines)
                self._save_model()  # 保存训练好的模型
                self.logger.info("初始训练完成")
            else:
                self.logger.warning("K线数据不足，无法进行初始训练")
        else:
            self.logger.info("成功加载保存的模型，可以直接使用")
    
    def _build_actor_model(self):
        """构建策略网络(Actor)"""
        inputs = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.policy_learning_rate),
            loss='categorical_crossentropy'
        )
        return model
    
    def _build_critic_model(self):
        """构建价值网络(Critic)"""
        inputs = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.value_learning_rate),
            loss='mse'
        )
        return model
    
    def _load_model(self):
        """加载保存的模型"""
        try:
            # 加载Actor模型
            if os.path.exists(self.actor_model_path):
                self.actor_model = load_model(self.actor_model_path)
                self.logger.info(f"已加载策略网络(Actor)模型: {self.actor_model_path}")
                
                # 加载Critic模型
                if os.path.exists(self.critic_model_path):
                    self.critic_model = load_model(self.critic_model_path)
                    self.logger.info(f"已加载价值网络(Critic)模型: {self.critic_model_path}")
                    
                    # 尝试加载元数据
                    if os.path.exists(self.metadata_path):
                        import json
                        with open(self.metadata_path, 'r') as f:
                            metadata = json.load(f)
                            self.logger.info(f"已加载元数据: {self.metadata_path}")
                            
                    return True
                else:
                    self.logger.warning(f"未找到价值网络模型: {self.critic_model_path}")
                    return False
            else:
                self.logger.info("未找到保存的模型，将使用新模型")
                return False
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _save_model(self):
        """保存模型"""
        try:
            # 保存Actor模型
            self.actor_model.save(self.actor_model_path)
            self.logger.info(f"策略网络(Actor)模型已保存: {self.actor_model_path}")
            
            # 保存Critic模型
            self.critic_model.save(self.critic_model_path)
            self.logger.info(f"价值网络(Critic)模型已保存: {self.critic_model_path}")
            
            # 保存元数据
            metadata = {
                'last_update': time.time(),
                'cum_reward': self.cum_reward,
                'trade_count': len(self.trades_history)
            }
            
            # 保存到单独的文件
            import json
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            self.logger.info(f"元数据已保存: {self.metadata_path}")
                
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
            
            # 检查是否有缺失值
            if df.isnull().any().any():
                self.logger.warning("K线数据存在缺失值，进行插值处理")
                df = df.interpolate(method='linear')
            
            # 准备特征
            features = []
            
            # 1. 价格特征
            close = df['close'].values
            open_price = df['open'].values
            high = df['high'].values
            low = df['low'].values
            
            # 价格变化特征
            price_change = df['close'].pct_change()
            features.append(price_change.iloc[-1])  # 最近一根K线的价格变化率
            features.append(price_change.rolling(5).mean().iloc[-1])  # 5根K线的平均变化率
            features.append(price_change.rolling(10).mean().iloc[-1])  # 10根K线的平均变化率
            
            # 价格结构特征
            # K线实体占比
            body_ratio = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features.append(body_ratio.iloc[-1])
            features.append(body_ratio.rolling(5).mean().iloc[-1])
            
            # 上下影线占比
            upper_shadow = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            lower_shadow = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
            features.append(upper_shadow.iloc[-1])
            features.append(lower_shadow.iloc[-1])
            
            # 2. 技术指标
            # 多周期RSI
            for period in [6, 14, 21]:
                rsi = talib.RSI(close, timeperiod=period)
                features.append(rsi[-1] / 100)  # 归一化到0-1
                features.append((rsi[-1] - rsi[-2]) / 100)  # RSI变化率
            
            # 多周期移动平均
            for period in [5, 10, 20, 50]:
                ma = talib.SMA(close, timeperiod=period)
                features.append((close[-1] / ma[-1]) - 1)  # 价格相对于MA的偏离度
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features.append(macd[-1] / close[-1] if close[-1] != 0 else 0)
            features.append(macd_signal[-1] / close[-1] if close[-1] != 0 else 0)
            features.append(macd_hist[-1] / close[-1] if close[-1] != 0 else 0)
            features.append((macd_hist[-1] - macd_hist[-2]) / close[-1] if close[-1] != 0 else 0)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features.append((close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5)
            features.append((upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else 0)
            
            # 多周期ATR - 波动率
            for period in [7, 14, 21]:
                atr = talib.ATR(high, low, close, timeperiod=period)
                features.append(atr[-1] / close[-1] if close[-1] != 0 else 0)
            
            # 随机指标
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            features.append(slowk[-1] / 100)
            features.append(slowd[-1] / 100)
            features.append((slowk[-1] - slowd[-1]) / 100)
            
            # 3. 成交量特征
            volume = df['volume'].values
            vol_ma = talib.SMA(volume, timeperiod=10)
            features.append(volume[-1] / vol_ma[-1] if vol_ma[-1] != 0 else 1.0)
            
            # 成交量趋势
            features.append(volume[-1] / volume[-6] if volume[-6] != 0 else 1.0)
            
            # OBV - 能量潮指标
            obv = talib.OBV(close, volume)
            features.append((obv[-1] - obv[-10]) / obv[-10] if obv[-10] != 0 else 0)
            
            # 4. 市场模式特征
            # ADX - 趋势强度
            adx = talib.ADX(high, low, close, timeperiod=14)
            features.append(adx[-1] / 100)
            
            # Aroon - 趋势方向和强度
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features.append(aroon_up[-1] / 100)
            features.append(aroon_down[-1] / 100)
            
            # 多周期CCI - 顺势指标
            for period in [14, 20]:
                cci = talib.CCI(high, low, close, timeperiod=period)
                features.append(cci[-1] / 200 + 0.5)  # 归一化到0-1
            
            # 5. 当前持仓特征
            position = self.trader.get_position()
            position_amt = 0
            entry_price = 0
            holding_time = 0
            
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    entry_price = float(position['info'].get('entryPrice', 0))
                    if self.position_entry_time:
                        holding_time = (time.time() - self.position_entry_time) / 60  # 转为分钟
                        holding_time = min(holding_time / self.max_position_hold_time, 1.0)  # 归一化
            
            # 持仓方向
            features.append(1 if position_amt > 0 else (-1 if position_amt < 0 else 0))
            
            # 持仓盈亏
            if abs(position_amt) > 0 and entry_price != 0:
                if position_amt > 0:  # 多仓
                    profit_pct = (close[-1] - entry_price) / entry_price
                else:  # 空仓
                    profit_pct = (entry_price - close[-1]) / entry_price
                features.append(profit_pct)
            else:
                features.append(0)
                
            # 持仓时间
            features.append(holding_time)
            
            # 确保维度正确
            features = np.array(features, dtype=np.float32)
            
            # 检查是否有NaN或无穷值
            if np.isnan(features).any() or np.isinf(features).any():
                self.logger.warning("特征中包含NaN或Inf值，将被替换为0")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 验证特征数量
            if len(features) != self.state_size:
                self.logger.warning(f"特征维度不匹配! 预期 {self.state_size}, 实际 {len(features)}")
                if len(features) < self.state_size:
                    padding = np.zeros(self.state_size - len(features), dtype=np.float32)
                    features = np.concatenate([features, padding])
                else:
                    features = features[:self.state_size]
            
            return features
            
        except Exception as e:
            self.logger.error(f"提取状态特征失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _calculate_reward(self, action, prev_state, next_state, current_price, next_price):
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
            
            # 价格变化奖励
            price_change_pct = (next_price - current_price) / current_price
            
            # 持仓奖励
            if abs(position_amt) > 0:
                if position_amt > 0:  # 多仓
                    position_reward = price_change_pct * 100  # 放大奖励
                else:  # 空仓
                    position_reward = -price_change_pct * 100
                
                # 持仓方向与动作一致性奖励
                if (position_amt > 0 and action == 2) or (position_amt < 0 and action == 0):
                    # 持有正确方向的仓位
                    action_reward = 0.2
                elif (position_amt > 0 and action == 0) or (position_amt < 0 and action == 2):
                    # 持仓方向与动作不一致
                    action_reward = -0.2
                else:
                    # 观望
                    action_reward = 0
            else:
                # 无持仓
                position_reward = 0
                
                # 根据价格趋势给予观望的奖励
                if action == 1:  # 观望
                    if abs(price_change_pct) < 0.0005:  # 价格变化很小时，观望是好的
                        action_reward = 0.1
                    else:
                        action_reward = -0.05  # 价格明显变化时，观望是次优的
                else:
                    action_reward = 0
            
            # 风险控制奖励
            risk_reward = 0
            
            # 如果有持仓，计算当前盈亏
            if abs(position_amt) > 0 and entry_price > 0:
                current_profit_pct = 0
                
                if position_amt > 0:  # 多仓
                    current_profit_pct = (current_price - entry_price) / entry_price
                else:  # 空仓
                    current_profit_pct = (entry_price - current_price) / entry_price
                
                # 止盈奖励
                if current_profit_pct >= self.profit_target_pct * 0.7 and action != 1:
                    risk_reward += 0.5  # 接近目标利润且准备平仓
                
                # 止损惩罚
                if current_profit_pct <= -self.stop_loss_pct * 0.7 and action != 1:
                    risk_reward += 0.3  # 接近止损点且准备平仓
                elif current_profit_pct <= -self.stop_loss_pct and action == 1:
                    risk_reward -= 0.5  # 亏损严重还坚持观望
            
            # 交易频率控制奖励
            freq_reward = 0
            
            # 过度交易惩罚
            if self.last_action != action and action != 1:  # 动作改变且不是观望
                freq_reward -= 0.1
            
            # 长时间持仓惩罚
            if abs(position_amt) > 0 and self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60  # 转为分钟
                if holding_time > self.max_position_hold_time * 0.8:
                    freq_reward -= 0.2  # 接近最大持仓时间
            
            # 总奖励计算
            total_reward = position_reward + action_reward + risk_reward + freq_reward
            
            self.logger.debug(f"""
                奖励组成:
                - 持仓奖励: {position_reward:.4f}
                - 动作奖励: {action_reward:.4f}
                - 风险奖励: {risk_reward:.4f}
                - 频率奖励: {freq_reward:.4f}
                = 总奖励: {total_reward:.4f}
            """)
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"计算奖励失败: {str(e)}")
            return 0
    
    def _remember(self, state, action, reward, next_state, done, action_prob):
        """将经验添加到回放缓冲区"""
        try:
            if state is not None and next_state is not None:
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(done)
                self.action_probs.append(action_prob)
        except Exception as e:
            self.logger.error(f"保存经验失败: {str(e)}")
    
    def _train_ppo(self):
        """使用PPO算法训练模型"""
        try:
            if len(self.states) < self.batch_size:
                return
            
            states = np.array(self.states)
            actions = np.array(self.actions)
            rewards = np.array(self.rewards)
            next_states = np.array(self.next_states)
            dones = np.array(self.dones)
            old_action_probs = np.array(self.action_probs)
            
            # 1. 计算优势函数(Advantage)和回报(Returns)
            # 使用GAE(Generalized Advantage Estimation)
            values = self.critic_model.predict(states, verbose=0).flatten()
            next_values = self.critic_model.predict(next_states, verbose=0).flatten()
            
            # 计算时序差分目标(TD targets)和优势
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            gae = 0
            
            # 从后向前计算GAE
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    gae = deltas[t]
                else:
                    gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
            
            # 标准化优势函数
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 2. 训练Actor网络
            for _ in range(self.update_epochs):
                # 为每个动作创建one-hot编码
                actions_one_hot = np.zeros((len(actions), self.action_size))
                for i, a in enumerate(actions):
                    actions_one_hot[i, a] = 1
                
                # 获取当前策略下的动作概率
                current_probs = self.actor_model.predict(states, verbose=0)
                
                # 对于每个样本，提取对应动作的概率
                current_action_probs = np.sum(current_probs * actions_one_hot, axis=1)
                
                # 计算概率比率
                ratio = current_action_probs / (old_action_probs + 1e-8)
                
                # 计算裁剪的目标函数
                clip_1 = ratio * advantages
                clip_2 = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                
                # 最小化负的目标函数(相当于最大化目标函数)
                actor_loss = -np.minimum(clip_1, clip_2)
                
                # 自定义训练步骤
                with tf.GradientTape() as tape:
                    current_probs = self.actor_model(states, training=True)
                    current_action_probs = tf.reduce_sum(current_probs * actions_one_hot, axis=1)
                    ratio = current_action_probs / (old_action_probs + 1e-8)
                    clip_1 = ratio * advantages
                    clip_2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(clip_1, clip_2))
                
                grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                self.actor_model.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))
            
            # 3. 训练Critic网络
            for _ in range(self.update_epochs):
                self.critic_model.fit(states, returns, epochs=1, verbose=0, batch_size=min(64, len(states)))
            
            # 清空回合数据
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.action_probs = []
            
            self.logger.info("PPO模型训练完成")
            
        except Exception as e:
            self.logger.error(f"PPO训练失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
            
            # 使用Actor网络预测动作概率
            action_probs = self.actor_model.predict(state.reshape(1, -1), verbose=0)[0]
            self.logger.info(f"动作概率: 卖出={action_probs[0]:.4f}, 观望={action_probs[1]:.4f}, 买入={action_probs[2]:.4f}")
            
            # 根据概率随机选择动作
            # 探索与利用的平衡 - 高概率动作被选中的几率更大
            action = np.random.choice(self.action_size, p=action_probs)
            
            # 记录选择的动作和概率
            self.last_action = action
            
            # 映射动作到信号
            # 动作空间: [0=卖出, 1=观望, 2=买入]
            # 信号映射: [0->-1, 1->0, 2->1]
            signal_mapping = {0: -1, 1: 0, 2: 1}
            signal = signal_mapping[action]
            
            # 记录到日志
            signal_name = "卖出" if signal == -1 else ("观望" if signal == 0 else "买入")
            self.logger.info(f"选择动作: {signal_name}, 概率: {action_probs[action]:.4f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0
    
    def update_model(self, prev_state, action, next_state, reward, done, action_prob):
        """更新PPO模型"""
        try:
            if prev_state is not None and next_state is not None:
                # 记录经验
                self._remember(prev_state, action, reward, next_state, done, action_prob)
                
                # 累计奖励
                self.cum_reward += reward
                
                # 如果回合结束或积累了足够的样本，则进行训练
                if done or len(self.states) >= self.batch_size:
                    self._train_ppo()
                    
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
    
    def should_retrain(self):
        """检查是否需要重新训练"""
        current_time = time.time()
        if current_time - self.last_training_time >= self.retraining_interval:
            return True
        return False
    
    def train_model(self, klines):
        """使用历史K线数据训练模型"""
        try:
            if not isinstance(klines, list) or len(klines) < 100:
                self.logger.error("K线数据不足，无法训练模型")
                return False
                
            self.logger.info(f"使用 {len(klines)} 根K线数据训练PPO模型")
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 获取价格序列
            prices = df['close'].values
            
            # 训练参数
            episodes = min(200, len(klines) - self.lookback_period)
            
            self.logger.info(f"开始训练，训练轮次: {episodes}")
            
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
                    # 使用Actor网络选择动作
                    action_probs = self.actor_model.predict(state.reshape(1, -1), verbose=0)[0]
                    action = np.random.choice(self.action_size, p=action_probs)
                    
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
                    reward = self._calculate_reward(action, state, next_state, current_price, next_price)
                    
                    # 记录经验
                    self._remember(state, action, reward, next_state, done, action_probs[action])
                    
                    # 更新状态
                    state = next_state
                    
                    if done:
                        break
                
                # 每积累一定样本后进行训练
                if len(self.states) >= self.batch_size:
                    self._train_ppo()
            
            # 训练后保存模型
            self._save_model()
            
            self.logger.info("模型训练完成")
            self.last_training_time = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {str(e)}")
            return False
    
    def print_metrics(self):
        """打印PPO模型的指标和性能统计"""
        try:
            self.logger.info("========== PPO模型指标 ==========")
            
            # 1. 基本信息
            self.logger.info(f"模型名称: {self.MODEL_NAME}")
            self.logger.info(f"版本: {self.VERSION}")
            self.logger.info(f"交易品种: {self.trader.symbol}")
            
            # 2. 模型参数
            self.logger.info("\n----- 模型参数 -----")
            self.logger.info(f"状态维度: {self.state_size}")
            self.logger.info(f"动作空间: {self.action_size}")
            self.logger.info(f"批次大小: {self.batch_size}")
            self.logger.info(f"奖励折扣因子: {self.gamma}")
            self.logger.info(f"PPO裁剪参数: {self.clip_ratio}")
            self.logger.info(f"策略网络学习率: {self.policy_learning_rate}")
            self.logger.info(f"价值网络学习率: {self.value_learning_rate}")
            
            # 3. 性能指标
            self.logger.info("\n----- 性能指标 -----")
            self.logger.info(f"累积奖励: {self.cum_reward:.4f}")
            self.logger.info(f"交易次数: {len(self.trades_history)}")
            
            # 4. 当前状态
            self.logger.info("\n----- 当前状态 -----")
            position = self.trader.get_position()
            if position and 'info' in position:
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    entry_price = float(position['info'].get('entryPrice', 0))
                    self.logger.info(f"当前持仓: {'多仓' if position_amt > 0 else '空仓'}")
                    self.logger.info(f"持仓数量: {abs(position_amt):.6f}")
                    self.logger.info(f"开仓价格: {entry_price:.2f}")
                    
                    # 计算持仓时间
                    if self.position_entry_time:
                        holding_time = (time.time() - self.position_entry_time) / 60  # 转为分钟
                        self.logger.info(f"持仓时间: {holding_time:.2f}分钟")
                    
                    # 计算当前盈亏
                    current_price = self.trader.get_market_price()
                    if position_amt > 0:  # 多仓
                        profit_pct = (current_price - entry_price) / entry_price * 100
                    else:  # 空仓
                        profit_pct = (entry_price - current_price) / entry_price * 100
                    self.logger.info(f"当前盈亏: {profit_pct:.2f}%")
                else:
                    self.logger.info("当前无持仓")
            else:
                self.logger.info("当前无持仓")
            
            # 5. 模型预测
            self.logger.info("\n----- 模型预测 -----")
            # 获取最新K线数据
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
            if len(klines) >= self.lookback_period:
                state = self._get_state(klines)
                if state is not None:
                    # 使用Actor网络预测动作概率
                    action_probs = self.actor_model.predict(state.reshape(1, -1), verbose=0)[0]
                    self.logger.info(f"动作概率: 卖出={action_probs[0]:.4f}, 观望={action_probs[1]:.4f}, 买入={action_probs[2]:.4f}")
                    
                    # 计算价值估计
                    value = self.critic_model.predict(state.reshape(1, -1), verbose=0)[0][0]
                    self.logger.info(f"当前状态价值估计: {value:.4f}")
            
            # 6. 交易历史统计
            if len(self.trades_history) > 0:
                self.logger.info("\n----- 交易历史统计 -----")
                wins = sum(1 for trade in self.trades_history if trade.get('profit', 0) > 0)
                losses = sum(1 for trade in self.trades_history if trade.get('profit', 0) <= 0)
                total_trades = len(self.trades_history)
                win_rate = wins / total_trades * 100 if total_trades > 0 else 0
                self.logger.info(f"总交易次数: {total_trades}")
                self.logger.info(f"盈利次数: {wins}")
                self.logger.info(f"亏损次数: {losses}")
                self.logger.info(f"胜率: {win_rate:.2f}%")
                
                # 计算平均盈亏
                if total_trades > 0:
                    avg_profit = sum(trade.get('profit', 0) for trade in self.trades_history) / total_trades
                    self.logger.info(f"平均盈亏: {avg_profit:.4f}")
                    
                    # 计算最大盈利和最大亏损
                    max_profit = max((trade.get('profit', 0) for trade in self.trades_history), default=0)
                    max_loss = min((trade.get('profit', 0) for trade in self.trades_history), default=0)
                    self.logger.info(f"最大盈利: {max_profit:.4f}")
                    self.logger.info(f"最大亏损: {max_loss:.4f}")
            
            # 7. 模型元数据
            if os.path.exists(self.metadata_path):
                self.logger.info("\n----- 模型元数据 -----")
                import json
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    last_update_time = datetime.fromtimestamp(metadata.get('last_update', 0))
                    self.logger.info(f"上次更新时间: {last_update_time}")
                    self.logger.info(f"记录的累积奖励: {metadata.get('cum_reward', 0):.4f}")
                    self.logger.info(f"记录的交易次数: {metadata.get('trade_count', 0)}")
            
            self.logger.info("=================================")
            
            # 返回一个字典，包含所有指标，方便其他地方使用
            return {
                "model_name": self.MODEL_NAME,
                "version": self.VERSION,
                "cum_reward": self.cum_reward,
                "trades_count": len(self.trades_history),
                "action_probs": action_probs.tolist() if 'action_probs' in locals() else None,
                "state_value": float(value) if 'value' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"打印指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
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
                        self.logger.info("PPO模型重新训练完成")
                    else:
                        self.logger.error("PPO模型重新训练失败")
                
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
                    
                    # 执行开多仓操作
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
                        
                        # 记录当前Actor网络的动作概率，用于后续更新
                        action_probs = self.actor_model.predict(current_state.reshape(1, -1), verbose=0)[0]
                        self.update_model(current_state, 2, None, 0, False, action_probs[2])
                
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
                        
                        # 记录交易历史
                        self.trade_id += 1
                        self.trades_history.append({
                            'id': self.trade_id,
                            'type': 'SHORT',
                            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_price': current_price,
                            'amount': trade_amount
                        })
                        
                        # 记录当前Actor网络的动作概率，用于后续更新
                        action_probs = self.actor_model.predict(current_state.reshape(1, -1), verbose=0)[0]
                        self.update_model(current_state, 0, None, 0, False, action_probs[0])
            
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
                    self.logger.info(f"根据策略信号({signal})平{position_side}仓")
                    self.print_metrics()
                    self._handle_position_close(current_state, current_price, "signal_reverse")
                    return
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
    
    def _handle_position_close(self, current_state, current_price, reason):
        """处理平仓操作并更新PPO模型"""
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
            
            # 获取当前观望动作(平仓)的概率
            action_probs = self.actor_model.predict(current_state.reshape(1, -1), verbose=0)[0]
            
            # 更新PPO模型
            action = 1  # 观望/平仓动作
            done = True  # 交易完成
            
            self.update_model(current_state, action, next_state, reward, done, action_probs[action])
            
            # 重置持仓记录
            self.position_entry_time = None
            self.position_entry_price = None
            
        except Exception as e:
            self.logger.error(f"处理平仓失败: {str(e)}")
    
    def run(self):
        """运行策略"""
        try:
            self.logger.info("启动PPO强化学习趋势跟踪策略")
            
            # 初始训练
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
            if len(klines) >= self.training_lookback:
                if not self._load_model() or self.should_retrain():
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
