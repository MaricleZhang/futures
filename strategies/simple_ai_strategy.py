import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
import talib
import config
from utils.logger import Logger

class SimpleAIStrategy(BaseStrategy):
    def __init__(self, trader):
        """初始化简单AI策略"""
        super().__init__(trader)
        self.logger = Logger.get_logger()
        
        # 策略参数
        self.rsi_period = 14
        self.ma_fast_period = 10
        self.ma_slow_period = 20
        self.atr_period = 14
        
        # 信号阈值
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # 止损和止盈设置
        self.stop_loss_percent = config.DEFAULT_STOP_LOSS_PERCENT / 100
        self.take_profit_percent = config.DEFAULT_TAKE_PROFIT_PERCENT / 100
        
        # 记录上一次的信号
        self.last_signal = 0
        
    def calculate_indicators(self, klines):
        """计算技术指标"""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        
        # 转换为数值类型
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        # 计算指标
        df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['MA_fast'] = talib.SMA(df['close'], timeperiod=self.ma_fast_period)
        df['MA_slow'] = talib.SMA(df['close'], timeperiod=self.ma_slow_period)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        return df
    
    def generate_signals(self, klines):
        """生成交易信号
        返回值:
        2: 平空开多
        1: 开多
        0: 持仓不变
        -1: 开空
        -2: 平多开空
        """
        if len(klines) < max(self.ma_slow_period, self.rsi_period):
            self.logger.warning("K线数据不足，无法生成信号")
            return 0
            
        df = self.calculate_indicators(klines)
        
        # 获取最新的指标值
        current_rsi = df['RSI'].iloc[-1]
        current_ma_fast = df['MA_fast'].iloc[-1]
        current_ma_slow = df['MA_slow'].iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # 记录指标值
        self.logger.info(f"当前指标 - RSI: {current_rsi:.2f}, 快速MA: {current_ma_fast:.2f}, 慢速MA: {current_ma_slow:.2f}")
        
        # 生成信号
        signal = 0
        
        # 超买超卖信号
        if current_rsi < self.rsi_oversold and current_ma_fast > current_ma_slow:
            if self.last_signal <= 0:  # 如果之前是空仓或无仓位
                signal = 2  # 平空开多
            else:
                signal = 1  # 开多
        elif current_rsi > self.rsi_overbought and current_ma_fast < current_ma_slow:
            if self.last_signal >= 0:  # 如果之前是多仓或无仓位
                signal = -2  # 平多开空
            else:
                signal = -1  # 开空
                
        # 更新上一次信号
        if signal != 0:
            self.last_signal = signal
            self.logger.info(f"生成新信号: {signal}")
        
        return signal
        
    def should_close_position(self, position_info, current_price):
        """判断是否应该平仓"""
        if not position_info:
            return False
            
        entry_price = float(position_info['entryPrice'])
        position_amount = float(position_info['positionAmt'])
        unrealized_pnl = float(position_info['unrealizedProfit'])
        
        # 计算收益率
        if position_amount > 0:  # 多仓
            profit_percent = (current_price - entry_price) / entry_price
        else:  # 空仓
            profit_percent = (entry_price - current_price) / entry_price
            
        # 止损平仓
        if profit_percent < -self.stop_loss_percent:
            self.logger.info(f"触发止损: 收益率 {profit_percent:.2%}")
            return True
            
        # 止盈平仓
        if profit_percent > self.take_profit_percent:
            self.logger.info(f"触发止盈: 收益率 {profit_percent:.2%}")
            return True
            
        return False
