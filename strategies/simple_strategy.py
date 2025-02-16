"""
简单交易策略
结合MACD和RSI指标
"""
import os
import sys
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import talib
from strategies.base_strategy import BaseStrategy
from utils.logger import Logger
import time

class SimpleStrategy(BaseStrategy):
    def __init__(self, trader):
        super().__init__(trader)
        self.logger = Logger.get_logger()
        
        # 策略参数
        self.timeframe = '15m'  # 15分钟K线
        self.rsi_period = 14    # RSI周期
        self.rsi_buy = 30       # RSI买入阈值
        self.rsi_sell = 70      # RSI卖出阈值
        
        # MACD参数
        self.macd_fast = 12     # MACD快线周期
        self.macd_slow = 26     # MACD慢线周期
        self.macd_signal = 9    # MACD信号线周期
        
        # 仓位控制
        self.position_size = 0.7  # 每次开仓比例（占账户余额的百分比）
        self.current_position = None  # 当前持仓信息
        self.check_interval = 60  # 策略执行间隔（秒）
        
    def calculate_indicators(self, klines):
        """计算技术指标"""
        try:
            # 将K线数据转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = pd.to_numeric(df['close'])
            
            # 计算RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # 计算MACD
            macd, signal, hist = talib.MACD(df['close'], 
                                          fastperiod=self.macd_fast,
                                          slowperiod=self.macd_slow,
                                          signalperiod=self.macd_signal)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            return df
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {str(e)}")
            return None
            
    def update_position(self):
        """更新当前持仓信息"""
        try:
            # 从交易器获取当前持仓信息
            self.current_position = self.trader.get_position()
            return True
        except Exception as e:
            self.logger.error(f"更新持仓信息失败: {str(e)}")
            return False
            
    def should_close_position(self):
        """判断是否应该平仓"""
        try:
            # 如果没有持仓，不需要平仓
            if not self.current_position or float(self.current_position.get('positionAmt', 0)) == 0:
                return False
                
            # 获取最新K线数据
            klines = self.trader.get_klines(timeframe=self.timeframe)
            if not klines:
                return False
                
            # 计算技术指标
            df = self.calculate_indicators(klines)
            if df is None:
                return False
                
            # 获取最新的指标值
            latest = df.iloc[-1]
            
            # 如果是多仓
            if float(self.current_position.get('positionAmt', 0)) > 0:
                # RSI超过卖出阈值，或MACD死叉
                return (latest['rsi'] > self.rsi_sell or 
                        (latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0))
                        
            # 如果是空仓
            elif float(self.current_position.get('positionAmt', 0)) < 0:
                # RSI低于买入阈值，或MACD金叉
                return (latest['rsi'] < self.rsi_buy or 
                        (latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0))
                        
            return False
            
        except Exception as e:
            self.logger.error(f"检查平仓条件失败: {str(e)}")
            return False
            
    def should_open_position(self):
        """判断是否应该开仓"""
        try:
            # 如果已有持仓，不开新仓
            if self.current_position and float(self.current_position.get('positionAmt', 0)) != 0:
                return False
                
            # 获取最新K线数据
            klines = self.trader.get_klines(timeframe=self.timeframe)
            if not klines or len(klines) == 0:
                return False
                
            # 计算技术指标
            df = self.calculate_indicators(klines)
            if df is None or len(df) == 0:
                return False
                
            # 获取最新的指标值
            latest = df.iloc[-1]
            
            # RSI低于买入阈值且MACD金叉，开多仓
            rsi_condition = float(latest['rsi']) < self.rsi_buy
            macd_condition = (float(latest['macd']) > float(latest['macd_signal']) and 
                            float(latest['macd_hist']) > 0)
            
            return rsi_condition and macd_condition
            
        except Exception as e:
            self.logger.error(f"检查开仓条件失败: {str(e)}")
            return False
            
    def calculate_position_size(self):
        """计算开仓数量"""
        try:
            # 获取账户余额
            balance = self.trader.get_balance()
            if not balance:
                return 0
                
            # 获取当前市场价格
            price = self.trader.get_market_price()
            if not price:
                return 0
                
            # 计算开仓数量（根据账户余额的百分比）
            position_value = float(balance['free']) * self.position_size
            amount = position_value / price
            
            # 获取交易对信息
            market = self.trader.get_market_info()
            precision = market['precision']['amount']  # 数量精度
            
            # 根据精度截断数量
            amount = float(format(amount, f'.{precision}f'))
            
            return amount
            
        except Exception as e:
            self.logger.error(f"计算开仓数量失败: {str(e)}")
            return 0
            
    def should_open_long(self, df):
        """判断是否应该开多仓"""
        try:
            # 获取最新的指标值
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2]
            
            # RSI超卖且MACD金叉或柱状图由负转正
            if (current_rsi < self.rsi_buy and 
                ((current_macd > current_signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]) or
                 (current_hist > 0 and prev_hist <= 0))):
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"判断开多仓失败: {str(e)}")
            return False
            
    def should_open_short(self, df):
        """判断是否应该开空仓"""
        try:
            # 获取最新的指标值
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2]
            
            # RSI超买且MACD死叉或柱状图由正转负
            if (current_rsi > self.rsi_sell and 
                ((current_macd < current_signal and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]) or
                 (current_hist < 0 and prev_hist >= 0))):
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"判断开空仓失败: {str(e)}")
            return False
            
    def should_close_long(self, df):
        """判断是否应该平多仓"""
        try:
            # 获取最新的指标值
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2]
            
            # RSI超买或MACD死叉
            if (current_rsi > self.rsi_sell or 
                (current_macd < current_signal and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]) or
                (current_hist < 0 and prev_hist >= 0)):
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"判断平多仓失败: {str(e)}")
            return False
            
    def should_close_short(self, df):
        """判断是否应该平空仓"""
        try:
            # 获取最新的指标值
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2]
            
            # RSI超卖或MACD金叉
            if (current_rsi < self.rsi_buy or 
                (current_macd > current_signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]) or
                (current_hist > 0 and prev_hist <= 0)):
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"判断平空仓失败: {str(e)}")
            return False
            
    def execute_strategy(self):
        """执行策略（为了兼容性保留，实际调用monitor_position）"""
        self.monitor_position()
            
    def monitor_position(self):
        """监控并执行交易策略"""
        try:
            # 更新当前持仓信息
            if not self.update_position():
                return
                
            # 检查是否需要平仓
            if self.should_close_position():
                self.logger.info("触发平仓信号")
                self.trader.close_position()
                return
                
            # 获取最新K线数据
            klines = self.trader.get_klines(timeframe=self.timeframe)
            if klines is None or len(klines) == 0:
                return
                
            # 计算技术指标
            df = self.calculate_indicators(klines)
            if df is None or len(df) == 0:
                return
                
            # 获取最新的指标值
            latest = df.iloc[-1]
            
            # 如果没有持仓，检查是否需要开仓
            if not self.current_position or float(self.current_position.get('positionAmt', 0)) == 0:
                # 判断开仓方向
                self.logger.info(f"RSI: {latest['rsi']}, MACD: {latest['macd']}, MACD Signal: {latest['macd_signal']}")
                if float(latest['rsi']) < self.rsi_buy and float(latest['macd']) > float(latest['macd_signal']):
                    # RSI超卖且MACD金叉，开多仓
                    self.logger.info("触发做多信号")
                    self.trader.open_long_position(self.position_size)
                elif float(latest['rsi']) > self.rsi_sell and float(latest['macd']) < float(latest['macd_signal']):
                    # RSI超买且MACD死叉，开空仓
                    self.logger.info("触发做空信号")
                    self.trader.open_short_position(self.position_size)
                    
        except Exception as e:
            self.logger.error(f"监控策略执行失败: {str(e)}")
            
    def run(self):
        """运行策略"""
        try:
            # 初始化交易配置
            self.trader.setup_trading_config()
            
            # 检查现有持仓
            position = self.trader.get_position()
            if position:
                position_amount = float(position['positionAmt'])
                position_side = 'long' if position_amount > 0 else 'short'
                self.logger.info(f"检测到现有{position_side}仓位，数量: {abs(position_amount)}")
                self.logger.info("策略将只监控现有仓位，不开新仓")
                self.current_position = position
            
            while True:
                try:
                    self.monitor_position()
                except Exception as e:
                    self.logger.error(f"策略执行出错: {str(e)}")
                
                time.sleep(self.check_interval)
                
        except Exception as e:
            self.logger.error(f"策略运行失败: {str(e)}")
            raise
            
    def stop(self):
        """停止策略"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            if position and float(position['positionAmt']) != 0:
                # 平掉所有持仓
                if float(position['positionAmt']) > 0:
                    self.trader.close_long(abs(float(position['positionAmt'])))
                else:
                    self.trader.close_short(abs(float(position['positionAmt'])))
                self.logger.info("策略停止，已平掉所有持仓")
        except Exception as e:
            self.logger.error(f"停止策略失败: {str(e)}")
