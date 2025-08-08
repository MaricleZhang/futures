"""
基础策略类
"""
from abc import ABC, abstractmethod
import time
import logging

class BaseStrategy(ABC):
    def __init__(self, trader):
        """初始化基础策略
        Args:
            trader (BinanceFuturesTrader): 交易者实例
        """
        self.trader = trader
        self.logger = self.get_logger()
        
    def get_logger(self):
        """获取带有交易对标识的日志记录器"""
        return logging.getLogger(self.trader.symbol if self.trader and self.trader.symbol else 'root')
        
    def run(self):
        """运行策略"""
        try:
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
        pass
    
    @abstractmethod
    def monitor_position(self):
        """监控持仓并执行交易逻辑"""
        pass
    
    @abstractmethod
    def generate_signal(self, klines=None):
        """根据策略逻辑生成交易信号
        Args:
            klines: K线数据
        Returns:
            int: 信号值 (1=买入, -1=卖出, 2=平仓, 0=观望)
        """
        pass
