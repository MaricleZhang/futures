"""
基础策略类
"""
from abc import ABC, abstractmethod
import time

class BaseStrategy(ABC):
    def __init__(self, trader):
        """初始化策略"""
        self.trader = trader
        from utils.logger import Logger
        self.logger = Logger.get_logger()
        
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
