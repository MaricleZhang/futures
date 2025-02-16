"""
主程序入口
"""
import sys
import time
from utils.logger import Logger
from binance_futures_trader import BinanceFuturesTrader
from strategies.simple_strategy import SimpleStrategy

class TradingSystem:
    def __init__(self):
        self.trader = None
        self.strategy = None
        self.is_running = False
        self.logger = Logger.get_logger()
        
    def setup(self):
        """初始化交易系统"""
        try:
            # 初始化交易器
            self.trader = BinanceFuturesTrader()
            
            # 初始化策略
            self.strategy = SimpleStrategy(self.trader)
            
        except Exception as e:
            self.logger.error(f"初始化交易系统失败: {str(e)}")
            sys.exit(1)
    
    def run(self):
        """运行交易系统"""
        try:
            self.is_running = True
            
            while self.is_running:
                try:
                    self.strategy.execute_strategy()
                    # 休眠60秒
                    time.sleep(60)
                    
                except Exception as e:
                    self.logger.error(f"策略运行错误: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("收到退出信号，正在关闭交易系统...")
            self.stop()
    
    def stop(self):
        """停止交易系统"""
        self.is_running = False
        if self.trader:
            self.trader.close_position()  # 平掉所有持仓
            self.logger.info("交易系统已安全停止")

def main():
    system = TradingSystem()
    system.setup()
    system.run()

if __name__ == "__main__":
    main()
