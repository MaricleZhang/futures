from trading_manager import TradingManager
from utils.logger import Logger
import time
import signal
import sys
import ccxt
import config

def main():
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 初始化交易管理器
    trading_manager = None
    try:
        trading_manager = TradingManager()
        
        def signal_handler(signum, frame):
            """处理退出信号"""
            logger.info("接收到退出信号，正在关闭所有交易...")
            if trading_manager:
                trading_manager.stop_trading()
            sys.exit(0)
            
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 启动所有交易对的交易
        trading_manager.start_trading()
        logger.info(f"交易系统启动成功，正在交易的币对: {', '.join(config.SYMBOLS)}")
        
        # 保持主线程运行，定期检查账户状态
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭交易系统...")
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(0)
    except Exception as e:
        logger.error(f"交易系统运行错误: {str(e)}")
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(1)

if __name__ == "__main__":
    main()
