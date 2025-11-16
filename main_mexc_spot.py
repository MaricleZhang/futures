from mexc_spot_trading_manager import MexcSpotTradingManager
from utils.logger import Logger
import time
import signal
import sys
import config_mexc_spot as config

def main():
    """MEXC现货交易主程序 - 与Binance合约交易完全隔离"""
    # 初始化日志记录器
    logger = Logger.get_logger()
    logger.info("=" * 80)
    logger.info("MEXC现货交易系统启动")
    logger.info("=" * 80)

    # 初始化MEXC现货交易管理器
    trading_manager = None
    try:
        trading_manager = MexcSpotTradingManager()

        def signal_handler(signum, frame):
            """处理退出信号"""
            logger.info("接收到退出信号，正在关闭MEXC现货交易...")
            if trading_manager:
                trading_manager.stop_trading()
            sys.exit(0)

        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 启动所有交易对的现货交易
        trading_manager.start_trading()
        logger.info(f"MEXC现货交易系统启动成功，正在交易的币对: {', '.join(config.SYMBOLS)}")
        logger.info(f"使用策略: {config.STRATEGY_TYPE}")
        logger.info("=" * 80)

        # 保持主线程运行，定期检查账户状态
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭MEXC现货交易系统...")
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(0)
    except Exception as e:
        logger.error(f"MEXC现货交易系统运行错误: {str(e)}")
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(1)

if __name__ == "__main__":
    main()
