from trading_manager import TradingManager
from utils.logger import Logger
import time
import signal
import sys
import config

def main():
    # 初始化日志记录器
    logger = Logger.get_logger()

    # 初始化交易管理器
    trading_manager = None
    arb_manager = None
    try:
        trading_manager = TradingManager()

        # 可选启动套利管理器
        if getattr(config, 'ARBITRAGE_ENABLED', False):
            try:
                from arbitrage.arbitrage_manager import ArbitrageManager
                arb_manager = ArbitrageManager()
                logger.info("套利管理器初始化成功")
            except Exception as e:
                logger.error(f"套利管理器初始化失败: {e}", exc_info=True)
                arb_manager = None

        def signal_handler(signum, frame):
            """处理退出信号"""
            logger.info("接收到退出信号，正在关闭所有交易...")
            if arb_manager:
                try:
                    arb_manager.stop()
                except Exception as e:
                    logger.error(f"关闭套利管理器失败: {e}")
            if trading_manager:
                trading_manager.stop_trading()
            sys.exit(0)

        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 启动所有交易对的交易
        trading_manager.start_trading()
        logger.info(f"交易系统启动成功，正在交易的币对: {', '.join(config.SYMBOLS)}")

        if arb_manager:
            arb_manager.start()
            pair_ids = ', '.join(arb_manager.pairs.keys()) or '无'
            logger.info(f"套利系统启动成功，套利 pair: {pair_ids}")

        # 保持主线程运行，定期检查账户状态
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭交易系统...")
        if arb_manager:
            try:
                arb_manager.stop()
            except Exception:
                pass
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(0)
    except Exception as e:
        logger.error(f"交易系统运行错误: {str(e)}")
        if arb_manager:
            try:
                arb_manager.stop()
            except Exception:
                pass
        if trading_manager:
            trading_manager.stop_trading()
        sys.exit(1)

if __name__ == "__main__":
    main()
