from multi_symbol_trading_manager import MultiSymbolTradingManager
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
        trading_manager = MultiSymbolTradingManager()
        
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
            try:
                # 获取每个交易对的状态
                for symbol in config.SYMBOLS:
                    trader = trading_manager.traders.get(symbol)
                    if trader:
                        try:
                            # 获取账户余额
                            balance = trader.get_balance()
                            available_balance = float(balance['free'])
                            total_balance = float(balance['total'])
                            
                            # 获取持仓信息
                            position = trader.get_position(symbol)
                            position_size = float(position['info'].get('positionAmt', 0)) if position else 0
                            
                            logger.info(f"{symbol} 状态 - 可用余额: {available_balance:.2f} USDT, "
                                      f"总余额: {total_balance:.2f} USDT, 持仓量: {position_size}")
                                      
                        except ccxt.NetworkError as e:
                            logger.warning(f"{symbol} 网络错误: {str(e)}")
                        except Exception as e:
                            logger.error(f"{symbol} 获取状态失败: {str(e)}")
                            
            except Exception as e:
                logger.error(f"状态检查错误: {str(e)}")
                
            # 每1分钟检查一次状态
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
