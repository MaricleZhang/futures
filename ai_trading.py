from binance_futures_trader import BinanceFuturesTrader
from strategies import AIStrategy
from utils.logger import Logger
import pandas as pd
import time
import config

def main():
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 初始化交易器
    trader = BinanceFuturesTrader()
    
    try:
        # 初始化AI策略
        strategy = AIStrategy(trader)
        
        # 获取账户余额
        balance = trader.get_balance()
        available_balance = float(balance['free'])
        total_balance = float(balance['total'])
        logger.info(f"当前账户余额: 可用={available_balance:.8f} USDT, 总额={total_balance:.8f} USDT")
        
        while True:
            try:
                # 获取K线数据用于AI分析
                klines = trader.get_klines(limit=config.AI_KLINES_LIMIT)  # 使用配置的K线数量
                if len(klines) < config.MIN_KLINES_FOR_AI:
                    logger.info("K线数据不足，等待更多数据...")
                    time.sleep(60)
                    continue
                
                # 生成交易信号
                signal = strategy.generate_signals(klines)
                current_price = trader.get_market_price()
                logger.info(f"当前价格: {current_price}, AI信号: {signal}")
                trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                
                # 获取当前持仓
                try:
                    position = trader.get_position()
                    position_amount = float(position.get('positionAmt', 0)) if position else 0
                except Exception as e:
                    logger.info(f"获取持仓信息失败: {str(e)}")
                    position_amount = 0
                
                # 根据AI信号交易
                if signal == 1 and position_amount <= 0:  # 做多信号且当前无多仓
                    # 如果有空仓，先平仓
                    if position_amount < 0:
                        logger.info("平空仓...")
                        trader.close_short()
                        time.sleep(1)  # 等待订单执行
                        
                    # 开多仓
                    logger.info(f"开多仓，数量: {trade_amount}")
                    trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                    try:
                        order = trader.place_order(
                            side='buy',
                            amount=trade_amount
                        )
                        logger.info(f"开多仓: {order}")
                    except Exception as e:
                        logger.info(f"开多仓失败: {str(e)}")
                    
                elif signal == -1 and position_amount >= 0:  # 做空信号且当前无空仓
                    # 如果有多仓，先平仓
                    if position_amount > 0:
                        logger.info("平多仓...")
                        trader.close_long()
                        time.sleep(1)  # 等待订单执行
                        
                    # 开空仓
                    logger.info(f"开空仓，数量: {trade_amount}")
                    trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                    try:
                        order = trader.place_order(
                            side='sell',
                            amount=trade_amount
                        )
                        logger.info(f"开空仓: {order}")
                    except Exception as e:
                        logger.info(f"开空仓失败: {str(e)}")
                
                elif signal == 0 and position_amount != 0:  # 平仓信号
                    try:
                        close_order = trader.close_position()
                        logger.info(f"平仓: {close_order}")
                    except Exception as e:
                        logger.info(f"平仓失败: {str(e)}")
                
                # 打印当前持仓信息
                try:
                    position = trader.get_position()
                    logger.info(f"当前持仓: {position}")
                except Exception as e:
                    logger.info(f"获取持仓信息失败: {str(e)}")
                
                # 等待一定时间再次检查
                time.sleep(config.DEFAULT_CHECK_INTERVAL)
                
            except Exception as e:
                logger.info(f"交易过程中发生错误: {str(e)}")
                time.sleep(60)  # 发生错误时等待较长时间
                
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.info(f"程序发生错误: {str(e)}")
    finally:
        # 确保取消所有未完成的订单并平仓
        try:
            trader.cancel_all_orders()
            position = trader.get_position()
            if position and float(position.get('positionAmt', 0)) != 0:
                trader.close_position()
        except Exception as e:
            logger.info(f"清理订单和持仓时发生错误: {str(e)}")
        logger.info("程序已安全退出")

if __name__ == "__main__":
    main()
