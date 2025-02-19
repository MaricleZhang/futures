from binance_futures_trader import BinanceFuturesTrader
from strategies.ml_strategy import MLStrategy
from utils.logger import Logger
import pandas as pd
import time
import config

def main():
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 初始化交易器
    trader = BinanceFuturesTrader()
    
    def close_with_retry(trader, logger, max_retries=3):
        """带重试机制的平仓函数"""
        last_error = None
        for attempt in range(max_retries):
            try:
                # 获取当前持仓信息并记录
                position = trader.get_position()
                if position:
                    logger.info(f"尝试平仓前的持仓状态: {position}")
                else:
                    logger.info("尝试平仓前没有持仓")
                    return None

                close_order = trader.close_position()
                if close_order:
                    logger.info(f"平仓成功: {close_order}")
                    return close_order
                else:
                    logger.warning("平仓函数返回空值")
                    return None

            except ccxt.NetworkError as e:
                last_error = e
                if attempt == max_retries - 1:
                    logger.error(f"平仓失败(网络错误，已重试{max_retries}次): {str(e)}")
                    raise
                logger.warning(f"平仓重试({attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1)
            except Exception as e:
                last_error = e
                logger.error(f"平仓失败(其他错误): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"平仓重试({attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1)

        if last_error:
            raise last_error
        return None
    
    try:
        # 初始化AI策略
        strategy = MLStrategy(trader)
        
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
                position = trader.get_position()
                if position and float(position['info'].get('positionAmt', 0)) != 0:
                    position_size = float(position['info'].get('positionAmt', 0))
                    entry_price = float(position['info'].get('entryPrice', 0))
                    unrealized_pnl = float(position['info'].get('unrealizedProfit', 0))
                    position_value = abs(position_size * entry_price)
                    profit_rate = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                    
                    logger.info(f"当前价格: {current_price}, AI信号: {signal}, "
                              f"未实现盈亏: {unrealized_pnl:.2f} USDT, 盈亏率: {profit_rate:.2f}%")
                else:
                    logger.info(f"当前价格: {current_price}, AI信号: {signal}")
                trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                
                # 获取当前持仓
                try:
                    position = trader.get_position()
                    position_amount = float(position['info'].get('positionAmt', 0)) if position else 0
                except Exception as e:
                    logger.info(f"获取持仓信息失败: {str(e)}")
                    position_amount = 0
                
                # 根据AI信号交易
                if signal == -2:  # 平多仓信号
                    if position_amount > 0:  # 有多仓
                        logger.info("触发平多仓信号...")
                        try:
                            order = trader.close_long(position_amount)
                            logger.info(f"平多仓成功: {order}")
                        except Exception as e:
                            logger.error(f"平多仓失败: {str(e)}")
                        time.sleep(1)  # 等待订单执行
                elif signal == 2:  # 平空仓信号
                    if position_amount < 0:  # 有空仓
                        logger.info("触发平空仓信号...")
                        try:
                            order = trader.close_short(abs(position_amount))
                            logger.info(f"平空仓成功: {order}")
                        except Exception as e:
                            logger.error(f"平空仓失败: {str(e)}")
                        time.sleep(1)  # 等待订单执行
                elif signal == 1 and position_amount <= 0:  # 做多信号且当前无多仓
                    # 如果有空仓，先平仓
                    if position_amount < 0:
                        logger.info("平空仓...")
                        try:
                            order = trader.close_short(abs(position_amount))
                            logger.info(f"平空仓成功: {order}")
                            time.sleep(1)  # 等待订单执行
                        except Exception as e:
                            logger.error(f"平空仓失败: {str(e)}")
                            continue
                        
                    # 开多仓
                    logger.info(f"开多仓，数量: {trade_amount}")
                    trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                    try:
                        order = trader.open_long(trade_amount)
                        logger.info(f"开多仓成功: {order}")
                    except Exception as e:
                        logger.error(f"开多仓失败: {str(e)}")
                        
                elif signal == -1 and position_amount >= 0:  # 做空信号且当前无空仓
                    # 如果有多仓，先平仓
                    if position_amount > 0:
                        logger.info("平多仓...")
                        try:
                            order = trader.close_long(position_amount)
                            logger.info(f"平多仓成功: {order}")
                            time.sleep(1)  # 等待订单执行
                        except Exception as e:
                            logger.error(f"平多仓失败: {str(e)}")
                            continue
                        
                    # 开空仓
                    logger.info(f"开空仓，数量: {trade_amount}")
                    trade_amount = available_balance * config.AI_TRADE_AMOUNT_PERCENT / 100 / current_price
                    try:
                        order = trader.open_short(trade_amount)
                        logger.info(f"开空仓成功: {order}")
                    except Exception as e:
                        logger.error(f"开空仓失败: {str(e)}")
                
                # 当前持仓方向与信号相反时平仓
                elif (signal == -1 and position_amount > 0) or (signal == 1 and position_amount < 0):
                    try:
                        close_order = close_with_retry(trader, logger)
                        if close_order:
                            logger.info(f"平仓: {close_order}")
                    except Exception as e:
                        logger.error(f"平仓最终失败: {str(e)}")
                        continue
                
                # 中性信号保持当前仓位
                elif signal == 0:
                    logger.info("观望信号，保持当前仓位")
                
                # 打印当前持仓信息
                try:
                    position = trader.get_position()
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
                close_order = close_with_retry(trader, logger)
                if close_order:
                    logger.info(f"平仓: {close_order}")
        except Exception as e:
            logger.info(f"清理订单和持仓时发生错误: {str(e)}")
        logger.info("程序已安全退出")

if __name__ == "__main__":
    main()
