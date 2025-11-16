import logging
import threading
import time
from mexc_spot_trader import MexcSpotTrader
from strategies.simple_adx_di_15m_strategy import SimpleADXDIStrategy15m
from strategies.kama_roc_adx_strategy import KAMARocAdxStrategy
from utils.logger import Logger
import config_mexc_spot as config

class MexcSpotTradingManager:
    """MEXC现货交易管理器 - 与Binance合约交易完全隔离"""

    def __init__(self):
        """初始化MEXC现货交易管理器"""
        # 初始化日志记录器
        self.logger = logging.getLogger()
        self.symbol_loggers = {}

        self.traders = {}
        self.strategies = {}
        self.trading_threads = {}

        # 初始化每个交易对的交易器和策略
        for symbol in config.SYMBOLS:
            try:
                # 为每个交易对创建独立的logger
                symbol_logger = logging.getLogger(f'MEXC_{symbol}')
                self.symbol_loggers[symbol] = symbol_logger

                trader = MexcSpotTrader(symbol)

                # 根据配置选择策略（MEXC现货不使用DeepSeek策略）
                strategy_type = getattr(config, 'STRATEGY_TYPE', 'kama_roc_adx').lower()
                if strategy_type == 'kama_roc_adx':
                    strategy = KAMARocAdxStrategy(trader)
                    symbol_logger.info(f"使用 KAMA-ROC-ADX 策略")
                elif strategy_type == 'simple_adx_di':
                    strategy = SimpleADXDIStrategy15m(trader)
                    symbol_logger.info(f"使用 Simple ADX-DI 策略")
                else:  # 默认使用 kama_roc_adx
                    strategy = KAMARocAdxStrategy(trader)
                    symbol_logger.info(f"使用 KAMA-ROC-ADX 策略（默认）")

                self.traders[symbol] = trader
                self.strategies[symbol] = strategy
                symbol_logger.info(f"初始化 {symbol} 交易器和策略成功")
            except Exception as e:
                self.logger.error(f"初始化 {symbol} 交易器和策略失败: {str(e)}")

    def trade_symbol(self, symbol):
        """单个交易对的现货交易逻辑"""
        trader = self.traders[symbol]
        strategy = self.strategies[symbol]
        symbol_config = config.SYMBOL_CONFIGS[symbol]
        logger = self.symbol_loggers[symbol]

        while True:
            try:
                # 获取USDT账户余额
                usdt_balance = trader.get_balance('USDT')
                available_balance = float(usdt_balance['free'])
                total_balance = float(usdt_balance['total'])
                logger.info(f"当前USDT余额: 可用={available_balance:.2f} USDT, 总额={total_balance:.2f} USDT")

                # 获取K线数据用于策略分析
                klines = trader.get_klines(symbol=symbol, interval=strategy.kline_interval, limit=strategy.training_lookback)

                if not klines or len(klines) == 0:
                    logger.warning(f"未能获取到K线数据，跳过本次交易")
                    time.sleep(symbol_config.get('check_interval', 60))
                    continue

                # 生成交易信号
                signal = strategy.generate_signal(klines)
                current_price = trader.get_market_price(symbol)

                # 获取当前持仓（币的数量）
                position = trader.get_position(symbol)
                position_amount = 0
                base_currency = symbol.split('/')[0]

                if position:
                    position_amount = float(position.get('free', 0))

                self.logger_info(symbol, position, current_price, signal)

                # 计算交易数量（买入数量 = 可用USDT * 百分比 / 当前价格）
                trade_amount_usdt = available_balance * symbol_config['trade_amount_percent'] / 100
                trade_amount = trade_amount_usdt / current_price

                # 根据信号执行现货交易
                if signal == 1:  # 买入信号
                    if position_amount <= 0:  # 没有持仓时买入
                        if available_balance >= symbol_config.get('min_notional', 10):
                            trader.buy(symbol, trade_amount)
                            logger.info(f"买入信号执行: {trade_amount:.6f} {base_currency}")
                        else:
                            logger.warning(f"可用余额不足，无法买入")
                    else:
                        logger.info(f"已有持仓，跳过买入")

                elif signal == -1:  # 卖出信号
                    if position_amount > 0:  # 有持仓时卖出
                        trader.sell_all(symbol)
                        logger.info(f"卖出信号执行: 卖出所有 {base_currency}")
                    else:
                        logger.info(f"没有持仓，跳过卖出")

                elif signal == 2:  # 平仓信号（现货就是卖出）
                    if position_amount > 0:  # 有持仓时卖出
                        trader.sell_all(symbol)
                        logger.info(f"平仓信号执行: 卖出所有 {base_currency}")
                    else:
                        logger.info(f"没有持仓，无需平仓")

                else:  # 观望信号
                    logger.info(f"观望信号，不执行任何操作")

            except Exception as e:
                logger.error(f"{symbol} 交易过程出错: {str(e)}")
                time.sleep(10)  # 错误后等待较短时间

            # 等待下一个交易周期
            time.sleep(symbol_config.get('check_interval', 60))

    def start_trading(self):
        """启动所有交易对的现货交易"""
        for symbol in self.traders.keys():
            thread = threading.Thread(target=self.trade_symbol, args=(symbol,))
            thread.daemon = True
            thread.start()
            self.trading_threads[symbol] = thread
            self.symbol_loggers[symbol].info("MEXC现货交易线程启动成功")

        # 启动线程监控
        monitor_thread = threading.Thread(target=self.monitor_threads)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.logger.info("启动MEXC现货交易线程监控")

    def monitor_threads(self):
        """监控所有交易线程，确保它们在运行"""
        while True:
            for symbol, thread in self.trading_threads.items():
                if not thread.is_alive():
                    self.symbol_loggers[symbol].warning("MEXC现货交易线程已停止，正在重启...")
                    new_thread = threading.Thread(target=self.trade_symbol, args=(symbol,))
                    new_thread.daemon = True
                    new_thread.start()
                    self.trading_threads[symbol] = new_thread
                    self.symbol_loggers[symbol].info("MEXC现货交易线程重启成功")
            time.sleep(60)

    def stop_trading(self):
        """停止所有MEXC现货交易"""
        # 卖出所有持仓（根据需要）
        for symbol, trader in self.traders.items():
            try:
                # 如果需要在停止时卖出所有持仓，取消下面的注释
                # trader.sell_all(symbol)
                self.symbol_loggers[symbol].info(f"停止 {symbol} 交易")
            except Exception as e:
                self.symbol_loggers[symbol].error(f"停止 {symbol} 交易失败: {str(e)}")

        # 清理策略资源
        for symbol, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'cleanup'):
                    strategy.cleanup()
                    self.symbol_loggers[symbol].info(f"清理 {symbol} 策略资源成功")
            except Exception as e:
                self.symbol_loggers[symbol].error(f"清理 {symbol} 策略资源失败: {str(e)}")

    def logger_info(self, symbol, position, current_price, signal=0):
        """记录现货交易信息到日志

        Args:
            symbol: 交易对名称
            position: 当前持仓信息
            current_price: 当前价格
            signal: 策略信号
        """
        try:
            logger = self.symbol_loggers.get(symbol)
            if not logger:
                return

            base_currency = symbol.split('/')[0]

            if position and position.get('amount', 0) > 0:
                position_amount = float(position.get('amount', 0))
                position_free = float(position.get('free', 0))
                position_value = position_amount * current_price

                logger.info(f"当前持仓: {position_amount:.6f} {base_currency}, 可用: {position_free:.6f}")
                logger.info(f"当前价格: {current_price:.6f}, 策略信号: {signal}, "
                          f"持仓价值: {position_value:.2f} USDT")
            else:
                logger.info(f"当前价格: {current_price:.6f}, 策略信号: {signal}, 无持仓")

        except Exception as e:
            if logger:
                logger.error(f"记录日志信息失败: {str(e)}")
