import logging
import threading
import time
from trader import Trader
from strategies.ml_strategy_random_forest import RandomForestStrategy
from strategies.ml_strategy import MLStrategy #随机森林策略
from strategies.ml_strategy_deep_learning import DeepLearningStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.short_term_rf_strategy import ShortTermRFStrategy
import config

class TradingManager:
    def __init__(self):
        """初始化交易管理器"""
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
                symbol_logger = logging.getLogger(f'{symbol}')
                self.symbol_loggers[symbol] = symbol_logger
                
                trader = Trader(symbol)
                strategy = ShortTermRFStrategy(trader)
                self.traders[symbol] = trader
                self.strategies[symbol] = strategy
                symbol_logger.info(f"初始化 {symbol} 交易器和策略成功")
            except Exception as e:
                self.logger.error(f"初始化 {symbol} 交易器和策略失败: {str(e)}")
                
    def trade_symbol(self, symbol):
        """单个交易对的交易逻辑"""
        trader = self.traders[symbol]
        strategy = self.strategies[symbol]
        symbol_config = config.SYMBOL_CONFIGS[symbol]
        logger = self.symbol_loggers[symbol]
        
        while True:
            try:
                # 获取账户余额
                balance = trader.get_balance()
                available_balance = float(balance['free'])
                total_balance = float(balance['total'])
                logger.info(f"当前账户余额: 可用={available_balance:.8f} USDT, 总额={total_balance:.8f} USDT")
                
                # 获取K线数据用于AI分析
                klines = trader.get_klines(symbol=symbol, limit=config.AI_KLINES_LIMIT)
                if len(klines) < config.MIN_KLINES_FOR_AI:
                    logger.info("K线数据不足，等待更多数据...")
                    time.sleep(60)
                    continue
                
                # 生成交易信号
                signal = strategy.generate_signal(klines)
                current_price = trader.get_market_price(symbol)
                position = trader.get_position(symbol)
                position_amount = 0
                
                if position and 'info' in position:
                    position_amount = float(position['info'].get('positionAmt', 0))
                    if position['info'].get('positionSide', 'BOTH') == 'SHORT':  # 如果是空头仓位
                        position_amount = -abs(position_amount)  # 确保为负数
                    entry_price = float(position['info'].get('entryPrice', 0))
                    
                    # 计算未实现盈亏
                    if position_amount > 0:  # 多仓
                        unrealized_pnl = position_amount * (current_price - entry_price)
                    else:  # 空仓
                        unrealized_pnl = position_amount * (entry_price - current_price)
                        
                    position_value = abs(position_amount * entry_price)
                    profit_rate = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                    
                    logger.info(f"当前价格: {current_price}, AI信号: {signal}, "
                              f"持仓数量: {position_amount}, 开仓均价: {entry_price}, "
                              f"未实现盈亏: {unrealized_pnl:.2f} USDT ({profit_rate:.2f}%)")
                else:
                    logger.info(f"当前价格: {current_price}, AI信号: {signal}")
                    
                # 计算交易数量
                trade_amount = (available_balance * symbol_config['trade_amount_percent'] / 100) / current_price
                
                # 根据信号执行交易
                if signal == 1:  # 买入信号
                    if position_amount < 0:  # 有空仓，先平空
                        trader.close_position(symbol)
                    if position_amount <= 0:  # 没有多仓时开多
                        trader.open_long(symbol, trade_amount)
                elif signal == -1:  # 卖出信号
                    if position_amount > 0:  # 有多仓，先平多
                        trader.close_position(symbol)
                    if position_amount >= 0:  # 没有空仓时开空
                        trader.open_short(symbol, trade_amount)
                # else:  # 观望信号
                    # if abs(position_amount) > 0:  # 有持仓就平掉
                        # trader.close_position(symbol)
            except Exception as e:
                logger.error(f"{symbol} 交易过程出错: {str(e)}")
                time.sleep(10)  # 错误后等待较短时间
                
            # 等待下一个交易周期
            time.sleep(symbol_config.get('check_interval', 60))  # 使用交易对配置的间隔时间
                
    def start_trading(self):
        """启动所有交易对的交易"""
        for symbol in self.traders.keys():
            thread = threading.Thread(target=self.trade_symbol, args=(symbol,))
            thread.daemon = True
            thread.start()
            self.trading_threads[symbol] = thread
            self.symbol_loggers[symbol].info("交易线程启动成功")
            
        # 启动线程监控
        monitor_thread = threading.Thread(target=self.monitor_threads)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.logger.info("启动线程监控")
            
    def monitor_threads(self):
        """监控所有交易线程，确保它们在运行"""
        while True:
            for symbol, thread in self.trading_threads.items():
                if not thread.is_alive():
                    self.symbol_loggers[symbol].warning("交易线程已停止，正在重启...")
                    new_thread = threading.Thread(target=self.trade_symbol, args=(symbol,))
                    new_thread.daemon = True
                    new_thread.start()
                    self.trading_threads[symbol] = new_thread
                    self.symbol_loggers[symbol].info("交易线程重启成功")
            time.sleep(60)

    def stop_trading(self):
        """停止所有交易"""
        # 关闭所有持仓
        for symbol, trader in self.traders.items():
            try:
                # trader.close_position(symbol)
                self.symbol_loggers[symbol].info(f"关闭 {symbol} 所有持仓")
            except Exception as e:
                self.symbol_loggers[symbol].error(f"关闭 {symbol} 持仓失败: {str(e)}")
