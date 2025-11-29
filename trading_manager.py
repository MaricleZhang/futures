import logging
import threading
import time
from trader import Trader
from strategies.simple_adx_di_15m_strategy import SimpleADXDIStrategy15m
from strategies.deepseek_trading_strategy import DeepSeekTradingStrategy
from strategies.pattern_probability_strategy import PatternProbabilityStrategy
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.xgboost_price_strategy import XGBoostPriceStrategy
from strategies.qwen_trading_strategy import QwenTradingStrategy
from strategies.kimi_trading_strategy import KimiTradingStrategy
from utils.logger import Logger
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

                # 根据配置选择策略
                strategy_type = getattr(config, 'STRATEGY_TYPE', 'deepseek').lower()

                if strategy_type == 'simple_adx_di':
                    strategy = SimpleADXDIStrategy15m(trader)
                    symbol_logger.info(f"使用 Simple ADX-DI 策略")
                elif strategy_type == 'pattern_probability':
                    strategy = PatternProbabilityStrategy(trader)
                    symbol_logger.info(f"使用 K线形态概率策略")
                elif strategy_type == 'trend_following':
                    strategy = TrendFollowingStrategy(trader)
                    symbol_logger.info(f"使用 趋势跟随策略")
                elif strategy_type == 'xgboost':
                    strategy = XGBoostPriceStrategy(trader)
                    symbol_logger.info(f"使用 XGBoost 策略")
                elif strategy_type == 'qwen':
                    strategy = QwenTradingStrategy(trader)
                    symbol_logger.info(f"使用 Qwen 策略")
                elif strategy_type == 'kimi':
                    strategy = KimiTradingStrategy(trader)
                    symbol_logger.info(f"使用 Kimi AI 策略")
                else:  # 默认使用 deepseek
                    strategy = DeepSeekTradingStrategy(trader)
                    symbol_logger.info(f"使用 DeepSeek AI 策略")


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
                klines = trader.get_klines(symbol=symbol, interval=strategy.kline_interval, limit=strategy.training_lookback)
                
                # 生成交易信号
                signal = strategy.generate_signal(klines)
                current_price = trader.get_market_price(symbol)
                position = trader.get_position(symbol)
                position_amount = 0
                
                self.logger_info(symbol, position, current_price, signal)
                
                # strategy.monitor_position()
                if position and 'info' in position:
                    position_amount = float(position['info'].get('positionAmt', 0))
                    
                # 计算交易数量
                trade_amount = (available_balance * symbol_config['trade_amount_percent'] / 100) / current_price
                
                # 根据信号执行交易
                if signal == 1:  # 买入信号
                    if position_amount < 0:  # 有空仓，先平空
                        trader.close_position(symbol)
                    if position_amount <= 0:  # 没有多仓时开多
                        # 使用配置中的止损作为兜底保护
                        trader.open_long(symbol, trade_amount, 
                                       stop_loss_pct=config.DEFAULT_STOP_LOSS_PERCENT,
                                       take_profit_pct=config.DEFAULT_TAKE_PROFIT_PERCENT if config.DEFAULT_TAKE_PROFIT_PERCENT > 0 else None)
                elif signal == -1:  # 卖出信号
                    if position_amount > 0:  # 有多仓，先平多
                        trader.close_position(symbol)
                    if position_amount >= 0:  # 没有空仓时开空
                        # 使用配置中的止损作为兜底保护
                        trader.open_short(symbol, trade_amount,
                                        stop_loss_pct=config.DEFAULT_STOP_LOSS_PERCENT,
                                        take_profit_pct=config.DEFAULT_TAKE_PROFIT_PERCENT if config.DEFAULT_TAKE_PROFIT_PERCENT > 0 else None)
                elif signal == 2:  # 平仓信号
                    if abs(position_amount) > 0:  # 有持仓就平掉
                        trader.close_position(symbol)

                else:  # 观望信号
                    if abs(position_amount) > 0:  # 有持仓就平掉
                        trader.close_position(symbol)
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
        
        # 清理策略资源
        for symbol, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'cleanup'):
                    strategy.cleanup()
                    self.symbol_loggers[symbol].info(f"清理 {symbol} 策略资源成功")
            except Exception as e:
                self.symbol_loggers[symbol].error(f"清理 {symbol} 策略资源失败: {str(e)}")

    def logger_info(self, symbol, position, current_price, signal=0):
        """记录交易信息到日志
        
        Args:
            symbol: 交易对名称
            position: 当前持仓信息
            current_price: 当前价格
            signal: AI信号
        """
        try:
            logger = self.symbol_loggers.get(symbol)
            if not logger:
                return
            
            # 获取交易统计信息
            try:
                trader = self.traders.get(symbol)
                if trader and hasattr(trader, 'trade_recorder'):
                    stats = trader.trade_recorder.get_statistics(symbol)
                    if stats['total_trades'] > 0:
                        logger.info(f"=== 交易统计 ===")
                        logger.info(f"总交易次数: {stats['total_trades']}, "
                                  f"胜率: {stats['win_rate']:.2f}%, "
                                  f"累计盈亏: {stats['total_profit']:.2f} USDT, "
                                  f"平均收益率: {stats['avg_profit_rate']:.2f}%")
            except Exception as e:
                logger.debug(f"获取交易统计失败: {str(e)}")
                
            if position and 'info' in position:
                position_amount = float(position['info'].get('positionAmt', 0))
                position_amount = abs(position_amount)
                entry_price = float(position['info'].get('entryPrice', 0))
                
                # 获取持仓方向
                position_side = position['info'].get('positionSide', 'BOTH')
                if position_side == 'BOTH':
                    # 在双向持仓模式下，通过持仓数量判断方向
                    direction = "多" if float(position['info'].get('positionAmt', 0)) > 0 else "空"
                else:
                    # 在单向持仓模式下，直接使用positionSide
                    direction = "多" if position_side == 'LONG' else "空"
                    
                # 计算未实现盈亏
                if direction == "多":  # 如果是多头仓位
                    unrealized_pnl = position_amount * (current_price - entry_price)
                else:  # 空仓
                    unrealized_pnl = position_amount * (entry_price - current_price)
                    
                position_value = abs(position_amount * entry_price)
                profit_rate = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0

                logger.info(f"当前持仓方向: {direction}，持仓金额: {abs(float(position['info'].get('notional', 0))):.4f}")
                logger.info(f"当前价格: {current_price}, AI信号: {signal}, "
                          f"持仓数量: {position_amount}, 开仓均价: {entry_price}, "
                          f"未实现盈亏: {unrealized_pnl:.2f} USDT ({profit_rate:.2f}%)")
            else:
                logger.info(f"当前价格: {current_price}, AI信号: {signal}")
        except Exception as e:
            if logger:
                logger.error(f"记录日志信息失败: {str(e)}")
