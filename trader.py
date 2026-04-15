import os
import ccxt
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from utils.logger import Logger
from utils.exchange import init_exchange, check_exchange_status
from utils.exchange_adapter import make_adapter
from utils.trade_recorder import TradeRecorder
import config
import logging
import time
from functools import wraps

# 加载环境变量
load_dotenv()

def retry_on_error(max_retries=None, retry_delay=None):
    """重试装饰器，用于处理API调用失败的情况"""
    max_retries = max_retries or config.PROXY_MAX_RETRIES
    retry_delay = retry_delay or config.PROXY_RETRY_DELAY
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        self.logger.warning(f"{func.__name__} 调用失败，{attempt + 1}/{max_retries}次尝试: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        self.logger.error(f"{func.__name__} 调用失败，已达到最大重试次数: {str(e)}")
            raise last_error
        return wrapper
    return decorator

class Trader:
    def __init__(self, symbol=None, exchange_name='binance'):
        """初始化交易器

        Args:
            symbol: 交易对（允许多种输入格式，例如 'BTCUSDT' / 'BTC/USDT' / 'BTC/USDT:USDT'，
                    内部会通过 adapter 归一化为当前交易所可用的形式）
            exchange_name: 'binance' 或 'gate'，默认 'binance' 保持向下兼容
        """
        try:
            # 清除可能的代理设置
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)

            self.exchange_name = (exchange_name or 'binance').lower()

            # 初始化日志记录器，添加交易所 + 交易对标识
            self.logger = logging.getLogger()
            if symbol:
                self.logger = logging.getLogger(f"{self.exchange_name}.{symbol}")

            # 初始化交易所 + 适配器
            self.exchange = init_exchange(self.exchange_name)
            self.adapter = make_adapter(self.exchange_name, self.exchange)

            # 设置交易对（统一归一化）
            self.symbol = self.adapter.normalize_symbol(symbol) if symbol else None
            self.symbol_config = config.SYMBOL_CONFIGS.get(symbol, {})
            
            # 初始化交易记录器
            self.trade_recorder = TradeRecorder()
            
            # 取消所有未完成的订单
            if symbol:                
                # 设置持仓模式为单向持仓
                self.set_position_mode(False)
                
                # 设置杠杆倍数
                if self.symbol in config.SYMBOL_CONFIGS:
                    leverage = config.SYMBOL_CONFIGS[self.symbol].get('leverage', config.DEFAULT_LEVERAGE)
                else:
                    leverage = config.DEFAULT_LEVERAGE
                self.set_leverage(leverage)
                
                # 设置保证金模式
                self.set_margin_type(config.MARGIN_TYPE)
            
            self.logger.info(f"交易器初始化成功 {'for ' + symbol if symbol else 'for all symbols'}")
            
        except Exception as e:
            self.logger.error(f"交易器初始化失败: {str(e)}")
            raise
            
    @retry_on_error()
    def get_market_price(self, symbol=None):
        """获取当前市场价格"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"获取市场价格失败: {str(e)}")
            raise
            
    @retry_on_error()
    def get_balance(self, symbol=None):
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']
            self.logger.info(f"USDT余额: 可用={usdt_balance['free']}, 总额={usdt_balance['total']}")
            return usdt_balance
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {str(e)}")
            raise
            
    @retry_on_error()
    def get_position(self, symbol=None):
        """获取当前持仓（多交易所统一：通过 adapter 解析 position 结构）"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            positions = self.exchange.fetch_positions([symbol])
            if not positions:
                self.logger.info("当前没有持仓信息")
                return None

            for position in positions:
                if position is None:
                    continue
                position_amt = self.adapter.get_position_amt(position)
                if abs(position_amt) > 0:
                    return position
            return None
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}")
            raise

    @retry_on_error()
    def fetch_funding_rate(self, symbol=None):
        """获取当前交易对的资金费率 / 下次结算时间 / 结算周期（统一结构）"""
        symbol = symbol or self.symbol
        if not symbol:
            raise ValueError("Symbol not specified")
        return self.adapter.fetch_funding_rate(symbol)
            
    @retry_on_error()
    def get_market_info(self, symbol=None):
        """获取交易对信息"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            market = self.exchange.market(symbol)
            return market
        except Exception as e:
            self.logger.error(f"获取交易对信息失败: {str(e)}")
            raise
            
    @retry_on_error()
    def fetch_klines(self, symbol=None, timeframe=None, limit=None):
        """获取K线数据，带重试机制"""
        symbol = symbol or self.symbol
        if not symbol:
            raise ValueError("Symbol not specified")
        timeframe = timeframe or config.DEFAULT_TIMEFRAME
        limit = limit or config.DEFAULT_KLINE_LIMIT
        
        try:
            klines = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {symbol} {timeframe}, 错误: {str(e)}")
            raise
            
    @retry_on_error()
    def check_order_amount(self, symbol=None, amount=None):
        """检查下单数量是否符合要求"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            if not amount:
                raise ValueError("Amount not specified")
            market = self.get_market_info(symbol)
            min_amount = market['limits']['amount']['min']
            
            if amount < min_amount:
                self.logger.warning(f"下单数量 {amount} 小于最小下单数量 {min_amount}，将使用最小下单数量")
                return min_amount
            return amount
        except Exception as e:
            self.logger.error(f"检查下单数量失败: {str(e)}")
            raise
            
    def _cancel_pending_limit_orders(self, symbol=None):
        """
        取消指定交易对的所有挂单（包括限价单、止损单、止盈单等）
        
        Args:
            symbol: 交易对
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
                
            # 获取所有未完成的订单
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            
            if not open_orders:
                return
            
            if not open_orders:
                self.logger.debug(f"[{symbol}] 没有需要取消的挂单")
                return
                
            self.logger.info(f"[{symbol}] 检测到 {len(open_orders)} 个挂单，准备全部取消...")
            
            # 取消所有挂单
            cancelled_count = 0
            for order in open_orders:
                try:
                    order_id = order['id']
                    order_type = order.get('type', 'N/A')
                    order_side = order.get('side', 'N/A')
                    order_price = order.get('price', order.get('stopPrice', 'N/A'))
                    self.exchange.cancel_order(order_id, symbol)
                    self.logger.info(f"✓ 取消挂单: 类型={order_type}, ID={order_id}, 方向={order_side}, 价格={order_price}")
                    cancelled_count += 1
                except Exception as e:
                    self.logger.error(f"✗ 取消订单失败 {order.get('id', 'N/A')}: {str(e)}")
            
            if cancelled_count > 0:
                self.logger.info(f"[{symbol}] 成功取消 {cancelled_count} 个挂单")
                
        except Exception as e:
            self.logger.error(f"取消挂单时出错: {str(e)}")
            # 不抛出异常，避免影响后续下单流程
    
    @retry_on_error()
    def place_order(self, symbol=None, side=None, amount=None, order_type=None, price=None, stop_loss=None, take_profit=None):
        """
        下单函数
        :param symbol: 交易对
        :param side: 'buy' 或 'sell'
        :param amount: 下单数量
        :param order_type: 订单类型，默认为配置中的类型
        :param price: 限价单价格
        :param stop_loss: 止损价格
        :param take_profit: 止盈价格
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            if not side:
                raise ValueError("Side not specified")
            if not amount:
                raise ValueError("Amount not specified")
            if order_type is None:
                order_type = config.DEFAULT_ORDER_TYPE
            
            # 下单前先取消该交易对的所有限价挂单
            self._cancel_pending_limit_orders(symbol)
                
            # 检查并调整下单数量
            amount = self.check_order_amount(symbol, amount)
                
            current_price = self.get_market_price(symbol)
            # 检查下单数量是否小于最小名义价值
            min_notional = self.symbol_config.get('min_notional', 20)  # 默认值为20
            if amount * current_price < min_notional:
                raise ValueError(f"下单金额 {amount * current_price} USDT 小于最小名义价值 {min_notional} USDT")

            # 创建订单
            params = {
                'timeInForce': config.TIME_IN_FORCE,
            }
            
            # 基础订单参数
            order_params = {
                'symbol': symbol,
                'type': order_type.upper(),
                'side': side.upper(),
                'amount': amount,
            }
            
            # 如果是限价单，添加价格
            if order_type == 'limit' and price is not None:
                order_params['price'] = price
                
            # 下单
            order = self.exchange.create_order(**order_params)
            self.logger.info(f"订单创建成功: {order}")
            
            # 如果没有指定止损价格，使用默认百分比
            if stop_loss is None and config.DEFAULT_STOP_LOSS_PERCENT > 0:
                stop_loss = current_price * (1 - config.DEFAULT_STOP_LOSS_PERCENT/100) if side.upper() == 'BUY' else current_price * (1 + config.DEFAULT_STOP_LOSS_PERCENT/100)
            
            # 如果没有指定止盈价格，使用默认百分比
            if take_profit is None and config.DEFAULT_TAKE_PROFIT_PERCENT > 0:
                take_profit = current_price * (1 + config.DEFAULT_TAKE_PROFIT_PERCENT/100) if side.upper() == 'BUY' else current_price * (1 - config.DEFAULT_TAKE_PROFIT_PERCENT/100)
            
            # 设置止损单
            if stop_loss is not None:
                stop_loss_params = {
                    'symbol': symbol,
                    'type': 'STOP_MARKET',
                    'side': 'SELL' if side.upper() == 'BUY' else 'BUY',
                    'amount': amount,
                    'params': {
                        'stopPrice': stop_loss,
                        'workingType': 'MARK_PRICE',
                    }
                }
                stop_order = self.exchange.create_order(**stop_loss_params)
                self.logger.info(f"🛑 止损订单创建成功: 止损价={stop_loss:.6f}, 订单ID={stop_order.get('id', 'N/A')}")
                
            # 设置止盈单
            if take_profit is not None:
                take_profit_params = {
                    'symbol': symbol,
                    'type': 'TAKE_PROFIT_MARKET',
                    'side': 'SELL' if side.upper() == 'BUY' else 'BUY',
                    'amount': amount,
                    'params': {
                        'stopPrice': take_profit,
                        'workingType': 'MARK_PRICE',
                    }
                }
                tp_order = self.exchange.create_order(**take_profit_params)
                self.logger.info(f"🎯 止盈订单创建成功: 止盈价={take_profit:.6f}, 订单ID={tp_order.get('id', 'N/A')}")
                
            return order
            
        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
            raise
    
    def place_limit_order_with_fallback(self, symbol=None, side=None, amount=None, stop_loss=None, take_profit=None):
        """
        先尝试挂单(maker)，超时未成交则改用市价单(taker)
        
        Args:
            symbol: 交易对
            side: 'buy' 或 'sell'
            amount: 下单数量
            stop_loss: 止损价格（可选）
            take_profit: 止盈价格（可选）
            
        Returns:
            最终成交的订单信息
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            if not side:
                raise ValueError("Side not specified")
            if not amount:
                raise ValueError("Amount not specified")
            
            # 下单前先取消该交易对的所有限价挂单
            self._cancel_pending_limit_orders(symbol)
            
            # 获取当前市场价格
            current_price = self.get_market_price(symbol)
            
            # 计算挂单价格
            # 买入时略高于当前价（更容易成交），卖出时略低于当前价
            if side.lower() == 'buy':
                limit_price = current_price * (1 + config.LIMIT_ORDER_DISTANCE)
            else:  # sell
                limit_price = current_price * (1 - config.LIMIT_ORDER_DISTANCE)
            
            # 提交限价单
            self.logger.info(f"尝试挂单 {side.upper()}: 数量={amount}, 挂单价={limit_price:.6f}, 当前价={current_price:.6f}")
            order = self.place_order(symbol, side, amount, order_type='limit', price=limit_price, stop_loss=stop_loss, take_profit=take_profit)
            order_id = order['id']
            
            # 循环检查订单状态
            start_time = time.time()
            max_wait_time = config.MAX_ORDER_WAITING_TIME
            check_interval = config.PRICE_CHECK_INTERVAL
            
            while True:
                elapsed_time = time.time() - start_time
                
                # 检查是否超时
                if elapsed_time >= max_wait_time:
                    self.logger.warning(f"挂单等待超时({max_wait_time}秒)，准备切换为市价单")
                    break
                
                # 等待一段时间后再检查
                time.sleep(check_interval)
                
                # 查询订单状态
                try:
                    order_status = self.exchange.fetch_order(order_id, symbol)
                    status = order_status['status']
                    
                    self.logger.debug(f"订单状态检查: {status}, 已等待 {elapsed_time:.1f}秒")
                    
                    if status == 'closed':
                        # 订单已成交
                        self.logger.info(f"✓ 挂单成交成功! 成交价={order_status.get('average', limit_price):.6f}")
                        return order_status
                    elif status == 'canceled':
                        # 订单被取消（可能被用户或其他程序取消）
                        self.logger.warning("挂单被取消，切换为市价单")
                        break
                    elif status == 'open':
                        # 订单仍在等待成交
                        continue
                    else:
                        # 其他状态（如rejected等）
                        self.logger.warning(f"挂单状态异常: {status}，切换为市价单")
                        break
                        
                except Exception as e:
                    self.logger.error(f"查询订单状态失败: {str(e)}")
                    # 继续等待，不立即放弃
                    continue
            
            # 超时或失败，取消挂单并使用市价单
            try:
                self.logger.info(f"取消挂单 {order_id}")
                self.exchange.cancel_order(order_id, symbol)
                self.logger.info("挂单已取消")
            except Exception as e:
                self.logger.warning(f"取消挂单失败（订单可能已成交或已取消）: {str(e)}")
            
            # 使用市价单确保成交
            self.logger.info(f"使用市价单 {side.upper()}: 数量={amount}")
            market_order = self.place_order(symbol, side, amount, order_type='market', stop_loss=stop_loss, take_profit=take_profit)
            self.logger.info(f"✓ 市价单成交成功!")
            
            return market_order
            
        except Exception as e:
            self.logger.error(f"智能下单失败: {str(e)}")
            raise
            
    @retry_on_error()
    def close_position(self, symbol=None):
        """平掉当前所有持仓"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            position = self.get_position(symbol)
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                self.logger.info("当前没有持仓")
                return
                
            # 确定平仓方向和持仓方向
            position_amt = float(position['info'].get('positionAmt', 0))
            side = 'sell' if position_amt > 0 else 'buy'
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            amount = abs(position_amt)
            
            # 市价平仓
            order = self.place_order(symbol, side, amount)
            
            # 记录平仓信息
            try:
                close_price = self.get_market_price(symbol)
                trade_info = self.trade_recorder.record_close_position(symbol, position_side, close_price)
                if trade_info:
                    self.logger.info(f"平仓记录成功: 盈亏 {trade_info['profit_loss']:.2f} USDT ({trade_info['profit_rate']:.2f}%)")
            except Exception as e:
                self.logger.error(f"记录平仓信息失败: {str(e)}")
            
            self.logger.info(f"平仓成功: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"平仓失败: {str(e)}")
            raise
            
    @retry_on_error()
    def cancel_all_orders(self, symbol=None):
        """
        取消所有未完成的订单（包括止损止盈委托）
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            # 获取所有未完成的订单
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            
            # 取消每个订单
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['id'], symbol)
                    self.logger.info(f"取消订单成功: {order['id']}")
                except Exception as e:
                    self.logger.error(f"取消订单失败 {order['id']}: {str(e)}")
            
            if not open_orders:
                self.logger.info("没有未成交的订单")
                
        except Exception as e:
            self.logger.error(f"取消所有订单失败: {str(e)}")
            raise
            
    @retry_on_error()
    def get_klines(self, symbol=None, interval='1m', limit=100):
        """获取K线数据
        
        Args:
            symbol: 交易对
            interval: K线周期，支持 1m, 5m, 15m, 1h, 4h, 1d
            limit: 获取的K线数量
        
        Returns:
            K线数据列表，每个K线包含 [timestamp, open, high, low, close, volume]
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
                
            # 解析时间周期为毫秒
            timeframe_duration_seconds = self.exchange.parse_timeframe(interval)
            timeframe_duration_ms = timeframe_duration_seconds * 1000
            
            all_klines = []
            remaining_limit = limit
            
            # 如果请求的数量超过单次最大限制（通常是1000或1500），需要分批获取
            # 这里我们设置单次获取的最大数量为1000，以确保兼容性
            BATCH_LIMIT = 1000
            
            # 计算起始时间
            # 当前时间 - 需要获取的K线数量 * K线周期
            end_time = self.exchange.milliseconds()
            start_time = end_time - (limit * timeframe_duration_ms)
            
            current_since = start_time
            
            while remaining_limit > 0:
                fetch_limit = min(remaining_limit, BATCH_LIMIT)
                
                # 获取K线数据
                klines = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=interval,
                    limit=fetch_limit,
                    since=int(current_since),
                    params={'type': 'future'}  # 指定为期货
                )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                remaining_limit -= len(klines)
                
                # 更新下一次获取的起始时间
                # 取最后一条K线的时间 + 1个周期
                last_kline_time = klines[-1][0]
                current_since = last_kline_time + timeframe_duration_ms
                
                # 如果获取的数量少于请求的数量，说明已经获取到了最新的数据
                if len(klines) < fetch_limit:
                    break
                    
                # 防止请求过快
                time.sleep(0.1)
            
            if all_klines:
                # 转换数据格式
                formatted_klines = []
                for k in all_klines:
                    formatted_klines.append([
                        int(k[0]),  # timestamp
                        float(k[1]),  # open
                        float(k[2]),  # high
                        float(k[3]),  # low
                        float(k[4]),  # close
                        float(k[5])   # volume
                    ])
                
                # 确保返回的数量不超过请求的数量
                if len(formatted_klines) > limit:
                    formatted_klines = formatted_klines[-limit:]
                    
                self.logger.info(f"成功获取{len(formatted_klines)}根{interval}K线数据")
                return formatted_klines
            else:
                self.logger.error(f"获取{interval}K线数据失败")
                return []
                
        except Exception as e:
            self.logger.error(f"获取K线数据时出错: {str(e)}")
            return []
        
    def open_long(self, symbol=None, amount=None, stop_loss_pct=None, take_profit_pct=None):
        """开多仓
        
        Args:
            symbol: 交易对
            amount: 开仓数量
            stop_loss_pct: 止损百分比（可选），默认使用配置中的值
            take_profit_pct: 止盈百分比（可选），默认使用配置中的值
        """
        symbol = symbol or self.symbol
        current_price = self.get_market_price(symbol)
        
        # 计算止损止盈价格
        stop_loss = None
        take_profit = None
        
        if stop_loss_pct is not None and stop_loss_pct > 0:
            stop_loss = current_price * (1 - stop_loss_pct / 100)
            
        if take_profit_pct is not None and take_profit_pct > 0:
            take_profit = current_price * (1 + take_profit_pct / 100)
        
        # 使用挂单策略开仓，并设置止损止盈
        order = self.place_limit_order_with_fallback(symbol, 'buy', amount, stop_loss=stop_loss, take_profit=take_profit)
        
        # 记录开仓信息
        try:
            leverage = self.symbol_config.get('leverage', config.DEFAULT_LEVERAGE)
            self.trade_recorder.record_open_position(symbol, 'LONG', amount, current_price, leverage, config.STRATEGY_TYPE)
        except Exception as e:
            self.logger.error(f"记录开多仓信息失败: {str(e)}")
        return order
        
    def open_short(self, symbol=None, amount=None, stop_loss_pct=None, take_profit_pct=None):
        """开空仓
        
        Args:
            symbol: 交易对
            amount: 开仓数量
            stop_loss_pct: 止损百分比（可选），默认使用配置中的值
            take_profit_pct: 止盈百分比（可选），默认使用配置中的值
        """
        symbol = symbol or self.symbol
        current_price = self.get_market_price(symbol)
        
        # 计算止损止盈价格
        stop_loss = None
        take_profit = None
        
        if stop_loss_pct is not None and stop_loss_pct > 0:
            stop_loss = current_price * (1 + stop_loss_pct / 100)
            
        if take_profit_pct is not None and take_profit_pct > 0:
            take_profit = current_price * (1 - take_profit_pct / 100)
        
        # 使用挂单策略开仓，并设置止损止盈
        order = self.place_limit_order_with_fallback(symbol, 'sell', amount, stop_loss=stop_loss, take_profit=take_profit)
        
        # 记录开仓信息
        try:
            leverage = self.symbol_config.get('leverage', config.DEFAULT_LEVERAGE)
            self.trade_recorder.record_open_position(symbol, 'SHORT', amount, current_price, leverage, config.STRATEGY_TYPE)
        except Exception as e:
            self.logger.error(f"记录开空仓信息失败: {str(e)}")
        return order
        
    def close_long(self, symbol=None, amount=None):
        """平多仓"""
        return self.place_order(symbol, 'sell', amount)
        
    def close_short(self, symbol=None, amount=None):
        """平空仓"""
        return self.place_order(symbol, 'buy', amount)
        
    @retry_on_error()
    def set_position_mode(self, symbol=None, dual_side_position=False):
        """设置持仓模式（通过 adapter 适配各交易所差异）"""
        try:
            self.adapter.set_position_mode(bool(dual_side_position))
            self.logger.info(f"[{self.exchange_name}] 设置{'双向' if dual_side_position else '单向'}持仓模式成功")
        except Exception as e:
            self.logger.error(f"设置持仓模式失败: {str(e)}")
            raise

    @retry_on_error()
    def set_leverage(self, leverage=None):
        """设置杠杆倍数（通过 adapter 适配各交易所差异）"""
        try:
            if not self.symbol:
                raise ValueError("Symbol not specified")

            if leverage is None:
                if self.symbol in config.SYMBOL_CONFIGS:
                    leverage = config.SYMBOL_CONFIGS[self.symbol].get('leverage', config.DEFAULT_LEVERAGE)
                else:
                    leverage = config.DEFAULT_LEVERAGE

            self.adapter.set_leverage(self.symbol, int(leverage))
            self.logger.info(f"[{self.exchange_name}] 设置杠杆倍数 {leverage}倍 成功")
        except Exception as e:
            self.logger.error(f"设置杠杆倍数失败: {str(e)}")
            raise

    @retry_on_error()
    def set_margin_type(self, margin_type=None):
        """设置保证金模式（通过 adapter 适配各交易所差异）"""
        try:
            if not self.symbol:
                raise ValueError("Symbol not specified")
            if not margin_type:
                raise ValueError("Margin type not specified")
            self.adapter.set_margin_mode(self.symbol, margin_type)
            self.logger.info(f"[{self.exchange_name}] 设置 {margin_type} 保证金模式成功")
        except Exception as e:
            self.logger.error(f"设置保证金模式失败: {str(e)}")
            raise

    @retry_on_error()
    def get_all_symbols(self):
        """获取所有可交易的合约交易对
        
        Returns:
            list: 交易对列表
        """
        try:
            # 获取市场信息
            markets = self.exchange.load_markets()
            # 只获取USDT合约交易对
            symbols = [symbol for symbol in markets.keys() if symbol.endswith('USDT')]
            self.logger.info(f"获取到 {len(symbols)} 个USDT合约交易对")
            return symbols
        except Exception as e:
            self.logger.error(f"获取交易对失败: {str(e)}")
            raise
