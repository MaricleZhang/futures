import os
import ccxt
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from utils.logger import Logger
from utils.exchange import init_exchange, check_exchange_status
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
    def __init__(self, symbol=None):
        """初始化交易器"""
        try:
            # 清除可能的代理设置
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            
            # 初始化日志记录器，添加交易对标识
            self.logger = logging.getLogger()
            if symbol:
                self.logger = logging.getLogger(symbol)
            
            # 初始化交易所
            self.exchange = ccxt.binanceusdm({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000
                },
                'proxies': {
                    'http': config.PROXY_URL,
                    'https': config.PROXY_URL
                } if config.USE_PROXY else None,
                'timeout': config.PROXY_TIMEOUT * 1000 if config.USE_PROXY else 30000,  # 毫秒
            })
            
            # 设置交易对
            self.symbol = symbol
            self.symbol_config = config.SYMBOL_CONFIGS.get(symbol, {})
            
            # 取消所有未完成的订单
            if symbol:
                self.cancel_all_orders()
                
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
        """获取当前持仓"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            positions = self.exchange.fetch_positions([symbol])
            if not positions:
                self.logger.info("当前没有持仓信息")
                return None
                
            for position in positions:
                if position is None or 'info' not in position or 'positionAmt' not in position['info']:
                    continue
                    
                position_amt = float(position['info'].get('positionAmt', 0))
                if abs(position_amt) > 0:
                    return position
            return None
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}")
            raise
            
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
            
            # # 如果没有指定止损价格，使用默认百分比
            # if stop_loss is None and config.DEFAULT_STOP_LOSS_PERCENT > 0:
            #     current_price = self.get_market_price()
            #     stop_loss = current_price * (1 - config.DEFAULT_STOP_LOSS_PERCENT/100) if side.upper() == 'BUY' else current_price * (1 + config.DEFAULT_STOP_LOSS_PERCENT/100)
            
            # # 如果没有指定止盈价格，使用默认百分比
            # if take_profit is None and config.DEFAULT_TAKE_PROFIT_PERCENT > 0:
            #     current_price = self.get_market_price()
            #     take_profit = current_price * (1 + config.DEFAULT_TAKE_PROFIT_PERCENT/100) if side.upper() == 'BUY' else current_price * (1 - config.DEFAULT_TAKE_PROFIT_PERCENT/100)
            
            # # 设置止损单
            # if stop_loss is not None:
            #     stop_loss_params = {
            #         'symbol': symbol,
            #         'type': 'STOP_MARKET',
            #         'side': 'SELL' if side.upper() == 'BUY' else 'BUY',
            #         'amount': amount,
            #         'params': {
            #             'stopPrice': stop_loss,
            #             'workingType': 'MARK_PRICE',
            #         }
            #     }
            #     stop_order = self.exchange.create_order(**stop_loss_params)
            #     self.logger.info(f"止损订单创建成功: {stop_order}")
                
            # # 设置止盈单
            # if take_profit is not None:
            #     take_profit_params = {
            #         'symbol': symbol,
            #         'type': 'TAKE_PROFIT_MARKET',
            #         'side': 'SELL' if side.upper() == 'BUY' else 'BUY',
            #         'amount': amount,
            #         'params': {
            #             'stopPrice': take_profit,
            #             'workingType': 'MARK_PRICE',
            #         }
            #     }
            #     tp_order = self.exchange.create_order(**take_profit_params)
            #     self.logger.info(f"止盈订单创建成功: {tp_order}")
                
            return order
            
        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
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
                
            # 确定平仓方向
            side = 'sell' if float(position['info'].get('positionAmt', 0)) > 0 else 'buy'
            amount = abs(float(position['info'].get('positionAmt', 0)))
            
            # 市价平仓
            order = self.place_order(symbol, side, amount)
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
            limit: 获取的K线数量，最大1500
        
        Returns:
            K线数据列表，每个K线包含 [timestamp, open, high, low, close, volume]
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            # 限制最大获取数量
            if limit > 1500:
                limit = 1500
                self.logger.warning("K线数量超过最大限制1500，已自动调整")
                
            # 获取K线数据
            klines = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval,
                limit=limit,
                params={'type': 'future'}  # 指定为期货
            )
            
            if klines and len(klines) > 0:
                # 转换数据格式
                formatted_klines = []
                for k in klines:
                    formatted_klines.append([
                        int(k[0]),  # timestamp
                        float(k[1]),  # open
                        float(k[2]),  # high
                        float(k[3]),  # low
                        float(k[4]),  # close
                        float(k[5])   # volume
                    ])
                self.logger.info(f"成功获取{len(formatted_klines)}根{interval}K线数据")
                return formatted_klines
            else:
                self.logger.error(f"获取{interval}K线数据失败")
                return []
                
        except Exception as e:
            self.logger.error(f"获取K线数据时出错: {str(e)}")
            return []
        
    def open_long(self, symbol=None, amount=None):
        """开多仓"""
        return self.place_order(symbol, 'buy', amount)
        
    def open_short(self, symbol=None, amount=None):
        """开空仓"""
        return self.place_order(symbol, 'sell', amount)
        
    def close_long(self, symbol=None, amount=None):
        """平多仓"""
        return self.place_order(symbol, 'sell', amount)
        
    def close_short(self, symbol=None, amount=None):
        """平空仓"""
        return self.place_order(symbol, 'buy', amount)
        
    @retry_on_error()
    def set_position_mode(self, symbol=None, dual_side_position=False):
        """设置持仓模式
        Args:
            symbol: 交易对
            dual_side_position (bool): True为双向持仓，False为单向持仓
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")
            self.exchange.fapiPrivatePostPositionSideDual({
                'dualSidePosition': 'true' if dual_side_position else 'false'
            })
            self.logger.info(f"设置{'双向' if dual_side_position else '单向'}持仓模式成功")
        except Exception as e:
            # 如果已经是目标模式，会抛出异常，这种情况可以忽略
            if 'No need to change position side' in str(e):
                self.logger.info(f"当前已经是{'双向' if dual_side_position else '单向'}持仓模式")
            else:
                self.logger.error(f"设置持仓模式失败: {str(e)}")
                raise

    @retry_on_error()
    def set_leverage(self, leverage=None):
        """设置杠杆倍数
        Args:
            leverage (int): 杠杆倍数
        """
        try:
            if not self.symbol:
                raise ValueError("Symbol not specified")
                
            # 如果没有指定杠杆倍数，使用配置中的默认值
            if leverage is None:
                if self.symbol in config.SYMBOL_CONFIGS:
                    leverage = config.SYMBOL_CONFIGS[self.symbol].get('leverage', config.DEFAULT_LEVERAGE)
                else:
                    leverage = config.DEFAULT_LEVERAGE
                
            self.exchange.fapiPrivatePostLeverage({
                'symbol': self.exchange.market(self.symbol)['id'],
                'leverage': leverage
            })
            self.logger.info(f"设置杠杆倍数 {leverage}倍 成功")
        except Exception as e:
            self.logger.error(f"设置杠杆倍数失败: {str(e)}")
            raise

    @retry_on_error()
    def set_margin_type(self, margin_type=None):
        """设置保证金模式
        Args:
            margin_type (str): 'ISOLATED' 或 'CROSSED'
        """
        try:
            if not self.symbol:
                raise ValueError("Symbol not specified")
            if not margin_type:
                raise ValueError("Margin type not specified")
            self.exchange.fapiPrivatePostMarginType({
                'symbol': self.exchange.market(self.symbol)['id'],
                'marginType': margin_type.upper()
            })
            self.logger.info(f"设置{margin_type}保证金模式成功")
        except Exception as e:
            # 如果已经是目标模式，会抛出异常，这种情况可以忽略
            if 'No need to change margin type' in str(e):
                self.logger.info(f"当前已经是{margin_type}保证金模式")
            else:
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
