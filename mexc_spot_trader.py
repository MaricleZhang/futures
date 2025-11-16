import os
import ccxt
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import config_mexc_spot as config
import logging
import time
from functools import wraps

# 加载MEXC环境变量
load_dotenv('.env.mexc')

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

class MexcSpotTrader:
    """MEXC现货交易器 - 与Binance合约交易完全隔离"""

    def __init__(self, symbol=None):
        """初始化MEXC现货交易器"""
        try:
            # 清除可能的代理设置
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)

            # 初始化日志记录器，添加交易对标识
            self.logger = logging.getLogger()
            if symbol:
                self.logger = logging.getLogger(f"MEXC_{symbol}")

            # 初始化MEXC交易所
            self.exchange = self._init_mexc_exchange()

            # 设置交易对
            self.symbol = symbol
            self.symbol_config = config.SYMBOL_CONFIGS.get(symbol, {})

            self.logger.info(f"MEXC现货交易器初始化成功 {'for ' + symbol if symbol else 'for all symbols'}")

        except Exception as e:
            self.logger.error(f"MEXC现货交易器初始化失败: {str(e)}")
            raise

    def _init_mexc_exchange(self):
        """初始化MEXC交易所连接"""
        try:
            # 从环境变量获取API密钥
            api_key = os.getenv('MEXC_API_KEY')
            secret_key = os.getenv('MEXC_SECRET_KEY')

            if not api_key or not secret_key:
                raise ValueError("MEXC API密钥未设置，请在.env.mexc文件中配置")

            # 创建交易所实例
            exchange_params = {
                'apiKey': api_key,
                'secret': secret_key,
                'timeout': config.API_TIMEOUT,
                'enableRateLimit': config.ENABLE_RATE_LIMIT,
                'options': {
                    'defaultType': 'spot',  # 设置为现货交易
                }
            }

            # 如果启用代理
            if config.USE_PROXY and config.PROXY_URL:
                exchange_params['proxies'] = {
                    'http': config.PROXY_URL,
                    'https': config.PROXY_URL
                }
                self.logger.info(f"使用代理: {config.PROXY_URL}")

            # 创建MEXC交易所实例
            exchange = ccxt.mexc(exchange_params)

            # 测试连接
            exchange.load_markets()
            self.logger.info("MEXC现货交易所连接成功")

            return exchange

        except Exception as e:
            self.logger.error(f"初始化MEXC交易所失败: {str(e)}")
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
    def get_balance(self, currency='USDT'):
        """获取账户余额

        Args:
            currency: 货币类型，默认USDT

        Returns:
            dict: 包含free(可用)和total(总额)的余额信息
        """
        try:
            balance = self.exchange.fetch_balance()
            currency_balance = balance.get(currency, {'free': 0, 'total': 0})
            self.logger.info(f"{currency}余额: 可用={currency_balance['free']}, 总额={currency_balance['total']}")
            return currency_balance
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {str(e)}")
            raise

    @retry_on_error()
    def get_position(self, symbol=None):
        """获取当前持仓（现货持仓）

        Args:
            symbol: 交易对，如 'BTC/USDT'

        Returns:
            dict: 持仓信息，包含币种数量
        """
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")

            # 从交易对中提取基础货币（如BTC/USDT中的BTC）
            base_currency = symbol.split('/')[0]

            balance = self.exchange.fetch_balance()
            base_balance = balance.get(base_currency, {'free': 0, 'total': 0})

            if base_balance['total'] > 0:
                self.logger.info(f"{base_currency}持仓: {base_balance['total']}")
                return {
                    'currency': base_currency,
                    'amount': base_balance['total'],
                    'free': base_balance['free'],
                    'used': base_balance['used']
                }
            else:
                self.logger.info(f"当前没有{base_currency}持仓")
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

            # 获取K线数据
            klines = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval,
                limit=limit
            )

            if klines and len(klines) > 0:
                # 转换数据格式
                formatted_klines = []
                for k in klines:
                    formatted_klines.append([
                        int(k[0]),    # timestamp
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
    def place_order(self, symbol=None, side=None, amount=None, order_type=None, price=None):
        """下单函数

        Args:
            symbol: 交易对
            side: 'buy' 或 'sell'
            amount: 下单数量（币的数量）
            order_type: 订单类型，默认为配置中的类型
            price: 限价单价格

        Returns:
            订单信息
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

            # 检查下单金额是否满足最小要求
            current_price = self.get_market_price(symbol)
            min_notional = self.symbol_config.get('min_notional', 10)

            if amount * current_price < min_notional:
                raise ValueError(f"下单金额 {amount * current_price} USDT 小于最小名义价值 {min_notional} USDT")

            # 创建订单参数
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
            }

            # 如果是限价单，添加价格
            if order_type == 'limit' and price is not None:
                order_params['price'] = price

            # 下单
            order = self.exchange.create_order(**order_params)
            self.logger.info(f"订单创建成功: {order['id']}, {side} {amount} {symbol} @ {current_price}")

            return order

        except Exception as e:
            self.logger.error(f"下单失败: {str(e)}")
            raise

    def buy(self, symbol=None, amount=None):
        """买入（现货）"""
        return self.place_order(symbol, 'buy', amount)

    def sell(self, symbol=None, amount=None):
        """卖出（现货）"""
        return self.place_order(symbol, 'sell', amount)

    @retry_on_error()
    def sell_all(self, symbol=None):
        """卖出所有持仓"""
        try:
            symbol = symbol or self.symbol
            if not symbol:
                raise ValueError("Symbol not specified")

            position = self.get_position(symbol)
            if position is None or position['free'] <= 0:
                self.logger.info("当前没有可用持仓")
                return None

            # 卖出所有可用数量
            amount = position['free']
            order = self.sell(symbol, amount)
            self.logger.info(f"卖出所有持仓成功: {amount} {position['currency']}")
            return order

        except Exception as e:
            self.logger.error(f"卖出所有持仓失败: {str(e)}")
            raise

    @retry_on_error()
    def cancel_all_orders(self, symbol=None):
        """取消所有未完成的订单"""
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
    def get_all_symbols(self):
        """获取所有可交易的现货交易对

        Returns:
            list: 交易对列表
        """
        try:
            # 获取市场信息
            markets = self.exchange.load_markets()
            # 只获取现货交易对（通常是币币交易）
            symbols = [symbol for symbol in markets.keys()
                      if markets[symbol]['spot'] and markets[symbol]['active']]
            self.logger.info(f"获取到 {len(symbols)} 个现货交易对")
            return symbols
        except Exception as e:
            self.logger.error(f"获取交易对失败: {str(e)}")
            raise
