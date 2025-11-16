"""
MEXC现货交易配置文件
与Binance合约交易完全隔离
"""

# 代理设置
USE_PROXY = True  # 是否使用代理
PROXY_URL = 'http://127.0.0.1:7890'  # 代理地址
PROXY_TIMEOUT = 10  # 代理超时时间（秒）
PROXY_MAX_RETRIES = 3  # 代理连接最大重试次数
PROXY_TEST_TIMEOUT = 10  # 代理测试超时时间(秒)
PROXY_RETRY_DELAY = 5  # 重试延迟时间(秒)

# MEXC现货交易对设置
# 注意：MEXC现货交易对格式通常是 BTC/USDT, ETH/USDT 等
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # 支持多个现货交易对

# 每个交易对的配置
SYMBOL_CONFIGS = {
    'BTC/USDT': {
        'min_notional': 10,  # 最小交易金额（USDT）
        'trade_amount_percent': 30,  # 单次交易占可用余额的百分比
        'check_interval': 60,  # 检查间隔（秒）
    },
    'ETH/USDT': {
        'min_notional': 10,
        'trade_amount_percent': 30,
        'check_interval': 60,
    },
    'SOL/USDT': {
        'min_notional': 10,
        'trade_amount_percent': 30,
        'check_interval': 60,
    },
}

# 订单设置
DEFAULT_ORDER_TYPE = 'market'  # 默认订单类型：market(市价) 或 limit(限价)
TIME_IN_FORCE = 'GTC'  # 订单有效期: GTC(永久有效) IOC(立即成交剩余取消) FOK(全部成交或取消)

# 挂单策略设置
LIMIT_ORDER_DISTANCE = 0.0002  # 挂单距离当前价格的比例（0.02%）
MAX_ORDER_WAITING_TIME = 30    # 挂单最大等待时间（秒）
PRICE_CHECK_INTERVAL = 5       # 价格检查间隔（秒）

# 风控设置（现货交易）
DEFAULT_STOP_LOSS_PERCENT = 2  # 默认止损百分比
DEFAULT_TAKE_PROFIT_PERCENT = 5  # 默认止盈百分比

# K线设置
DEFAULT_TIMEFRAME = '1m'  # 默认K线周期
DEFAULT_KLINE_LIMIT = 100  # 默认K线获取数量

# API设置
API_TIMEOUT = 10000  # API超时时间(毫秒)
ENABLE_RATE_LIMIT = True  # 是否启用频率限制

# 策略选择配置
# 可选策略: 'simple_adx_di', 'kama_roc_adx'
# 注意：MEXC现货交易不使用DeepSeek策略
STRATEGY_TYPE = 'kama_roc_adx'

# 交易参数
DEFAULT_CHECK_INTERVAL = 60  # 默认检查间隔(秒)
MIN_TRADE_INTERVAL = 300  # 最小交易间隔(秒)，防止频繁交易
MAX_TRADES_PER_DAY = 10  # 每日最大交易次数

# MEXC交易所特定设置
EXCHANGE_ID = 'mexc'  # 交易所ID
SANDBOX_MODE = False  # 是否使用沙盒模式（测试环境）
