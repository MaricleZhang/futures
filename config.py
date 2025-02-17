"""
交易配置文件
"""

# 代理设置
USE_PROXY = True  # 是否使用代理
PROXY_URL = 'http://127.0.0.1:7890'  # 代理地址
PROXY_TIMEOUT = 10  # 代理超时时间（秒）
PROXY_MAX_RETRIES = 3  # 代理连接最大重试次数
PROXY_TEST_TIMEOUT = 10  # 代理测试超时时间(秒)
PROXY_RETRY_DELAY = 5  # 重试延迟时间(秒)

# 请求头设置
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# 交易设置
SYMBOL = 'LAYERUSDT'  # 交易对
LEVERAGE = 5  # 杠杆倍数
MARGIN_TYPE = 'CROSSED'  # 保证金模式：CROSSED(全仓) ISOLATED(逐仓)
POSITION_SIDE = 'BOTH'  # 持仓模式：BOTH(单向持仓) LONG/SHORT(双向持仓)
MIN_NOTIONAL = 5.0  # 最小名义价值（USDT）

# 订单设置
DEFAULT_ORDER_TYPE = 'market'  # 默认订单类型：market(市价) 或 limit(限价)
TIME_IN_FORCE = 'GTC'  # 订单有效期: GTC(永久有效) GTX(立即成交或取消) IOC(立即成交剩余取消)

# 风控设置
MAX_POSITION_SIZE = 200  # 最大持仓量(BTC)
MAX_LEVERAGE = 10  # 最大允许杠杆
DEFAULT_STOP_LOSS_PERCENT = 50  # 默认止损百分比
DEFAULT_TAKE_PROFIT_PERCENT = 10000  # 默认止盈百分比

# K线设置
DEFAULT_TIMEFRAME = '1m'  # 默认K线周期
DEFAULT_KLINE_LIMIT = 100  # 默认K线获取数量

# API设置
API_TIMEOUT = 10000  # API超时时间(毫秒)
ENABLE_RATE_LIMIT = True  # 是否启用频率限制

# 交易所设置
EXCHANGE_ID = 'binance'  # 交易所ID
EXCHANGE_OPTIONS = {
    'defaultType': 'future',  # 使用期货模式
    'adjustForTimeDifference': True,  # 调整服务器时间差
    'recvWindow': 5000,  # 请求有效时间窗口(毫秒)
    'urls': {
        'api': {
            'public': 'https://fapi.binance.com/fapi/v1',
            'private': 'https://fapi.binance.com/fapi/v1',
        }
    }
}

# AI策略配置
DEFAULT_CHECK_INTERVAL = 60  # AI策略检查间隔(秒)
AI_TRADE_AMOUNT_PERCENT = 50  # AI交易每次使用账户余额的百分比
AI_KLINES_LIMIT = 200  # AI策略使用的K线数量，建议范围：100-500
MIN_KLINES_FOR_AI = 100  # AI策略所需的最小K线数量
RETRAIN_INTERVAL = 100  # 每多少根K线重新训练一次模型
