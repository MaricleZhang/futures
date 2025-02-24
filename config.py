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

# 交易设置
SYMBOLS = ['BERAUSDT']  # 支持多个交易对
SYMBOL_CONFIGS = {
    # 'BTCUSDT': {
    #     'leverage': 5,
    #     'min_notional': 20,
    #     'check_interval': 60,  # 交易检查间隔(秒)
    #     'trade_amount_percent': 50,  # 每次交易使用的资金百分比
    # },
    'BERAUSDT': {
        'leverage': 5,
        'min_notional': 20,
        'trade_amount_percent': 200,
        'check_interval': 60,  # 交易检查间隔(秒)
    },
    # 'SOLUSDT': {
    #     'leverage': 5,
    #     'min_notional': 20,
    #     'trade_amount_percent': 200,
    #     'check_interval': 60,  # 交易检查间隔(秒)
    # },
    # 'BTCUSDT': {
    #     'leverage': 3,
    #     'min_notional': 100,
    #     'trade_amount_percent': 50,
    # },
    # 'ETHUSDT': {
    #     'leverage': 2,
    #     'min_notional': 20,
    #     'trade_amount_percent': 50,
    #     'check_interval': 60,  # 交易检查间隔(秒)
    # }
}
DEFAULT_LEVERAGE = 5  # 默认杠杆倍数
MARGIN_TYPE = 'CROSSED'  # 保证金模式：CROSSED(全仓) ISOLATED(逐仓)
POSITION_SIDE = 'BOTH'  # 持仓模式：BOTH(单向持仓) LONG/SHORT(双向持仓)

# 订单设置
DEFAULT_ORDER_TYPE = 'market'  # 默认订单类型：market(市价) 或 limit(限价)
TIME_IN_FORCE = 'GTC'  # 订单有效期: GTC(永久有效) GTX(立即成交或取消) IOC(立即成交剩余取消)

# 风控设置
MAX_LEVERAGE = 10  # 最大允许杠杆
DEFAULT_STOP_LOSS_PERCENT = 50  # 默认止损百分比
DEFAULT_TAKE_PROFIT_PERCENT = 500  # 默认止盈百分比

# K线设置
DEFAULT_TIMEFRAME = '1m'  # 默认K线周期
DEFAULT_KLINE_LIMIT = 100  # 默认K线获取数量

# API设置
API_TIMEOUT = 10000  # API超时时间(毫秒)
ENABLE_RATE_LIMIT = True  # 是否启用频率限制

# AI策略设置
AI_KLINES_LIMIT = 1000  # AI分析所需的K线数量
MIN_KLINES_FOR_AI = 500  # AI分析所需的最小K线数量
AI_TRADE_AMOUNT_PERCENT = 100  # AI交易金额占可用余额的百分比
AI_TRAIN_INTERVAL = 60  # AI模型训练间隔（分钟）
AI_MIN_TRADE_INTERVAL = 5  # AI最小交易间隔（分钟）

# AI策略配置
DEFAULT_CHECK_INTERVAL = 300  # AI策略检查间隔(秒)
RSI_PERIOD = 14
MA_FAST_PERIOD = 10
MA_SLOW_PERIOD = 20
ATR_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MIN_TRADE_INTERVAL = 3600  # 最小交易间隔(秒)
MAX_TRADES_PER_DAY = 5  # 每日最大交易次数
