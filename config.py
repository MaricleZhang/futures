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

# 1000CHEEMSUSDT BMTUSDT API3USDT ARCUSDT TUTUSDT MUBARAKUSDT DOGEUSDT AUCTIONUSDT  EOSUSDT 1000SATSUSDT
# GUNUSDT TNSRUSDT PARTIUSDT DUSDT GASUSDT JELLYJELLYUSDT TAOUSDT AEROUSDT EOSUSDT TONUSDT REDUSDT ROSEUSDT
# BNBUSDT BTCDOMUSDT XRPUSDC DUSDT GASUSDT BABYUSDT  AVAAIUSDT NKNUSDT PAXGUSDT VOXELUSDT
 # BTCUSDC SOLUSDC 1000PEPEUSDC AINIMEUSDT ARBUSDC SQDUSDT SPKUSDT  FUSDT SAHARAUSDT UNIUSDC PROVEUSDT 
# XNYUSDT  AIOTUSDT ARIAUSST IMXUSDT WUSDT  ETHUSDC
# 交易设置


SYMBOLS = ['ETHUSDC']  # 支持多个交易对
SYMBOL_CONFIGS = {
    'ETHUSDC': {
        'leverage':10,
        'min_notional': 20,
        'trade_amount_percent': 200,  # 降低单次交易比例适应深度学习策略
        'check_interval': 60,  # 调整为15分钟策略的检查间隔(秒) 
    },
    'ZECUSDT': {
        'leverage':10,
        'min_notional': 20,
        'trade_amount_percent': 200,  # 降低单次交易比例适应深度学习策略
        'check_interval': 60,  # 调整为15分钟策略的检查间隔(秒) 
    }
}
DEFAULT_LEVERAGE = 5  # 默认杠杆倍数
MARGIN_TYPE = 'CROSSED'  # 保证金模式：CROSSED(全仓) ISOLATED(逐仓)
POSITION_SIDE = 'BOTH'  # 持仓模式：BOTH(单向持仓) LONG/SHORT(双向持仓)

# 订单设置
DEFAULT_ORDER_TYPE = 'market'  # 默认订单类型：market(市价) 或 limit(限价)
TIME_IN_FORCE = 'GTC'  # 订单有效期: GTC(永久有效) GTX(立即成交或取消) IOC(立即成交剩余取消)

# 挂单策略设置
LIMIT_ORDER_DISTANCE = 0.0002  # 挂单距离当前价格的比例（0.02%）
MAX_ORDER_WAITING_TIME = 30    # 挂单最大等待时间（秒）
PRICE_CHECK_INTERVAL = 5       # 价格检查间隔（秒）

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

# DeepSeek API设置
DEEPSEEK_API_CONNECT_TIMEOUT = 10  # DeepSeek API连接超时(秒)
DEEPSEEK_API_READ_TIMEOUT = 30     # DeepSeek API读取超时(秒)
DEEPSEEK_API_RETRY_COUNT = 3       # DeepSeek API重试次数
DEEPSEEK_API_RETRY_BACKOFF = 1     # DeepSeek API重试间隔倍数
DEEPSEEK_API_MIN_INTERVAL = 10     # DeepSeek API最小调用间隔(秒)

# AI策略设置
AI_KLINES_LIMIT = 1000  # AI分析所需的K线数量
MIN_KLINES_FOR_AI = 500  # AI分析所需的最小K线数量
AI_TRADE_AMOUNT_PERCENT = 100  # AI交易金额占可用余额的百分比
AI_TRAIN_INTERVAL = 60  # AI模型训练间隔（分钟）
AI_MIN_TRADE_INTERVAL = 5  # AI最小交易间隔（分钟）

# 策略选择配置
# 可选策略: 'deepseek', 'simple_adx_di', 'kama_roc_adx', 'advanced_short_term', 'candlestick_pattern'
STRATEGY_TYPE = 'kama_roc_adx'

# K线形态策略配置
PATTERN_STRATEGY_CONFIG = {
    'kline_interval': '15m',           # K线周期
    'lookback_period': 100,            # 回溯周期
    'check_interval': 300,             # 检查间隔(秒)
    'min_probability': 0.50,           # 最小交易概率阈值
    'min_confidence': 0.60,            # 最小置信度
    'enable_auto_trade': True,         # 是否自动交易
    'weights': {                       # 各因素权重
        'pattern': 0.40,               # 形态权重
        'trend': 0.30,                 # 趋势权重
        'momentum': 0.20,              # 动量权重
        'volume': 0.10                 # 成交量权重
    },
    'pattern_weights': {               # 各形态的权重
        'hammer': 0.8,
        'inverted_hammer': 0.75,
        'hanging_man': 0.8,
        'shooting_star': 0.75,
        'engulfing': 0.9,
        'doji': 0.5,
        'three_white_soldiers': 0.85,
        'three_black_crows': 0.85,
    }
}


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
