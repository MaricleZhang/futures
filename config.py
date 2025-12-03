"""
交易配置文件
"""

# 代理设置
USE_PROXY = False  # 是否使用代理
PROXY_URL = 'http://127.0.0.1:7890'  # 代理地址
PROXY_TIMEOUT = 10  # 代理超时时间（秒）
PROXY_MAX_RETRIES = 3  # 代理连接最大重试次数
PROXY_TEST_TIMEOUT = 10  # 代理测试超时时间(秒)
PROXY_RETRY_DELAY = 5  # 重试延迟时间(秒)

# 可选策略: 'deepseek', 'deepseek_multi_timeframe', 'simple_adx_di', 'qwen', 'kimi'
STRATEGY_TYPE = 'deepseek_multi_timeframe'

# 交易设置
SYMBOLS = ['ENAUSDC']  # LSTM策略专用于ZECUSDT
SYMBOL_CONFIGS = {
    'ENAUSDC': {
        'leverage': 5,
        'min_notional': 20,
        'trade_amount_percent': 95,
        'check_interval': 300,  # 15分钟策略的检查间隔(秒)
    },
    'IRYSUSDT': {
        'leverage':10,
        'min_notional': 20,
        'trade_amount_percent': 200,
        'check_interval': 300,  # 5分钟策略的检查间隔(秒) 
    },
    'ENAUSDC': {
        'leverage':10,
        'min_notional': 20,
        'trade_amount_percent': 200,
        'check_interval': 300,  # 5分钟策略的检查间隔(秒) 
    },
    'BEATUSDT': {
        'leverage':10,
        'min_notional': 20,
        'trade_amount_percent': 200,
        'check_interval': 300,  # 5分钟策略的检查间隔(秒) 
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
DEFAULT_STOP_LOSS_PERCENT = 3  # 默认止损百分比（交易所兜底止损，策略层面可用更小的动态止损）
DEFAULT_TAKE_PROFIT_PERCENT = 0  # 默认止盈百分比（设为0禁用交易所止盈，策略层面控制止盈更灵活）

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

# Kimi API设置
KIMI_API_CONNECT_TIMEOUT = 10  # Kimi API连接超时(秒)
KIMI_API_READ_TIMEOUT = 30     # Kimi API读取超时(秒)
KIMI_API_RETRY_COUNT = 3       # Kimi API重试次数
KIMI_API_RETRY_BACKOFF = 1     # Kimi API重试间隔倍数
KIMI_API_MIN_INTERVAL = 10     # Kimi API最小调用间隔(秒)

# 回测设置
BACKTEST_CONFIG = {
    'initial_capital': 1000,        # 初始资金(USDT)
    'default_leverage': 10,         # 默认杠杆倍数
    'fee_rate': 0.0004,            # 手续费率 (0.04% Binance futures taker fee)
    'slippage_rate': 0.0001,       # 滑点率 (0.01%)
    'data_cache_dir': 'data/backtest',  # 数据缓存目录
    'results_dir': 'results/backtest'   # 结果输出目录
}

# 市场扫描配置
MARKET_SCAN_CONFIG = {
    'scan_interval_hours': 6,              # 扫描间隔（小时）
    'strategy_type': 'simple_adx_di',               # 使用的策略类型（默认使用STRATEGY_TYPE）
    'min_volume_24h': 1000000,             # 最小24小时交易量（USDT）过滤
    'max_symbols_to_scan': 100,            # 最大扫描数量限制（0表示不限制）
    'top_n_results': 5,                    # 输出前N个结果
    'results_dir': 'data/market_scan_results',  # 结果保存目录
    'enable_notifications': False,          # 是否启用通知（未来扩展）
}

