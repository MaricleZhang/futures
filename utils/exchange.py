"""
交易所工具模块
"""
import ccxt
import config
import logging
import time
import requests
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

# 配置日志记录器
logger = logging.getLogger(__name__)

def test_proxy(proxy_url, timeout=10):
    """
    测试代理连接是否可用
    """
    try:
        response = requests.get('https://api1.binance.com/api/v3/time', 
                              proxies={'http': proxy_url, 'https': proxy_url},
                              timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"代理测试失败: {str(e)}")
        return False

_EXCHANGE_API_KEYS = {
    'binance': ('BINANCE_API_KEY', 'BINANCE_SECRET_KEY'),
    'gate': ('GATE_API_KEY', 'GATE_SECRET_KEY'),
}


def _build_exchange(exchange_name, api_key, api_secret):
    """根据 exchange_name 创建对应的 ccxt 实例并设置期货/永续默认值"""
    name = exchange_name.lower()
    if name == 'binance':
        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
            },
        }
        return ccxt.binance(params), params
    if name == 'gate':
        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'defaultSettle': 'usdt',
                'adjustForTimeDifference': True,
            },
        }
        return ccxt.gate(params), params
    raise ValueError(f"不支持的交易所: {exchange_name}")


def init_exchange(exchange_name='binance', max_retries=3, retry_delay=5):
    """
    初始化交易所接口（支持 binance / gate）
    - exchange_name: 'binance' 或 'gate'
    - 支持重试、代理、时间同步、testnet 沙盒模式
    """
    # 加载环境变量
    load_dotenv()
    exchange_name = (exchange_name or 'binance').lower()

    # 获取API密钥
    env_keys = _EXCHANGE_API_KEYS.get(exchange_name)
    if not env_keys:
        raise ValueError(f"不支持的交易所: {exchange_name}")
    api_key_env, api_secret_env = env_keys
    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)

    if not api_key or not api_secret:
        raise ValueError(f"未找到 {exchange_name} API密钥（{api_key_env} / {api_secret_env}），请检查 .env 文件")

    exchange, exchange_params = _build_exchange(exchange_name, api_key, api_secret)

    # 代理设置
    if config.USE_PROXY:
        retry_count = 0
        while retry_count < config.PROXY_MAX_RETRIES:
            if test_proxy(config.PROXY_URL, config.PROXY_TEST_TIMEOUT):
                logger.info(f"[{exchange_name}] 代理连接测试成功")
                exchange.proxies = {
                    'http': config.PROXY_URL,
                    'https': config.PROXY_URL,
                }
                exchange.timeout = config.PROXY_TIMEOUT * 1000
                break
            retry_count += 1
            if retry_count < config.PROXY_MAX_RETRIES:
                logger.warning(f"[{exchange_name}] 代理连接失败，{config.PROXY_RETRY_DELAY}秒后重试 ({retry_count}/{config.PROXY_MAX_RETRIES})")
                time.sleep(config.PROXY_RETRY_DELAY)
            else:
                logger.error(f"[{exchange_name}] 代理连接失败，将不使用代理继续运行")

    # testnet 模式
    if getattr(config, 'USE_TESTNET', False):
        try:
            exchange.set_sandbox_mode(True)
            logger.info(f"[{exchange_name}] 已启用 TESTNET 沙盒模式")
        except Exception as e:
            logger.warning(f"[{exchange_name}] 启用 testnet 失败: {e}")

    for attempt in range(max_retries):
        try:
            # 测试API连接
            server_time = exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            logger.info(f"[{exchange_name}] 服务器时间差: {time_diff} ms")

            if abs(time_diff) > 1000:
                logger.warning(f"[{exchange_name}] 检测到较大的时间差异: {time_diff}ms，正在调整...")
                exchange.options['timeDifference'] = time_diff

            # 测试API权限
            try:
                exchange.fetch_balance()
                logger.info(f"[{exchange_name}] API权限验证成功")
            except Exception as e:
                raise ValueError(f"[{exchange_name}] API权限验证失败: {str(e)}")

            return exchange

        except ccxt.NetworkError as e:
            if attempt < max_retries - 1:
                logger.warning(f"[{exchange_name}] 连接失败，{retry_delay}秒后重试 ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"初始化交易所 {exchange_name} 失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"初始化交易所 {exchange_name} 失败: {str(e)}")

def check_exchange_status(exchange):
    """
    检查交易所连接状态
    """
    try:
        # 检查系统状态
        try:
            status = exchange.fetchStatus()
            if status:
                logger.info(f"系统状态: {status}")
        except:
            # 如果获取状态失败，尝试直接检查API可用性
            exchange.fetch_time()
            logger.info("系统状态检查：API可用")
        
        # 获取交易所服务器时间
        server_time = exchange.fetch_time()
        local_time = int(time.time() * 1000)
        time_diff = server_time - local_time
        logger.info(f"服务器时间差: {time_diff} ms")
        
        # 获取期货账户余额
        balance = exchange.fetch_balance({'type': 'future'})
        logger.info("期货账户余额获取成功")
        
        # 获取当前可用的交易对
        markets = exchange.load_markets()
        logger.info(f"成功加载 {len(markets)} 个交易对")
        
        # 检查目标交易对是否可用
        if config.SYMBOL not in markets:
            raise ValueError(f"交易对 {config.SYMBOL} 不可用")
        logger.info(f"交易对 {config.SYMBOL} 可用")
        
        return True
    except Exception as e:
        logger.error(f"交易所状态检查失败: {str(e)}")
        return False
