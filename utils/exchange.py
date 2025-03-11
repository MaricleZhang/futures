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
        response = requests.get('https://api.binance.com/api/v3/time', 
                              proxies={'http': proxy_url, 'https': proxy_url},
                              timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"代理测试失败: {str(e)}")
        return False

def init_exchange(max_retries=3, retry_delay=5):
    """
    初始化交易所接口
    支持重试机制
    """
    # 加载环境变量
    load_dotenv()
    
    # 获取API密钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not api_secret:
        raise ValueError("未找到API密钥，请检查.env文件")
    
    # 初始化交易所设置
    exchange_params = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True,
            'recvWindow': 60000  # 增加到60秒，给予更大的时间窗口
        }
    }
    
    # 如果启用代理，添加代理设置
    if config.USE_PROXY:
        # 测试代理连接
        retry_count = 0
        while retry_count < config.PROXY_MAX_RETRIES:
            if test_proxy(config.PROXY_URL, config.PROXY_TEST_TIMEOUT):
                logger.info("代理连接测试成功")
                exchange_params['proxies'] = {
                    'http': config.PROXY_URL,
                    'https': config.PROXY_URL
                }
                exchange_params['timeout'] = config.PROXY_TIMEOUT * 1000  # 转换为毫秒
                break
            else:
                retry_count += 1
                if retry_count < config.PROXY_MAX_RETRIES:
                    logger.warning(f"代理连接失败，{config.PROXY_RETRY_DELAY}秒后重试 ({retry_count}/{config.PROXY_MAX_RETRIES})")
                    time.sleep(config.PROXY_RETRY_DELAY)
                else:
                    logger.error("代理连接失败，将不使用代理继续运行")
    
    # 创建交易所实例
    exchange = ccxt.binance(exchange_params)
    
    for attempt in range(max_retries):
        try:
            # 测试API连接
            server_time = exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            logger.info(f"服务器时间差: {time_diff} ms")
            
            # 如果时间差异过大，进行本地时间调整
            if abs(time_diff) > 1000:  # 如果时间差超过1秒
                logger.warning(f"检测到较大的时间差异: {time_diff}ms，正在调整...")
                # 将交易所时间差异保存到exchange对象中，用于后续请求
                exchange.options['timeDifference'] = time_diff
                logger.info(f"已设置时间差异补偿: {time_diff}ms")
            
            # 设置市场类型
            exchange.options['defaultType'] = 'future'
            
            # 测试API权限
            try:
                exchange.fetch_balance()
                logger.info("API权限验证成功")
            except Exception as e:
                raise ValueError(f"API权限验证失败: {str(e)}")
                
            return exchange
            
        except ccxt.NetworkError as e:
            if attempt < max_retries - 1:
                logger.warning(f"连接失败，{retry_delay}秒后重试 ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"初始化交易所失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"初始化交易所失败: {str(e)}")

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
