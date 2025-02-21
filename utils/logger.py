"""
日志工具模块
"""
import os
import logging
from datetime import datetime
from colorama import Fore, Back, Style, init

# 初始化colorama
init(autoreset=True)

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 如果name中包含USDT，则移除
        if record.name and 'USDT' in record.name:
            record.name = record.name.replace('USDT', '')
        return super().format(record)

class ColoredFormatter(logging.Formatter):
    """为不同级别的日志添加颜色"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    NUMBER_COLOR = Fore.CYAN  # 数字使用青色
    
    def colorize_numbers(self, text):
        """为文本中的数字添加颜色"""
        import re
        
        # 匹配不同格式的数字
        patterns = [
            r'(\d+\.\d+)',  # 匹配小数
            r'(?<!\.)\b(\d+)\b(?!\.)',  # 匹配整数
            r'(\d+\.\d+%)',  # 匹配百分比
            r'(0\.\d+)',  # 匹配小于1的小数
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, f'{self.NUMBER_COLOR}\\1{Style.RESET_ALL}', text)
        
        return text

    def format(self, record):
        # 如果name中包含USDT，则移除
        if record.name and 'USDT' in record.name:
            record.name = record.name.replace('USDT', '')
            
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            # 为消息中的数字添加颜色
            record.msg = self.colorize_numbers(str(record.msg))
            
        return super().format(record)

class Logger:
    _instance = None
    _logger = None
    
    @classmethod
    def get_logger(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._logger
        
    def __init__(self):
        if Logger._logger is not None:
            raise Exception("Logger类是单例的!")
            
        self.setup_logger()
        
    def setup_logger(self):
        """设置日志记录器"""
        # 创建日志目录
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # 设置日志格式
        file_formatter = CustomFormatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        log_filename = f'{datetime.now().strftime("%Y-%m-%d")}.log'
        file_handler = logging.FileHandler(
            os.path.join('logs', log_filename),
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # 设置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        root_logger.handlers = []
        
        # 添加新的处理器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 输出启动信息
        logger = logging.getLogger()
        logger.info('='*50)
        logger.info(f'日志系统初始化完成 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info('='*50)
        
        Logger._logger = logger
