"""
日志工具模块
"""
import os
import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 如果name中包含USDT，则移除
        if record.name and 'USDT' in record.name:
            record.name = record.name.replace('USDT', '')
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
        formatter = CustomFormatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        log_filename = f'{datetime.now().strftime("%Y-%m-%d")}.log'
        file_handler = logging.FileHandler(
            os.path.join('logs', log_filename),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
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
