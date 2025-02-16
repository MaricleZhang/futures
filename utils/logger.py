"""
日志工具模块
"""
import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

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
            
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 获取当前日期作为文件名
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f'{current_date}.log')
        
        # 创建logger
        logger = logging.getLogger('futures_trading')
        logger.setLevel(logging.INFO)
        
        # 创建TimedRotatingFileHandler
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',  # 每天午夜切换文件
            interval=1,       # 间隔为1天
            backupCount=30,   # 保留30天的日志
            encoding='utf-8'
        )
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # 输出分隔线
        logger.info("="*50)
        logger.info(f"日志系统初始化完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        
        Logger._logger = logger
