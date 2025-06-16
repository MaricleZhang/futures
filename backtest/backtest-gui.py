"""
回测系统图形界面
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.logger import Logger
from backtest import Backtester

# 导入所有策略
from strategies.strategy_module import DirectionalIndexStrategy15m
from strategies.multi_timeframe_di_adx_strategy import MultiTimeframeDIADXStrategy

# 策略字典
STRATEGIES = {
    'DI变化率策略': DirectionalIndexStrategy15m,
    '多周期DIADX策略': MultiTimeframeDIADXStrategy,
}

# 时间周期选项
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']

class RedirectText:
    """用于重定向日志输出到GUI的文本框"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_interval = 100  # 更新频率(毫秒)
        
        # 设置文本框样式
        self.text_widget.tag_configure("INFO", foreground="green")
        self.text_widget.tag_configure("WARNING", foreground="orange")
        self.text_widget.tag_configure("ERROR", foreground="red")
        self.text_widget.tag_configure("DEBUG", foreground="blue")
        
        self.update_widget()
    
    def write(self, string):
        """写入队列"""
        self.queue.put(string)
    
    def flush(self):
        """刷新"""
        pass
    
    def update_widget(self):
        """更新文本框"""
        while not self.queue.empty():
            msg = self.queue.get()
            if msg:
                # 获取消息类型
                tag = "INFO"  # 默认为INFO
                if "WARNING" in msg:
                    tag = "WARNING"
                elif "ERROR" in msg:
                    tag = "ERROR"
                elif "DEBUG" in msg:
                    tag = "DEBUG"
                
                # 插入文本并应用标签
                self.text_widget.insert(tk.END, msg, tag)
                self.text_widget.see(tk.END)  # 自动滚动到底部
        
        # 安排下一次更新
        self.text_widget.after(self.update_interval, self.update_widget)

class BacktestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("加密货币交易策略回测系统")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 创建日志记录器
        self.logger = Logger.get_logger()
        
        # 当前回测结果
        self.current_results = None
        
        # 创建界面
        self.create_widgets()
        
        # 设置日志重定向
        self.redirect_logging()
    
    def create_widgets(self):
        """创建GUI界面控件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧设置面板
        left_frame = ttk.LabelFrame(main_frame, text="回测设置", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # 右侧内容面板
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== 左侧设置面板内容 =====
        # 交易对输入
        ttk.Label(left_frame, text="交易对:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(left_frame, textvariable=self.symbol_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 策略选择
        ttk.Label(left_frame, text="策略:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(left_frame, textvariable=self.strategy_var, width=20)
        strategy_combo['values'] = list(STRATEGIES.keys())
        strategy_combo.current(0)
        strategy_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 时间周期
        ttk.Label(left_frame, text="K线周期:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.timeframe_var = tk.StringVar()
        timeframe_combo = ttk.Combobox(left_frame, textvariable=self.timeframe_var, width=20)
        timeframe_combo['values'] = TIMEFRAMES
        timeframe_combo.current(2)  # 默认选择5m
        timeframe_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 开始日期
        ttk.Label(left_frame, text="开始日期:").grid(row=3, column=0, sticky=tk.W, pady=5)
        default_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.start_var = tk.StringVar(value=default_start)
        ttk.Entry(left_frame, textvariable=self.start_var, width=20).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 结束日期
        ttk.Label(left_frame, text="结束日期:").grid(row=4, column=0, sticky=tk.W, pady=5)
        default_end = datetime.now().strftime('%Y-%m-%d')
        self.end_var = tk.StringVar(value=default_end)
        ttk.Entry(left_frame, textvariable=self.end_var, width=20).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # 初始资金
        ttk.Label(left_frame, text="初始资金:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.balance_var = tk.StringVar(value="10000")
        ttk.Entry(left_frame, textvariable=self.balance_var, width=20).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # 杠杆倍数
        ttk.Label(left_frame, text="杠杆倍数:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.leverage_var = tk.StringVar(value="5")
        ttk.Entry(left_frame, textvariable=self.leverage_var, width=20).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # 强制重新下载数据
        self.reload_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="强制重新下载数据", variable=self.reload_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 回测按钮
        ttk.Button(left_frame, text="开始回测", command=self.start_backtest).grid(row=8, column=0, columnspan=2, pady=10)
        
        # 结果保存按钮
        ttk.Button(left_frame, text="保存回测结果", command=self.save_results).grid(row=9, column=0, columnspan=2, pady=5)
        
        # 帮助按钮
        ttk.Button(left_frame, text="帮助", command=self.show_help).grid(row=10, column=0, columnspan=2, pady=5)
        
        # ===== 右侧内容面板内容 =====
        # 创建选项卡控件
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 日志选项卡
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="日志")
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)