"""
市场扫描定时调度脚本

功能：
- 定时执行市场扫描（默认每6小时）
- 支持立即执行一次扫描
- 支持自定义扫描间隔和策略
- 优雅退出处理
"""

import argparse
import logging
import signal
import sys
import time
import schedule
from datetime import datetime
from market_scanner import MarketScanner


class ScheduledScanner:
    """定时扫描调度器"""
    
    def __init__(self, interval_hours: int = 6, strategy_type: str = None, top_n: int = 5):
        """
        初始化调度器
        
        Args:
            interval_hours: 扫描间隔（小时）
            strategy_type: 策略类型
            top_n: 返回前N个结果
        """
        self.interval_hours = interval_hours
        self.strategy_type = strategy_type
        self.top_n = top_n
        self.running = True
        
        # 配置日志
        self.logger = logging.getLogger('ScheduledScanner')
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Scheduled Scanner initialized - Interval: {interval_hours}h, Strategy: {strategy_type}")
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        self.logger.info("\nReceived shutdown signal, stopping scanner...")
        self.running = False
    
    def run_scan(self):
        """执行一次扫描"""
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Scheduled scan started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"{'='*80}\n")
            
            # 创建扫描器并执行
            scanner = MarketScanner(strategy_type=self.strategy_type, top_n=self.top_n)
            results = scanner.run()
            
            # 输出下次扫描时间
            next_run = datetime.now().replace(microsecond=0)
            next_run = next_run.replace(hour=(next_run.hour + self.interval_hours) % 24)
            self.logger.info(f"\nNext scan scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error during scan: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def start_scheduler(self):
        """启动定时调度"""
        self.logger.info(f"Starting scheduler with {self.interval_hours} hour interval...")
        
        # 立即执行一次
        self.run_scan()
        
        # 设置定时任务
        schedule.every(self.interval_hours).hours.do(self.run_scan)
        
        # 主循环
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                self.logger.error(f"Scheduler error: {str(e)}")
                time.sleep(60)
        
        self.logger.info("Scheduler stopped gracefully")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Market Scanner - 定时扫描所有USDT永续合约')
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='立即执行一次扫描后退出'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=6,
        help='扫描间隔（小时），默认6小时'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        choices=['qwen', 'deepseek', 'kimi', 'simple_adx_di', 'xgboost'],
        help='使用的策略类型，默认使用config.py中的配置'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='显示前N个结果，默认5'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    args = parser.parse_args()
    
    # 配置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('main')
    
    # 打印配置信息
    logger.info("="*80)
    logger.info("Market Scanner Starting...")
    logger.info("="*80)
    logger.info(f"Mode: {'Once' if args.once else f'Scheduled ({args.interval}h)'}")
    logger.info(f"Strategy: {args.strategy or 'config default'}")
    logger.info(f"Top N: {args.top_n}")
    logger.info("="*80)
    
    try:
        if args.once:
            # 单次扫描模式
            scanner = MarketScanner(strategy_type=args.strategy, top_n=args.top_n)
            results = scanner.run()
            logger.info("Scan completed successfully!")
        else:
            # 定时扫描模式
            scheduler = ScheduledScanner(
                interval_hours=args.interval,
                strategy_type=args.strategy,
                top_n=args.top_n
            )
            scheduler.start_scheduler()
    
    except KeyboardInterrupt:
        logger.info("\nScan interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
