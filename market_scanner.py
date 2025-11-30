"""
市场扫描器 - 扫描所有USDT永续合约并分析交易机会

功能：
1. 获取所有USDT永续合约
2. 使用指定策略分析每个合约的交易信号
3. 按信号确定性（置信度）排序
4. 输出做多和做空各前N个交易对
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from trader import Trader
import config

# 导入策略
from strategies.qwen_trading_strategy import QwenTradingStrategy
from strategies.deepseek_trading_strategy import DeepSeekTradingStrategy
from strategies.kimi_trading_strategy import KimiTradingStrategy
from strategies.simple_adx_di_15m_strategy import SimpleADXDIStrategy15m
from strategies.xgboost_price_strategy import XGBoostPriceStrategy

# 策略映射
STRATEGY_MAP = {
    'qwen': QwenTradingStrategy,
    'deepseek': DeepSeekTradingStrategy,
    'kimi': KimiTradingStrategy,
    'simple_adx_di': SimpleADXDIStrategy15m,
    'xgboost': XGBoostPriceStrategy,
}


class MarketScanner:
    """市场扫描器"""
    
    def __init__(self, strategy_type: str = None, top_n: int = 5):
        """
        初始化市场扫描器
        
        Args:
            strategy_type: 策略类型，默认使用config中的配置
            top_n: 返回前N个结果
        """
        self.logger = logging.getLogger('MarketScanner')
        self.logger.setLevel(logging.INFO)
        
        # 获取配置
        self.scan_config = getattr(config, 'MARKET_SCAN_CONFIG', {})
        self.strategy_type = strategy_type or self.scan_config.get('strategy_type', config.STRATEGY_TYPE)
        self.top_n = top_n or self.scan_config.get('top_n_results', 5)
        self.min_volume_24h = self.scan_config.get('min_volume_24h', 1000000)
        self.max_symbols = self.scan_config.get('max_symbols_to_scan', 100)
        self.results_dir = self.scan_config.get('results_dir', 'data/market_scan_results')
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 创建临时trader用于获取市场信息
        self.temp_trader = Trader()
        
        self.logger.info(f"Market Scanner initialized - Strategy: {self.strategy_type}, Top N: {self.top_n}")
    
    def get_all_usdt_perpetual_symbols(self) -> List[str]:
        """
        获取所有USDT永续合约
        
        Returns:
            符合条件的交易对列表
        """
        try:
            self.logger.info("Fetching all USDT perpetual contracts...")
            
            # 获取所有市场
            markets = self.temp_trader.exchange.fetch_markets()
            
            # 筛选USDT永续合约
            usdt_perpetuals = []
            for market in markets:
                # 检查是否为USDT永续合约
                if (market.get('quote') == 'USDT' and 
                    market.get('type') == 'swap' and 
                    market.get('linear') == True and
                    market.get('active') == True):
                    
                    symbol = market['symbol']
                    usdt_perpetuals.append(symbol)
            
            self.logger.info(f"Found {len(usdt_perpetuals)} USDT perpetual contracts")
            
            # 如果有数量限制，只取前N个
            if self.max_symbols and len(usdt_perpetuals) > self.max_symbols:
                usdt_perpetuals = usdt_perpetuals[:self.max_symbols]
                self.logger.info(f"Limited to {self.max_symbols} symbols for scanning")
            
            return usdt_perpetuals
            
        except Exception as e:
            self.logger.error(f"Error fetching markets: {str(e)}")
            return []
    
    def analyze_symbol(self, symbol: str, strategy_type: str) -> Optional[Dict]:
        """
        分析单个交易对
        
        Args:
            symbol: 交易对符号
            strategy_type: 策略类型
            
        Returns:
            分析结果字典，包含signal, confidence等信息
        """
        try:
            # 创建临时trader实例
            trader = Trader(symbol=symbol)
            
            # 获取策略类
            strategy_class = STRATEGY_MAP.get(strategy_type)
            if not strategy_class:
                self.logger.error(f"Unknown strategy type: {strategy_type}")
                return None
            
            # 创建策略实例
            strategy = strategy_class(trader=trader)
            
            # 生成信号
            signal = strategy.generate_signal()
            
            # 获取置信度（如果策略支持）
            confidence = 0.0
            prediction_info = {}
            
            # 对于AI策略，尝试获取详细预测信息
            if hasattr(strategy, 'last_ai_prediction') and strategy.last_ai_prediction:
                confidence = strategy.last_ai_prediction.get('confidence', 0.0)
                prediction_info = {
                    'prediction': strategy.last_ai_prediction.get('prediction', ''),
                    'reasoning': strategy.last_ai_prediction.get('reasoning', ''),
                    'support': strategy.last_ai_prediction.get('support'),
                    'resistance': strategy.last_ai_prediction.get('resistance'),
                }
            else:
                # 对于技术指标策略，使用固定置信度
                if signal != 0:
                    confidence = 0.70  # 默认技术指标的置信度
            
            # 获取当前价格
            current_price = trader.get_market_price()
            
            result = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'strategy': strategy_type,
                'timestamp': datetime.now().isoformat(),
                **prediction_info
            }
            
            # 清理资源
            if hasattr(strategy, 'cleanup'):
                strategy.cleanup()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def scan_market(self) -> Dict[str, List[Dict]]:
        """
        扫描市场，分析所有合约
        
        Returns:
            包含long和short机会的字典
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting market scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Strategy: {self.strategy_type}")
        self.logger.info("=" * 80)
        
        # 获取所有合约
        symbols = self.get_all_usdt_perpetual_symbols()
        
        if not symbols:
            self.logger.error("No symbols to scan")
            return {'long': [], 'short': []}
        
        # 分析每个合约
        long_opportunities = []
        short_opportunities = []
        
        total = len(symbols)
        for idx, symbol in enumerate(symbols, 1):
            self.logger.info(f"[{idx}/{total}] Analyzing {symbol}...")
            
            try:
                result = self.analyze_symbol(symbol, self.strategy_type)
                
                if result:
                    signal = result['signal']
                    confidence = result['confidence']
                    
                    # 分类做多和做空机会
                    if signal == 1 and confidence > 0:
                        long_opportunities.append(result)
                        self.logger.info(f"  ✓ LONG signal - Confidence: {confidence:.2%}")
                    elif signal == -1 and confidence > 0:
                        short_opportunities.append(result)
                        self.logger.info(f"  ✓ SHORT signal - Confidence: {confidence:.2%}")
                    else:
                        self.logger.info(f"  - No clear signal (signal={signal}, conf={confidence:.2%})")
                
            except Exception as e:
                self.logger.error(f"  ✗ Failed: {str(e)}")
                continue
        
        self.logger.info("=" * 80)
        self.logger.info(f"Scan complete - Long: {len(long_opportunities)}, Short: {len(short_opportunities)}")
        self.logger.info("=" * 80)
        
        return {
            'long': long_opportunities,
            'short': short_opportunities,
            'scan_time': datetime.now().isoformat(),
            'strategy': self.strategy_type,
            'total_scanned': total
        }
    
    def rank_opportunities(self, opportunities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        对交易机会进行排序
        
        Args:
            opportunities: 扫描结果
            
        Returns:
            排序后的结果
        """
        # 按置信度降序排序
        long_sorted = sorted(
            opportunities['long'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:self.top_n]
        
        short_sorted = sorted(
            opportunities['short'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:self.top_n]
        
        return {
            'long': long_sorted,
            'short': short_sorted,
            'scan_time': opportunities.get('scan_time'),
            'strategy': opportunities.get('strategy'),
            'total_scanned': opportunities.get('total_scanned')
        }
    
    def save_results(self, results: Dict) -> str:
        """
        保存结果到文件
        
        Args:
            results: 扫描结果
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scan_result_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return ""
    
    def print_results(self, results: Dict):
        """
        格式化输出结果到控制台
        
        Args:
            results: 扫描结果
        """
        print("\n")
        print("=" * 100)
        print(f"Market Scan Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategy: {results.get('strategy', 'Unknown')}")
        print(f"Total Scanned: {results.get('total_scanned', 0)} symbols")
        print("=" * 100)
        
        # 做多机会
        print("\n📈 TOP LONG OPPORTUNITIES (做多机会)")
        print("-" * 100)
        
        if results['long']:
            # 表头
            print(f"{'Rank':<6}{'Symbol':<15}{'Price':<15}{'Confidence':<12}{'Prediction':<20}{'Reasoning':<30}")
            print("-" * 100)
            
            for idx, opp in enumerate(results['long'], 1):
                symbol = opp['symbol']
                price = f"{opp['current_price']:.6f}"
                confidence = f"{opp['confidence']:.1%}"
                prediction = opp.get('prediction', 'N/A')[:18]
                reasoning = opp.get('reasoning', 'Technical analysis')[:28]
                
                print(f"{idx:<6}{symbol:<15}{price:<15}{confidence:<12}{prediction:<20}{reasoning:<30}")
        else:
            print("No long opportunities found.")
        
        # 做空机会
        print("\n📉 TOP SHORT OPPORTUNITIES (做空机会)")
        print("-" * 100)
        
        if results['short']:
            # 表头
            print(f"{'Rank':<6}{'Symbol':<15}{'Price':<15}{'Confidence':<12}{'Prediction':<20}{'Reasoning':<30}")
            print("-" * 100)
            
            for idx, opp in enumerate(results['short'], 1):
                symbol = opp['symbol']
                price = f"{opp['current_price']:.6f}"
                confidence = f"{opp['confidence']:.1%}"
                prediction = opp.get('prediction', 'N/A')[:18]
                reasoning = opp.get('reasoning', 'Technical analysis')[:28]
                
                print(f"{idx:<6}{symbol:<15}{price:<15}{confidence:<12}{prediction:<20}{reasoning:<30}")
        else:
            print("No short opportunities found.")
        
        print("\n" + "=" * 100 + "\n")
    
    def run(self) -> Dict:
        """
        执行完整的扫描流程
        
        Returns:
            扫描结果
        """
        # 扫描市场
        opportunities = self.scan_market()
        
        # 排序
        ranked = self.rank_opportunities(opportunities)
        
        # 输出结果
        self.print_results(ranked)
        
        # 保存结果
        self.save_results(ranked)
        
        return ranked


if __name__ == '__main__':
    # 测试代码
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 从命令行参数获取策略类型
    strategy = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 创建扫描器并运行
    scanner = MarketScanner(strategy_type=strategy)
    results = scanner.run()
