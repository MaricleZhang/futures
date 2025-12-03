"""
DeepSeek Multi-Timeframe AI Trading Strategy
Analyzes both 15m and 1h timeframes to generate comprehensive trading signals

File: strategies/deepseek_multi_timeframe_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from strategies.base_strategy import BaseStrategy
import config


class DeepSeekMultiTimeframeStrategy(BaseStrategy):
    """
    DeepSeek AI-powered multi-timeframe trading strategy
    
    Strategy Logic:
    1. Analyzes both 15m and 1h timeframes separately
    2. Generates individual signals and confidence for each timeframe
    3. Combines signals using weighted logic (1h: 60%, 15m: 40%)
    4. Outputs probabilities for long, short, and hold actions
    
    Trading Signals:
    - Individual timeframe signals: 1 (long), -1 (short), 0 (hold)
    - Combined signal: weighted combination of both timeframes
    - Probabilities: percentage distribution across long/short/hold
    """
    
    def __init__(self, trader, deepseek_api_key=None):
        """Initialize the DeepSeek multi-timeframe trading strategy
        
        Args:
            trader: Trader instance
            deepseek_api_key: DeepSeek API key (optional, can be set from environment)
        """
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # API Configuration
        self.deepseek_api_key = deepseek_api_key or self._get_api_key()
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        
        # Configure HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=getattr(config, 'DEEPSEEK_API_RETRY_COUNT', 3),
            backoff_factor=getattr(config, 'DEEPSEEK_API_RETRY_BACKOFF', 1),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Timeout configuration
        connect_timeout = getattr(config, 'DEEPSEEK_API_CONNECT_TIMEOUT', 10)
        read_timeout = getattr(config, 'DEEPSEEK_API_READ_TIMEOUT', 30)
        self.api_timeout = (connect_timeout, read_timeout)
        
        # Multi-timeframe configuration
        self.timeframes = {
            '15m': {
                'interval': '15m',
                'weight': 0.4,  # 40% weight
                'lookback': 100,
                'description': '15分钟'
            },
            '1h': {
                'interval': '1h',
                'weight': 0.6,  # 60% weight
                'lookback': 100,
                'description': '1小时'
            }
        }
        
        # Use 15m as primary interval for monitoring
        self.kline_interval = '15m'
        self.check_interval = 300  # 5 minutes
        self.lookback_period = 100
        self.training_lookback = 100
        
        # Technical Indicator Parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
        self.volume_ma_period = 20
        
        # Trading Parameters
        self.min_confidence = 0.65
        
        # Rate Limiting
        self.is_backtest = getattr(trader, 'is_backtest', False)
        if self.is_backtest:
            self.min_time_between_api_calls = 0
        else:
            self.min_time_between_api_calls = getattr(config, 'DEEPSEEK_API_MIN_INTERVAL', 10)
        
        self.last_api_call_time = {}  # Track per timeframe
        self.last_predictions = {}  # Cache predictions per timeframe
        
        self.logger.info("DeepSeek Multi-Timeframe Strategy initialized")
        self.logger.info(f"Analyzing timeframes: 15m (40%) + 1h (60%)")
        self.logger.info(f"API Key configured: {bool(self.deepseek_api_key)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
                self.logger.info("HTTP session closed")
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
    
    def _get_api_key(self):
        """Get DeepSeek API key from environment or config"""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            self.logger.warning("DeepSeek API key not found. Strategy will work in limited mode.")
        return api_key
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators
        
        Args:
            df (DataFrame): OHLCV data
            
        Returns:
            dict: Dictionary containing all calculated indicators
        """
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI
            rsi = ta.rsi(pd.Series(close), length=self.rsi_period).values
            
            # MACD
            macd_df = ta.macd(
                pd.Series(close), 
                fast=self.macd_fast,
                slow=self.macd_slow, 
                signal=self.macd_signal
            )
            macd_col = f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            signal_col = f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            hist_col = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            
            macd = macd_df[macd_col].values
            macd_signal = macd_df[signal_col].values
            macd_hist = macd_df[hist_col].values
            
            # Bollinger Bands
            bb_df = ta.bbands(
                pd.Series(close), 
                length=self.bb_period,
                std=self.bb_std
            )
            bb_lower = bb_df[f'BBL_{self.bb_period}_{float(self.bb_std)}'].values
            bb_middle = bb_df[f'BBM_{self.bb_period}_{float(self.bb_std)}'].values
            bb_upper = bb_df[f'BBU_{self.bb_period}_{float(self.bb_std)}'].values
            
            # ATR
            atr = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=self.atr_period).values
            
            # Moving Averages
            ema_fast = ta.ema(pd.Series(close), length=10).values
            ema_slow = ta.ema(pd.Series(close), length=30).values
            
            # Volume indicators
            volume_ma = ta.sma(pd.Series(volume), length=self.volume_ma_period).values
            
            # Stochastic
            stoch_df = ta.stoch(pd.Series(high), pd.Series(low), pd.Series(close), k=5, d=3, smooth_k=3)
            slowk = stoch_df['STOCHk_5_3_3'].values
            slowd = stoch_df['STOCHd_5_3_3'].values
            
            # ADX
            adx_df = ta.adx(pd.Series(high), pd.Series(low), pd.Series(close), length=14)
            adx = adx_df['ADX_14'].values
            
            # Momentum
            momentum = ta.mom(pd.Series(close), length=10).values
            
            # BB width
            bb_width = (bb_upper - bb_lower) / bb_middle * 100
            
            return {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'atr': atr,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'volume': volume,
                'volume_ma': volume_ma,
                'slowk': slowk,
                'slowd': slowd,
                'adx': adx,
                'momentum': momentum,
                'close': close
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
    
    def prepare_market_analysis(self, indicators, timeframe_desc):
        """Prepare market analysis text for DeepSeek API
        
        Args:
            indicators (dict): Technical indicators
            timeframe_desc (str): Timeframe description
            
        Returns:
            str: Formatted market analysis text
        """
        try:
            current_price = indicators['close'][-1]
            prev_price = indicators['close'][-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            rsi = indicators['rsi'][-1]
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_hist = indicators['macd_hist'][-1]
            
            bb_upper = indicators['bb_upper'][-1]
            bb_middle = indicators['bb_middle'][-1]
            bb_lower = indicators['bb_lower'][-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
            
            ema_fast = indicators['ema_fast'][-1]
            ema_slow = indicators['ema_slow'][-1]
            
            adx = indicators['adx'][-1]
            slowk = indicators['slowk'][-1]
            momentum = indicators['momentum'][-1]
            
            volume = indicators['volume'][-1]
            volume_ma = indicators['volume_ma'][-1]
            volume_ratio = (volume / volume_ma) * 100
            
            recent_prices = indicators['close'][-10:]
            trend = "上升" if recent_prices[-1] > recent_prices[0] else "下降"
            
            analysis = f"""
市场分析报告 ({self.trader.symbol} - {timeframe_desc}):

价格信息:
- 当前价格: {current_price:.6f}
- 价格变动: {price_change:+.2f}%
- 短期趋势: {trend}

技术指标:
- RSI(14): {rsi:.2f} {'超买' if rsi > 70 else '超卖' if rsi < 30 else '中性'}
- MACD: {macd:.6f} (信号线: {macd_signal:.6f}, 柱状图: {macd_hist:.6f})
- MACD趋势: {'多头' if macd > macd_signal else '空头'}

- 布林带位置: {bb_position:.1f}% {'接近上轨' if bb_position > 80 else '接近下轨' if bb_position < 20 else '中间位置'}
- 布林带范围: {bb_lower:.6f} - {bb_upper:.6f}

- EMA快线: {ema_fast:.6f}
- EMA慢线: {ema_slow:.6f}
- 均线关系: {'金叉' if ema_fast > ema_slow else '死叉'}

- ADX: {adx:.2f} {'强趋势' if adx > 25 else '弱趋势'}
- 随机指标: {slowk:.2f} {'超买' if slowk > 80 else '超卖' if slowk < 20 else '中性'}
- 动量: {momentum:.6f}

成交量分析:
- 当前成交量: {volume:.2f}
- 成交量比率: {volume_ratio:.1f}% {'放量' if volume_ratio > 120 else '缩量' if volume_ratio < 80 else '正常'}

最近价格走势:
{self._format_price_history(recent_prices)}
"""
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error preparing market analysis: {str(e)}")
            return None
    
    def _format_price_history(self, prices):
        """Format recent price history"""
        history = []
        for i, price in enumerate(prices[-5:], 1):
            history.append(f"  {i}前: {price:.6f}")
        return "\n".join(history)
    
    def query_deepseek_prediction(self, market_analysis, timeframe_desc):
        """Query DeepSeek API for price prediction
        
        Args:
            market_analysis (str): Formatted market analysis
            timeframe_desc (str): Timeframe description
            
        Returns:
            dict: Prediction results with signal and confidence
        """
        # Check rate limiting
        current_time = time.time()
        if timeframe_desc in self.last_api_call_time:
            if current_time - self.last_api_call_time[timeframe_desc] < self.min_time_between_api_calls:
                self.logger.debug(f"Rate limit: using cached prediction for {timeframe_desc}")
                return self.last_predictions.get(timeframe_desc)
        
        if not self.deepseek_api_key:
            self.logger.warning("No API key available, using fallback logic")
            return self._fallback_prediction()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""你是一位专业的加密货币交易分析师。请基于以下{timeframe_desc}周期的市场数据分析，预测未来价格趋势。

{market_analysis}

**分析原则**：
1. 考虑{timeframe_desc}周期的特点调整分析视角
2. 使用多重技术指标确认
3. 评估趋势强度和可持续性
4. 保守评估置信度

请提供:
1. 趋势预测: 上涨/下跌/震荡/震荡偏涨/震荡偏跌
2. 置信度: 0-100的数值
3. 建议操作: 买入/卖出/轻仓买入/轻仓卖出/观望
4. 关键支撑/阻力位
5. 分析推理

请用JSON格式回复:
{{
    "prediction": "上涨/下跌/震荡偏涨/震荡偏跌/震荡",
    "confidence": 75,
    "action": "买入/卖出/轻仓买入/轻仓卖出/观望",
    "support_level": 价格,
    "resistance_level": 价格,
    "reasoning": "详细说明",
    "risk_warning": "风险提示"
}}"""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"你是专业的加密货币{timeframe_desc}周期交易分析师，注重技术分析和风险控制。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            
            response = self.session.post(
                self.deepseek_api_url,
                headers=headers,
                json=payload,
                timeout=self.api_timeout
            )
            
            self.last_api_call_time[timeframe_desc] = current_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    prediction = json.loads(content)
                    
                    # Convert to standard format
                    signal = 0
                    action = prediction.get('action', '观望')
                    
                    if action in ['买入', '轻仓买入']:
                        signal = 1
                    elif action in ['卖出', '轻仓卖出']:
                        signal = -1
                    
                    if signal == 0:
                        pred_text = prediction.get('prediction', '')
                        confidence = prediction.get('confidence', 0) / 100
                        
                        if '震荡偏涨' in pred_text or '偏涨' in pred_text:
                            if confidence >= 0.60:
                                signal = 1
                        elif '震荡偏跌' in pred_text or '偏跌' in pred_text:
                            if confidence >= 0.60:
                                signal = -1
                        elif '上涨' in pred_text and confidence >= 0.65:
                            signal = 1
                        elif '下跌' in pred_text and confidence >= 0.65:
                            signal = -1
                    
                    result_dict = {
                        'signal': signal,
                        'confidence': prediction['confidence'] / 100,
                        'prediction': prediction['prediction'],
                        'reasoning': prediction.get('reasoning', ''),
                        'support': prediction.get('support_level'),
                        'resistance': prediction.get('resistance_level'),
                        'risk_warning': prediction.get('risk_warning', '')
                    }
                    
                    self.last_predictions[timeframe_desc] = result_dict
                    self.logger.info(f"[{timeframe_desc}] DeepSeek预测: {prediction['prediction']}, "
                                   f"置信度: {prediction['confidence']}%, "
                                   f"建议: {action}")
                    
                    return result_dict
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse DeepSeek response: {str(e)}")
                    return self._fallback_prediction()
            else:
                self.logger.error(f"DeepSeek API error: {response.status_code}")
                return self._fallback_prediction()
                
        except Exception as e:
            self.logger.error(f"Error querying DeepSeek: {str(e)}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self):
        """Fallback prediction when API is unavailable"""
        return {
            'signal': 0,
            'confidence': 0.5,
            'prediction': '观望',
            'reasoning': 'API不可用',
            'support': None,
            'resistance': None,
            'risk_warning': 'API不可用，使用备用分析'
        }
    
    def _analyze_timeframe(self, interval, lookback):
        """Analyze a single timeframe
        
        Args:
            interval (str): Timeframe interval (e.g., '15m', '1h')
            lookback (int): Number of candles to fetch
            
        Returns:
            dict: Analysis results with signal and confidence
        """
        try:
            # Fetch k-lines for this timeframe
            klines = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=interval,
                limit=lookback
            )
            
            if not klines or len(klines) < lookback:
                self.logger.warning(f"Insufficient data for {interval}")
                return None
            
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return None
            
            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)
            if indicators is None:
                return None
            
            # Prepare market analysis
            timeframe_desc = self.timeframes[interval]['description']
            market_analysis = self.prepare_market_analysis(indicators, timeframe_desc)
            if market_analysis is None:
                return None
            
            # Get AI prediction
            prediction = self.query_deepseek_prediction(market_analysis, timeframe_desc)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error analyzing {interval}: {str(e)}")
            return None
    
    def _combine_signals(self, signal_15m, signal_1h, conf_15m, conf_1h):
        """Combine signals from multiple timeframes
        
        Args:
            signal_15m (int): 15m signal (-1, 0, 1)
            signal_1h (int): 1h signal (-1, 0, 1)
            conf_15m (float): 15m confidence (0-1)
            conf_1h (float): 1h confidence (0-1)
            
        Returns:
            dict: Combined signal and probabilities
        """
        # Weights
        weight_15m = self.timeframes['15m']['weight']
        weight_1h = self.timeframes['1h']['weight']
        
        # Weighted signal (considering confidence)
        weighted_signal = (signal_15m * conf_15m * weight_15m + 
                          signal_1h * conf_1h * weight_1h)
        
        # Normalize by total weighted confidence
        total_weight = (conf_15m * weight_15m + conf_1h * weight_1h)
        if total_weight > 0:
            weighted_signal /= total_weight
        
        # Determine final signal
        if weighted_signal > 0.3:
            final_signal = 1  # Long
        elif weighted_signal < -0.3:
            final_signal = -1  # Short
        else:
            final_signal = 0  # Hold
        
        # Calculate probabilities
        probabilities = self._calculate_probabilities(
            signal_15m, signal_1h, conf_15m, conf_1h, weighted_signal
        )
        
        return {
            'signal': final_signal,
            'weighted_score': weighted_signal,
            'probabilities': probabilities
        }
    
    def _calculate_probabilities(self, signal_15m, signal_1h, conf_15m, conf_1h, weighted_signal):
        """Calculate probabilities for long/short/hold
        
        Args:
            signal_15m (int): 15m signal
            signal_1h (int): 1h signal
            conf_15m (float): 15m confidence
            conf_1h (float): 1h confidence
            weighted_signal (float): Weighted combined signal
            
        Returns:
            dict: Probabilities for long, short, hold
        """
        # Base probabilities on weighted signal
        # Map weighted_signal from [-1, 1] to probabilities
        
        # Calculate signal strength (0 to 1)
        signal_strength = abs(weighted_signal)
        
        # Average confidence
        avg_confidence = (conf_15m * 0.4 + conf_1h * 0.6)
        
        # Agreement factor (bonus when both timeframes agree)
        agreement = 1.0 if (signal_15m == signal_1h and signal_15m != 0) else 0.5
        
        # Calculate directional probability
        if weighted_signal > 0:
            # Bullish bias
            long_prob = 0.5 + (signal_strength * avg_confidence * agreement * 0.45)
            short_prob = 0.1 * (1 - signal_strength * avg_confidence)
            hold_prob = 1.0 - long_prob - short_prob
        elif weighted_signal < 0:
            # Bearish bias
            short_prob = 0.5 + (signal_strength * avg_confidence * agreement * 0.45)
            long_prob = 0.1 * (1 - signal_strength * avg_confidence)
            hold_prob = 1.0 - long_prob - short_prob
        else:
            # Neutral
            hold_prob = 0.7
            long_prob = 0.15
            short_prob = 0.15
        
        # Ensure probabilities sum to 1.0 and are non-negative
        total = long_prob + short_prob + hold_prob
        if total > 0:
            long_prob /= total
            short_prob /= total
            hold_prob /= total
        
        return {
            'long': max(0, min(1, long_prob)) * 100,
            'short': max(0, min(1, short_prob)) * 100,
            'hold': max(0, min(1, hold_prob)) * 100
        }
    
    def _prepare_dataframe(self, klines):
        """Convert k-line data to DataFrame
        
        Args:
            klines (list): K-line data
            
        Returns:
            DataFrame: Processed data
        """
        try:
            if not klines or len(klines) < 30:
                self.logger.error("Insufficient k-line data")
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def generate_signal(self, klines=None):
        """Generate multi-timeframe trading signal
        
        Returns:
            int: Combined trading signal (1=buy, -1=sell, 0=hold)
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("开始多时间周期分析")
            self.logger.info("=" * 60)
            
            # Analyze 15m timeframe
            self.logger.info("\n[1/2] 分析 15分钟 周期...")
            result_15m = self._analyze_timeframe('15m', self.timeframes['15m']['lookback'])
            
            if result_15m is None:
                self.logger.error("15m分析失败")
                return 0
            
            signal_15m = result_15m['signal']
            conf_15m = result_15m['confidence']
            
            self.logger.info(f"✓ 15m信号: {signal_15m}, 置信度: {conf_15m:.1%}")
            self.logger.info(f"  预测: {result_15m['prediction']}")
            
            # Analyze 1h timeframe
            self.logger.info("\n[2/2] 分析 1小时 周期...")
            result_1h = self._analyze_timeframe('1h', self.timeframes['1h']['lookback'])
            
            if result_1h is None:
                self.logger.error("1h分析失败")
                return 0
            
            signal_1h = result_1h['signal']
            conf_1h = result_1h['confidence']
            
            self.logger.info(f"✓ 1h信号: {signal_1h}, 置信度: {conf_1h:.1%}")
            self.logger.info(f"  预测: {result_1h['prediction']}")
            
            # Combine signals
            self.logger.info("\n" + "=" * 60)
            self.logger.info("信号组合分析")
            self.logger.info("=" * 60)
            
            combined = self._combine_signals(signal_15m, signal_1h, conf_15m, conf_1h)
            
            final_signal = combined['signal']
            probabilities = combined['probabilities']
            
            # Format output
            self.logger.info(f"\n{'指标':<15} {'15分钟':<15} {'1小时':<15} {'权重':<10}")
            self.logger.info("-" * 60)
            self.logger.info(f"{'信号':<15} {signal_15m:<15} {signal_1h:<15}")
            self.logger.info(f"{'置信度':<15} {conf_15m:<15.1%} {conf_1h:<15.1%}")
            self.logger.info(f"{'权重':<15} {self.timeframes['15m']['weight']:<15.1%} "
                           f"{self.timeframes['1h']['weight']:<15.1%}")
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("最终结果")
            self.logger.info("=" * 60)
            
            signal_name = {1: '做多', -1: '做空', 0: '观望'}
            self.logger.info(f"✓ 综合信号: {final_signal} ({signal_name[final_signal]})")
            self.logger.info(f"✓ 加权评分: {combined['weighted_score']:.3f}")
            
            self.logger.info("\n概率分布:")
            self.logger.info(f"  🟢 做多概率: {probabilities['long']:.1f}%")
            self.logger.info(f"  🔴 做空概率: {probabilities['short']:.1f}%")
            self.logger.info(f"  ⚪ 观望概率: {probabilities['hold']:.1f}%")
            
            self.logger.info("=" * 60 + "\n")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generating multi-timeframe signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """Monitor position - uses generate_signal for decision making"""
        # This strategy focuses on signal generation
        # Position management is handled by the trading manager
        pass
