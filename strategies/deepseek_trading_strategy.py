"""
DeepSeek AI Trading Strategy for Short-term Trading
Uses DeepSeek API to predict price trends based on technical indicators and market data

File: strategies/deepseek_trading_strategy.py
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

class DeepSeekTradingStrategy(BaseStrategy):
    """
    DeepSeek AI-powered short-term trading strategy
    
    Strategy Logic:
    1. Collects multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
    2. Sends market data to DeepSeek API for trend prediction
    3. Combines AI prediction with traditional indicators for confirmation
    4. Executes trades based on high-confidence signals
    
    Trading Signals:
    - Buy: AI predicts uptrend + technical indicators confirm
    - Sell: AI predicts downtrend + technical indicators confirm
    - Hold: Low confidence or conflicting signals
    """
    
    def __init__(self, trader, deepseek_api_key=None, interval='15m'):
        """Initialize the DeepSeek trading strategy
        
        Args:
            trader: Trader instance
            deepseek_api_key: DeepSeek API key (optional, can be set from environment)
            interval: K-line interval (default: '15m')
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
            total=3,  # 总重试次数
            backoff_factor=1,  # 重试间隔倍数
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["POST"]  # 允许重试的HTTP方法
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Timeout configuration
        self.api_timeout = (10, 30)  # (连接超时, 读取超时) 秒
        
        # Timeframe configuration (短线交易使用较短时间框架)
        self.kline_interval = interval  # Use the interval passed from backtest
        # Convert interval to seconds for check_interval
        interval_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 300, '30m': 600,
            '1h': 900, '2h': 1800, '4h': 3600, '6h': 7200, '12h': 14400, '1d': 28800
        }
        self.check_interval = interval_map.get(interval, 300)  # Default to 5 minutes
        self.lookback_period = 100  # 分析用的K线数量
        self.training_lookback = 100  # 与TradingManager兼容
        
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
        self.min_confidence = 0.65  # 提高最小信号置信度到65%，减少噪音信号
        self.stop_loss_pct = 0.015  # 1.5% 止损
        self.take_profit_pct = 0.045  # 4.5% 止盈
        self.max_position_hold_time = 360  # 6小时最大持仓时间（分钟）
        
        # Trailing Stop Configuration
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.02  # 2%利润激活追踪止损
        self.trailing_stop_distance = 0.008  # 0.8%追踪距离
        
        # Position Tracking
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_signal = 0
        self.last_signal_time = None
        self.last_ai_prediction = None
        
        # Signal Stability Control
        self.min_signal_hold_time = 900  # 信号最小持续时间15分钟（秒）
        self.signal_change_cooldown = 600  # 信号切换冷却时间10分钟（秒）
        self.last_signal_change_time = 0  # 上次信号切换时间
        self.signal_confirmation_count = 0  # 连续确认计数
        self.min_confirmation_count = 2  # 需要连续确认的次数
        
        # Rate Limiting
        self.min_time_between_api_calls = getattr(config, 'DEEPSEEK_API_MIN_INTERVAL', 10)  # 最小API调用间隔（秒）
        self.last_api_call_time = 0
        
        # Configure HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=getattr(config, 'DEEPSEEK_API_RETRY_COUNT', 3),  # 总重试次数
            backoff_factor=getattr(config, 'DEEPSEEK_API_RETRY_BACKOFF', 1),  # 重试间隔倍数
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["POST"]  # 允许重试的HTTP方法
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Timeout configuration
        connect_timeout = getattr(config, 'DEEPSEEK_API_CONNECT_TIMEOUT', 10)
        read_timeout = getattr(config, 'DEEPSEEK_API_READ_TIMEOUT', 30)
        self.api_timeout = (connect_timeout, read_timeout)  # (连接超时, 读取超时) 秒
        
        self.logger.info("DeepSeek Trading Strategy initialized for 5m timeframe")
        self.logger.info(f"API Key configured: {bool(self.deepseek_api_key)}")
        self.logger.info(f"API timeout configured: {self.api_timeout} seconds")
    
    def cleanup(self):
        """清理资源，关闭HTTP会话"""
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
        """
        Calculate comprehensive technical indicators
        
        Args:
            df (DataFrame): OHLCV data
            
        Returns:
            dict: Dictionary containing all calculated indicators
        """
        try:
            # Extract price data
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI (Relative Strength Index)
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
            # BB columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            bb_lower = bb_df[f'BBL_{self.bb_period}_{float(self.bb_std)}'].values
            bb_middle = bb_df[f'BBM_{self.bb_period}_{float(self.bb_std)}'].values
            bb_upper = bb_df[f'BBU_{self.bb_period}_{float(self.bb_std)}'].values
            
            # ATR (Average True Range)
            atr = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=self.atr_period).values
            
            # Moving Averages
            ema_fast = ta.ema(pd.Series(close), length=10).values
            ema_slow = ta.ema(pd.Series(close), length=30).values
            
            # Volume indicators
            volume_ma = ta.sma(pd.Series(volume), length=self.volume_ma_period).values
            
            # Stochastic Oscillator
            # talib defaults: fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            stoch_df = ta.stoch(pd.Series(high), pd.Series(low), pd.Series(close), k=5, d=3, smooth_k=3)
            # STOCH columns: STOCHk_5_3_3, STOCHd_5_3_3
            slowk = stoch_df['STOCHk_5_3_3'].values
            slowd = stoch_df['STOCHd_5_3_3'].values
            
            # ADX (Average Directional Index)
            adx_df = ta.adx(pd.Series(high), pd.Series(low), pd.Series(close), length=14)
            adx = adx_df['ADX_14'].values
            
            # Price momentum
            momentum = ta.mom(pd.Series(close), length=10).values
            
            # Calculate BB width (volatility indicator)
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
    
    def prepare_market_analysis(self, indicators):
        """
        Prepare market analysis text for DeepSeek API
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            str: Formatted market analysis text
        """
        try:
            # Get current values (last candle)
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
            
            # Price trend analysis
            recent_prices = indicators['close'][-10:]
            trend = "上升" if recent_prices[-1] > recent_prices[0] else "下降"
            
            analysis = f"""
当前市场分析报告 ({self.trader.symbol}):

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
    
    def query_deepseek_prediction(self, market_analysis):
        """
        Query DeepSeek API for price prediction
        
        Args:
            market_analysis (str): Formatted market analysis
            
        Returns:
            dict: Prediction results with signal and confidence
        """
        # Check rate limiting
        current_time = time.time()
        if current_time - self.last_api_call_time < self.min_time_between_api_calls:
            self.logger.debug("Rate limit: using cached prediction")
            return self.last_ai_prediction
        
        if not self.deepseek_api_key:
            self.logger.warning("No API key available, using fallback logic")
            return self._fallback_prediction(market_analysis)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""你是一位专业的加密货币短线交易分析师。请基于以下市场数据分析，预测未来15-45分钟的价格趋势。

{market_analysis}

**核心分析原则**：
1. **趋势优先**：首先识别当前主要趋势方向，短期波动不应违背主趋势
2. **价格行为验证**：技术指标必须与实际价格走势一致，如果冲突则以价格行为为准
3. **多重确认**：至少需要2-3个指标同向确认才给出明确信号
4. **保守操作**：宁可错过机会也不要在不确定时强行交易

**具体指引**：
- 当短期趋势明显上升时，不要轻易给出"震荡偏跌"信号
- 当短期趋势明显下降时，不要轻易给出"震荡偏涨"信号
- 震荡偏向信号需要至少75%置信度，且要有明确的技术支撑
- 如果价格连续上涨但某些指标显示超买，优先考虑"震荡"而非"偏跌"
- 置信度评估要考虑信号的一致性，指标冲突时应降低置信度

**置信度标准**：
- 90%+: 多个强势指标同向，趋势非常明确
- 75-89%: 主要指标同向，有少量分歧
- 65-74%: 指标基本同向，但存在一定不确定性
- 50-64%: 信号混合，建议观望
- <50%: 信号不明确或冲突严重

请提供:
1. 趋势预测: 上涨/下跌/震荡/震荡偏涨/震荡偏跌
2. 置信度: 0-100的数值（严格按照上述标准）
3. 建议操作: 买入/卖出/轻仓买入/轻仓卖出/观望
4. 关键支撑/阻力位
5. 分析推理和风险提示

请用JSON格式回复:
{{
    "prediction": "上涨/下跌/震荡偏涨/震荡偏跌/震荡",
    "confidence": 75,
    "action": "买入/卖出/轻仓买入/轻仓卖出/观望",
    "support_level": 价格,
    "resistance_level": 价格,
    "reasoning": "详细说明分析逻辑和指标一致性",
    "risk_warning": "具体的风险点和注意事项"
}}"""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位经验丰富的加密货币短线交易分析师，具有以下特质：1）严格遵循趋势分析原则，不会在明显趋势中给出相反信号；2）重视价格行为和成交量确认；3）保守谨慎，只在高确定性时给出交易建议；4）善于识别真实突破和假突破；5）注重风险控制，避免频繁交易。"
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
            
            self.last_api_call_time = current_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    # Extract JSON from markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    prediction = json.loads(content)
                    
                    # Convert to standard format with improved logic
                    signal = 0
                    
                    # 首先检查action字段
                    action = prediction.get('action', '观望')
                    if action in ['买入', '轻仓买入']:
                        signal = 1
                    elif action in ['卖出', '轻仓卖出']:
                        signal = -1
                    
                    # 如果action是观望，检查prediction字段是否包含方向信息
                    if signal == 0:
                        pred_text = prediction.get('prediction', '')
                        confidence = prediction.get('confidence', 0) / 100
                        
                        # 震荡偏涨/偏跌也应该产生信号（但需要较高置信度）
                        if '震荡偏涨' in pred_text or '偏涨' in pred_text:
                            if confidence >= 0.60:  # 震荡市场需要更高置信度
                                signal = 1
                                self.logger.info(f"震荡偏涨信号被激活，置信度: {confidence:.0%}")
                        elif '震荡偏跌' in pred_text or '偏跌' in pred_text:
                            if confidence >= 0.60:
                                signal = -1
                                self.logger.info(f"震荡偏跌信号被激活，置信度: {confidence:.0%}")
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
                    
                    self.last_ai_prediction = result_dict
                    self.logger.info(f"DeepSeek预测: {prediction['prediction']}, "
                                   f"置信度: {prediction['confidence']}%, "
                                   f"建议: {prediction['action']}")
                    self.logger.debug(f"分析原因: {prediction.get('reasoning', '')}")
                    
                    return result_dict
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse DeepSeek response: {str(e)}")
                    self.logger.debug(f"Response content: {content}")
                    return self._fallback_prediction(market_analysis)
            else:
                self.logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return self._fallback_prediction(market_analysis)
                
        except requests.exceptions.ConnectTimeout:
            self.logger.error("DeepSeek API connection timeout - 无法连接到服务器")
            return self._fallback_prediction(market_analysis)
        except requests.exceptions.ReadTimeout:
            self.logger.error("DeepSeek API read timeout - 服务器响应超时")
            return self._fallback_prediction(market_analysis)
        except requests.exceptions.Timeout:
            self.logger.error("DeepSeek API timeout")
            return self._fallback_prediction(market_analysis)
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"DeepSeek API connection error: {str(e)}")
            return self._fallback_prediction(market_analysis)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek API request error: {str(e)}")
            return self._fallback_prediction(market_analysis)
        except Exception as e:
            self.logger.error(f"Unexpected error querying DeepSeek: {str(e)}")
            return self._fallback_prediction(market_analysis)
    
    def _fallback_prediction(self, market_analysis):
        """
        Fallback prediction logic when API is unavailable
        Uses traditional technical analysis
        """
        self.logger.info("Using fallback technical analysis")
        
        # This will be filled by the technical indicator analysis
        # For now, return neutral
        return {
            'signal': 0,
            'confidence': 0.5,
            'prediction': '观望',
            'reasoning': '使用技术指标分析',
            'support': None,
            'resistance': None,
            'risk_warning': 'API不可用，使用备用分析'
        }
    
    def generate_signal(self, klines=None):
        """
        Generate trading signal combining AI prediction and technical analysis
        
        Args:
            klines (list): K-line data (optional, will fetch if not provided)
            
        Returns:
            int: Trading signal (1=buy, -1=sell, 0=hold)
        """
        try:
            # Fetch k-lines if not provided
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            # Check if we have enough data
            if not klines or len(klines) < self.lookback_period:
                self.logger.warning("Insufficient k-line data for analysis")
                return 0
            
            # Convert to DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(df)
            if indicators is None:
                return 0
            
            # Prepare market analysis
            market_analysis = self.prepare_market_analysis(indicators)
            if market_analysis is None:
                return 0
            
            # Get AI prediction
            ai_prediction = self.query_deepseek_prediction(market_analysis)
            if ai_prediction is None:
                return 0
            
            # Technical confirmation
            tech_signal = self._get_technical_confirmation(indicators)
            
            # Combine AI prediction with technical confirmation
            final_signal = 0
            ai_signal = ai_prediction['signal']
            confidence = ai_prediction['confidence']
            
            # Only trade if confidence is high enough
            if confidence >= self.min_confidence:
                return ai_signal
                # AI and technical indicators agree (strongest signal)
                if ai_signal == tech_signal and tech_signal != 0:
                    final_signal = ai_signal
                    self.logger.info(f"🎯 Strong signal: AI={ai_signal}, Technical={tech_signal}, "
                                   f"Confidence={confidence:.2%}")
                # AI signal strong but technical neutral (trust AI)
                elif ai_signal != 0 and tech_signal == 0 and confidence >= 0.70:
                    final_signal = ai_signal
                    self.logger.info(f"🤖 AI-led signal: {ai_signal}, "
                                   f"Confidence={confidence:.2%} (Technical neutral)")
                # Technical strong but AI neutral (use technical)
                elif ai_signal == 0 and tech_signal != 0 and confidence >= 0.60:
                    final_signal = tech_signal
                    self.logger.info(f"📊 Technical-led signal: {tech_signal}, "
                                   f"AI confidence={confidence:.2%}")
                # Conflicting signals (AI and technical disagree)
                elif ai_signal != 0 and tech_signal != 0 and ai_signal != tech_signal:
                    self.logger.info(f"⚠️ Conflicting signals - AI: {ai_signal}, "
                                   f"Technical: {tech_signal}, holding position")
                # Both neutral (market unclear)
                else:
                    if ai_signal == 0 and tech_signal == 0:
                        self.logger.info(f"😴 Market neutral - both AI and Technical suggest holding")
                    else:
                        self.logger.info(f"🤔 Weak signals - AI: {ai_signal}, Technical: {tech_signal}, "
                                       f"confidence {confidence:.2%} insufficient")
            else:
                self.logger.info(f"📉 Low confidence: {confidence:.2%} (< {self.min_confidence:.0%}), holding position")
            
            # Update signal tracking
            if final_signal != 0:
                self.last_signal = final_signal
                self.last_signal_time = time.time()
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _get_technical_confirmation(self, indicators):
        """
        Get technical indicator confirmation
        
        Args:
            indicators (dict): Technical indicators
            
        Returns:
            int: Technical signal (1=buy, -1=sell, 0=neutral)
        """
        try:
            # Get current values
            rsi = indicators['rsi'][-1]
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            ema_fast = indicators['ema_fast'][-1]
            ema_slow = indicators['ema_slow'][-1]
            slowk = indicators['slowk'][-1]
            adx = indicators['adx'][-1]
            
            close = indicators['close'][-1]
            bb_upper = indicators['bb_upper'][-1]
            bb_lower = indicators['bb_lower'][-1]
            
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI signals
            if rsi < 30:
                bullish_signals += 2  # Strong oversold
            elif rsi < 40:
                bullish_signals += 1  # Mild oversold
            elif rsi > 70:
                bearish_signals += 2  # Strong overbought
            elif rsi > 60:
                bearish_signals += 1  # Mild overbought
            
            # MACD signals
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # EMA signals
            if ema_fast > ema_slow:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Stochastic signals
            if slowk < 20:
                bullish_signals += 1
            elif slowk > 80:
                bearish_signals += 1
            
            # Bollinger Bands signals
            if close < bb_lower:
                bullish_signals += 1
            elif close > bb_upper:
                bearish_signals += 1
            
            # ADX trend strength requirement
            if adx < 20:
                # Weak trend, be more conservative
                return 0
            
            # Decision logic
            if bullish_signals >= 3 and bullish_signals > bearish_signals:
                return 1
            elif bearish_signals >= 3 and bearish_signals > bullish_signals:
                return -1
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error in technical confirmation: {str(e)}")
            return 0
    
    def _prepare_dataframe(self, klines):
        """
        Convert k-line data to DataFrame
        
        Args:
            klines (list): K-line data
            
        Returns:
            DataFrame: Processed data
        """
        try:
            if not klines or len(klines) < 30:
                self.logger.error("Insufficient k-line data")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def monitor_position(self):
        """Monitor current position and execute trading logic"""
        try:
            # Get current position
            position = self.trader.get_position()
            
            # No position - check for entry signals
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # Generate trading signal
                signal = self.generate_signal()
                
                if signal != 0:
                    # Get account balance
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # Get current price
                    current_price = self.trader.get_market_price()
                    
                    # Calculate trade amount (using config percentage)
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # Execute trade
                    if signal == 1:  # Buy signal
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"LONG position opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    elif signal == -1:  # Sell signal
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"SHORT position opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    
                    # Record entry details
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # Position exists - manage it
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"Error monitoring position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """
        Manage existing position with risk controls
        
        Args:
            position: Current position information
        """
        try:
            # Extract position details
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_side = "long" if position_amount > 0 else "short"
            
            # Calculate profit/loss percentage
            if position_side == "long":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price
            
            # Update maximum profit reached
            if profit_rate > self.max_profit_reached:
                self.max_profit_reached = profit_rate
                self.logger.debug(f"New max profit: {self.max_profit_reached:.3%}")
            
            # Check holding time
            if self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60  # minutes
                if holding_time >= self.max_position_hold_time:
                    self.logger.info(f"Maximum holding time reached ({holding_time:.1f} min), closing position")
                    self.trader.close_position()
                    return
            
            # Check stop loss
            if profit_rate <= -self.stop_loss_pct:
                self.logger.info(f"STOP LOSS triggered at {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check take profit
            if profit_rate >= self.take_profit_pct:
                self.logger.info(f"TAKE PROFIT triggered at {profit_rate:.3%}")
                self.trader.close_position()
                return
            
            # Check trailing stop
            if self.trailing_stop_enabled and profit_rate >= self.trailing_stop_activation:
                drawdown = self.max_profit_reached - profit_rate
                if drawdown >= self.trailing_stop_distance:
                    self.logger.info(f"TRAILING STOP triggered - Max profit: {self.max_profit_reached:.3%}, "
                                   f"Current: {profit_rate:.3%}, Drawdown: {drawdown:.3%}")
                    self.trader.close_position()
                    return
            
            # Check for trend reversal (less frequent to avoid API rate limits)
            current_time = time.time()
            if current_time - self.last_api_call_time >= self.check_interval:
                signal = self.generate_signal()
                if (position_side == "long" and signal == -1) or (position_side == "short" and signal == 1):
                    self.logger.info(f"TREND REVERSAL detected, closing {position_side} position")
                    self.trader.close_position()
                    return
            
            # Log current position status
            self.logger.debug(f"Position status - Side: {position_side}, P/L: {profit_rate:.3%}, "
                            f"Price: {current_price}, Entry: {entry_price}")
                
        except Exception as e:
            self.logger.error(f"Error managing position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
