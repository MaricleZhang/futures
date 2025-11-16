# MEXC现货交易系统使用指南

## 概述

MEXC现货交易系统是一个完全独立的自动化交易模块，与现有的Binance合约交易系统完全隔离，可以同时运行而互不干扰。

## 主要特点

- ✅ **完全隔离**：独立的配置、交易器、管理器，与Binance合约交易互不影响
- ✅ **现货交易**：纯现货买卖，无杠杆，风险相对较低
- ✅ **多策略支持**：支持KAMA-ROC-ADX和Simple ADX-DI策略
- ✅ **多币对并发**：支持同时交易多个币对，每个币对独立线程
- ✅ **自动重试**：API调用失败自动重试，增强稳定性
- ✅ **线程监控**：自动检测并重启异常线程

## 文件结构

```
futures/
├── config_mexc_spot.py              # MEXC现货交易配置文件
├── mexc_spot_trader.py              # MEXC现货交易器
├── mexc_spot_trading_manager.py     # MEXC现货交易管理器
├── main_mexc_spot.py                # MEXC现货交易启动脚本
├── .env.mexc.example                # MEXC环境变量示例
└── .env.mexc                        # MEXC环境变量（需自行创建）
```

## 安装与配置

### 1. 配置API密钥

复制环境变量示例文件并填入你的MEXC API密钥：

```bash
cp .env.mexc.example .env.mexc
```

编辑 `.env.mexc` 文件，填入你的API密钥：

```
MEXC_API_KEY=你的MEXC_API密钥
MEXC_SECRET_KEY=你的MEXC密钥
```

**安全建议：**
- 仅开启现货交易权限
- 不要开启提现权限
- 建议设置IP白名单

### 2. 配置交易参数

编辑 `config_mexc_spot.py` 文件：

```python
# 选择要交易的币对
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

# 配置每个币对的参数
SYMBOL_CONFIGS = {
    'BTC/USDT': {
        'min_notional': 10,          # 最小交易金额（USDT）
        'trade_amount_percent': 30,  # 单次交易占可用余额的百分比
        'check_interval': 60,        # 检查间隔（秒）
    },
    # ... 更多币对配置
}

# 选择策略
STRATEGY_TYPE = 'kama_roc_adx'  # 可选: 'kama_roc_adx', 'simple_adx_di'
```

### 3. 代理设置（可选）

如果需要使用代理访问MEXC API，在 `config_mexc_spot.py` 中配置：

```python
USE_PROXY = True
PROXY_URL = 'http://127.0.0.1:7890'
```

## 使用方法

### 启动MEXC现货交易

```bash
python main_mexc_spot.py
```

### 停止交易

按 `Ctrl+C` 优雅退出，系统会自动清理资源。

### 同时运行两个系统

MEXC现货交易系统和Binance合约交易系统可以同时运行：

```bash
# 终端1：运行Binance合约交易
python main.py

# 终端2：运行MEXC现货交易
python main_mexc_spot.py
```

两个系统完全独立，互不干扰。

## 交易逻辑

### 现货交易流程

1. **买入信号（signal=1）**：
   - 如果没有持仓，使用可用余额的配置百分比买入

2. **卖出信号（signal=-1）**：
   - 如果有持仓，卖出所有持仓

3. **平仓信号（signal=2）**：
   - 如果有持仓，卖出所有持仓

4. **观望信号（signal=0）**：
   - 不执行任何操作

### 与Binance合约交易的区别

| 特性 | Binance合约 | MEXC现货 |
|------|------------|----------|
| 交易类型 | 合约（期货） | 现货 |
| 杠杆 | 支持（最高10倍） | 无杠杆 |
| 做空 | 支持 | 不支持 |
| 保证金模式 | 全仓/逐仓 | 无 |
| 风险 | 较高 | 较低 |
| 交易方向 | 开多/开空/平仓 | 买入/卖出 |

## 策略说明

### 1. KAMA-ROC-ADX策略

基于趋势跟踪的策略，适合趋势明显的市场。

**核心指标：**
- KAMA（自适应移动平均线）
- ROC（变化率）
- ADX（平均趋向指数）

**配置：**
```python
STRATEGY_TYPE = 'kama_roc_adx'
```

### 2. Simple ADX-DI策略

简单高效的趋势策略，适合快速反应。

**核心指标：**
- ADX（趋势强度）
- +DI/-DI（方向指标）

**配置：**
```python
STRATEGY_TYPE = 'simple_adx_di'
```

## 风险控制

1. **资金管理**：
   - 建议单次交易不超过总资金的30%
   - 可通过 `trade_amount_percent` 配置

2. **交易频率**：
   - 通过 `check_interval` 控制检查间隔
   - 通过 `MIN_TRADE_INTERVAL` 防止过度交易

3. **最小交易额**：
   - 通过 `min_notional` 避免小额交易手续费损耗

## 日志查看

系统会自动生成日志文件，记录所有交易操作：

- 每个币对有独立的日志标识（如 `MEXC_BTC/USDT`）
- 日志包含：余额、价格、信号、持仓、交易操作等

## 常见问题

### Q1: 可以同时运行MEXC和Binance系统吗？

可以。两个系统使用不同的配置文件、环境变量和交易器，完全独立运行。

### Q2: MEXC现货交易支持做空吗？

不支持。现货交易只能买入后卖出，无法做空。

### Q3: 如何添加新的交易对？

在 `config_mexc_spot.py` 中：
1. 将币对添加到 `SYMBOLS` 列表
2. 在 `SYMBOL_CONFIGS` 中添加对应配置

### Q4: API密钥权限需要哪些？

- ✅ 现货交易权限
- ❌ 提现权限（不建议开启）
- ✅ 读取权限

### Q5: 如何切换策略？

修改 `config_mexc_spot.py` 中的 `STRATEGY_TYPE` 配置，然后重启程序。

## 安全建议

1. **API密钥安全**：
   - 不要分享你的API密钥
   - `.env.mexc` 文件应添加到 `.gitignore`
   - 定期更换API密钥

2. **资金安全**：
   - 建议先小资金测试
   - 不要开启提现权限
   - 设置IP白名单

3. **监控**：
   - 定期检查交易日志
   - 关注账户余额变化
   - 及时调整策略参数

## 技术支持

如有问题，请检查：
1. API密钥是否正确配置
2. 网络连接是否正常
3. 代理设置是否正确
4. 日志文件中的错误信息

## 隔离架构说明

```
┌─────────────────────────────────────┐
│   Binance合约交易系统                 │
├─────────────────────────────────────┤
│ • config.py                          │
│ • trader.py                          │
│ • trading_manager.py                 │
│ • main.py                            │
│ • .env (BINANCE_API_KEY)            │
└─────────────────────────────────────┘

         ⬇️  完全隔离  ⬇️

┌─────────────────────────────────────┐
│   MEXC现货交易系统                    │
├─────────────────────────────────────┤
│ • config_mexc_spot.py               │
│ • mexc_spot_trader.py               │
│ • mexc_spot_trading_manager.py      │
│ • main_mexc_spot.py                 │
│ • .env.mexc (MEXC_API_KEY)          │
└─────────────────────────────────────┘

        共享组件（只读）
┌─────────────────────────────────────┐
│ • strategies/ (策略模块)             │
│ • utils/logger.py (日志工具)         │
└─────────────────────────────────────┘
```

两个系统共享策略和工具模块，但使用独立的配置、交易器和启动脚本，确保完全隔离。
