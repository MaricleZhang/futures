from binance_futures_trader import BinanceFuturesTrader

def main():
    """获取并显示所有合约交易对"""
    trader = BinanceFuturesTrader()
    symbols = trader.get_all_symbols()
    
    print("\n可用的USDT合约交易对:")
    print("=" * 50)
    for i, symbol in enumerate(sorted(symbols), 1):
        print(f"{i:3d}. {symbol}")
    print("=" * 50)
    print(f"总计: {len(symbols)} 个交易对")

if __name__ == "__main__":
    main()
