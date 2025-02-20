from trader import Trader

def main():
    """获取并显示所有合约交易对"""
    try:
        trader = Trader()
        symbols = trader.get_all_symbols()
        
        print("\n可用的USDT合约交易对:")
        print("=" * 50)
        for i, symbol in enumerate(sorted(symbols), 1):
            print(f"{i:3d}. {symbol}")
        print("=" * 50)
        print(f"总计: {len(symbols)} 个交易对")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
