#!/bin/bash
# 市场扫描器演示脚本
# 用于快速测试市场扫描功能

echo "=================================="
echo "市场扫描器功能演示"
echo "=================================="
echo ""

# 检查依赖
echo "1. 检查依赖..."
python -c "import schedule; print('✓ schedule 已安装')" 2>/dev/null || echo "✗ schedule 未安装，请运行: pip install schedule"
python -c "from market_scanner import MarketScanner; print('✓ market_scanner 模块正常')" 2>/dev/null || echo "✗ market_scanner 模块导入失败"
echo ""

# 显示帮助信息
echo "2. 查看命令行参数..."
python run_market_scanner.py --help
echo ""

# 提供使用示例
echo "=================================="
echo "使用示例："
echo "=================================="
echo ""
echo "# 立即执行一次扫描"
echo "python run_market_scanner.py --once"
echo ""
echo "# 使用 simple_adx_di 策略扫描，显示前10个结果"
echo "python run_market_scanner.py --once --strategy simple_adx_di --top-n 10"
echo ""
echo "# 每6小时定时扫描（默认）"
echo "python run_market_scanner.py"
echo ""
echo "# 每2小时定时扫描"
echo "python run_market_scanner.py --interval 2"
echo ""
echo "# 后台运行"
echo "nohup python run_market_scanner.py > logs/market_scanner.log 2>&1 &"
echo ""
echo "=================================="
echo "提示："
echo "- 首次运行建议使用 --once 参数测试功能"
echo "- AI策略需要配置API密钥（.env文件）"
echo "- 技术指标策略（simple_adx_di）不需要API密钥"
echo "- 扫描结果保存在 data/market_scan_results/ 目录"
echo "=================================="
