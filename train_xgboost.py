"""
XGBoost模型训练脚本
用于训练交易信号预测模型

Usage:
    python train_xgboost.py --symbol ENAUSDC --start_date 2025-09-01 --end_date 2025-12-04
    python train_xgboost.py --symbol ENAUSDC --start_date 2025-09-01 --end_date 2025-12-04 --output_dir custom/path

File: train_xgboost.py
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import DataLoader as HistoricalDataLoader
from strategies.xgboost_features import XGBoostFeatureAdapter
from strategies.xgboost_trainer import XGBoostTrainer
from utils.logger import Logger
import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for trading signals')
    
    parser.add_argument('--symbol', type=str, default='ENAUSDC',
                       help='Trading symbol (default: ENAUSDC)')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of days for training data (overrides start_date/end_date)')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for training data (ignored if --days is set)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for training data (ignored if --days is set)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Kline interval (default: 15m)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: strategies/models/{symbol}/)')
    
    args = parser.parse_args()
    
    # Calculate dates based on --days or use defaults
    if args.days is not None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        args.start_date = (datetime.now() - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        # Default to 90 days if no dates specified
        if args.start_date is None or args.end_date is None:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
            args.start_date = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    
    return args


def generate_output_dir(symbol: str) -> str:
    """Auto-generate output directory path based on symbol
    
    Args:
        symbol: Trading symbol (e.g., 'ENAUSDC' or 'ENA/USDC')
        
    Returns:
        Output directory path as strategies/models/{symbol_lowercase}/
    """
    # Normalize symbol: remove '/' and convert to lowercase
    symbol_normalized = symbol.lower().replace('/', '')
    return f'strategies/models/{symbol_normalized}/'


def main():
    """Main training function"""
    args = parse_args()
    
    # Initialize logger
    logger = Logger.get_logger()
    
    # Convert symbol format for ccxt
    symbol = args.symbol
    if '/' not in symbol:
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            quote = 'USDT'
        elif symbol.endswith('USDC'):
            base = symbol[:-4]
            quote = 'USDC'
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")
        symbol = f"{base}/{quote}"
    
    # Auto-generate output directory based on symbol if not specified
    if args.output_dir is None:
        args.output_dir = generate_output_dir(args.symbol)
        logger.info(f"Auto-generated output directory: {args.output_dir}")
    
    # Get training config
    xgb_config = config.XGBOOST_STRATEGY_CONFIG
    training_config = xgb_config.get('training', {})
    
    logger.info("=" * 80)
    logger.info("XGBOOST MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Symbol:          {symbol}")
    logger.info(f"Period:          {args.start_date} to {args.end_date}")
    logger.info(f"Interval:        {args.interval}")
    logger.info(f"Output Dir:      {args.output_dir}")
    logger.info(f"IC Target:       {training_config.get('ic_target', 0.05)}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load historical data
        logger.info("Step 1: Loading historical data...")
        data_loader = HistoricalDataLoader()
        df = data_loader.load_data(
            symbol=symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if df is None or len(df) == 0:
            logger.error("Failed to load data")
            return 1
        
        logger.info(f"Loaded {len(df)} candles")
        
        # Step 2: Prepare features and labels
        logger.info("Step 2: Preparing features and labels...")
        feature_adapter = XGBoostFeatureAdapter()
        
        future_periods = training_config.get('future_periods', 5)
        threshold = training_config.get('threshold', 0.002)
        
        X, y = feature_adapter.prepare_training_data(
            df,
            future_periods=future_periods,
            threshold=threshold
        )
        
        if X is None or y is None:
            logger.error("Failed to prepare training data")
            return 1
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_names = ['跌', '持', '涨']
        logger.info("Class distribution:")
        for u, c in zip(unique, counts):
            logger.info(f"  {class_names[u]}: {c} ({c/len(y)*100:.1f}%)")
        
        # Step 3: Normalize features
        logger.info("Step 3: Normalizing features...")
        X = feature_adapter.normalize_features(X, fit=True)
        
        # Step 4: Split data
        logger.info("Step 4: Splitting train/validation sets...")
        test_size = training_config.get('test_size', 0.2)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Step 5: Initialize trainer and train model
        logger.info("Step 5: Training XGBoost model...")
        trainer = XGBoostTrainer()
        
        feature_names = feature_adapter.get_feature_names()
        trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
        
        # Step 6: Evaluate model
        logger.info("Step 6: Evaluating model...")
        metrics = trainer.evaluate(X_val, y_val)
        
        if not metrics:
            logger.error("Failed to evaluate model")
            return 1
        
        # Step 7: Check IC and log warning if needed
        logger.info("Step 7: Validating IC...")
        ic_value = metrics.get('ic', 0.0)
        ic_target = training_config.get('ic_target', 0.05)
        
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)
        logger.info(f"\n{metrics.get('classification_report', 'N/A')}")
        
        logger.info("CONFUSION MATRIX")
        cm = metrics.get('confusion_matrix', [])
        logger.info(f"\n{np.array(cm)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("IC VALIDATION")
        logger.info("=" * 60)
        logger.info(f"IC Value:  {ic_value:.4f}")
        logger.info(f"IC Target: {ic_target:.4f}")
        
        if ic_value <= ic_target:
            logger.warning("⚠️ IC未达标! IC <= 0.05")
            logger.warning("建议:")
            logger.warning("  - 调整超参数 (max_depth, learning_rate, n_estimators)")
            logger.warning("  - 增加训练数据量")
            logger.warning("  - 尝试不同的特征组合")
            logger.warning("  - 调整涨跌阈值 (threshold)")
        else:
            logger.info("✅ IC达标! IC > 0.05")
        
        # Step 8: Output feature importance
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE RANKINGS")
        logger.info("=" * 60)
        
        importance = trainer.get_feature_importance()
        if importance:
            for i, (feat, score) in enumerate(importance.items(), 1):
                logger.info(f"  {i:2d}. {feat}: {score:.4f}")
        else:
            logger.warning("无法获取特征重要性")
        
        # Step 9: Save model and scaler
        logger.info("\nStep 9: Saving model and scaler...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / xgb_config.get('model_filename', 'xgboost_model.json')
        scaler_path = output_dir / xgb_config.get('scaler_filename', 'xgboost_scaler.npz')
        
        trainer.save_model(str(model_path))
        feature_adapter.save_scaler(str(scaler_path))
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"F1 Score:  {metrics.get('f1', 0):.4f}")
        logger.info(f"IC Value:  {ic_value:.4f}")
        logger.info(f"IC Target: {ic_target:.4f}")
        
        if ic_value > ic_target:
            logger.info("✅ IC TARGET ACHIEVED!")
        else:
            logger.info(f"⚠️ IC target not reached. Gap: {ic_target - ic_value:.4f}")
        
        logger.info(f"\nModel saved to: {model_path}")
        logger.info(f"Scaler saved to: {scaler_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
