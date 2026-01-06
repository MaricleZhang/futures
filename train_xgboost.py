"""
XGBoost模型训练脚本
用于训练交易信号预测模型

Usage:
    python train_xgboost.py --symbol ZECUSDT --start_date 2025-01-01 --end_date 2025-10-01
    python train_xgboost.py --symbol ZECUSDT --days 180 --interval 15m

File: train_xgboost.py
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import DataLoader as HistoricalDataLoader
from strategies.xgboost_features import XGBoostFeatureExtractor
from utils.logger import Logger
import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for trading signals')
    
    parser.add_argument('--symbol', type=str, default='ZECUSDT',
                       help='Trading symbol (default: ZECUSDT)')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of days for training data (overrides start_date/end_date)')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for training data')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for training data')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Kline interval (default: 15m)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: strategies/models/{symbol})')
    
    args = parser.parse_args()
    
    # Calculate dates
    if args.days is not None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        args.start_date = (datetime.now() - pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        if args.start_date is None or args.end_date is None:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
            args.start_date = (datetime.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    
    return args


def main():
    """Main training function"""
    args = parse_args()
    
    # Initialize logger
    logger = Logger.get_logger()
    
    # Convert symbol format
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
    
    # Auto-generate output directory
    if args.output_dir is None:
        symbol_dir = args.symbol.lower().replace('/', '')
        args.output_dir = f'strategies/models/{symbol_dir}'
    
    # XGBoost config
    xgb_config = config.XGBOOST_STRATEGY_CONFIG
    training_config = xgb_config.get('training', {})
    
    logger.info("=" * 80)
    logger.info("XGBOOST MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Symbol:          {symbol}")
    logger.info(f"Period:          {args.start_date} to {args.end_date}")
    logger.info(f"Interval:        {args.interval}")
    logger.info(f"Output dir:      {args.output_dir}")
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
        
        # Step 2: Calculate features and prepare training data
        logger.info("Step 2: Calculating features and preparing data...")
        lookback = xgb_config.get('lookback_period', 150)
        feature_extractor = XGBoostFeatureExtractor(lookback_period=lookback)
        
        future_periods = training_config.get('future_periods', 5)
        threshold = training_config.get('threshold', 0.003)
        
        X, y = feature_extractor.prepare_training_data(
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
        X = feature_extractor.normalize_features(X, fit=True)
        
        # Save scaler
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = output_dir / 'xgboost_scaler.npz'
        feature_extractor.save_scaler(str(scaler_path))
        
        # Step 4: Split data
        logger.info("Step 4: Splitting train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Step 5: Train XGBoost model
        logger.info("Step 5: Training XGBoost model...")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': xgb_config.get('max_depth', 6),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'min_child_weight': xgb_config.get('min_child_weight', 3),
            'subsample': xgb_config.get('subsample', 0.8),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
            'eval_metric': ['mlogloss', 'merror'],
            'seed': 42
        }
        
        n_estimators = xgb_config.get('n_estimators', 200)
        early_stopping = training_config.get('early_stopping', 20)
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=10
        )
        
        # Step 6: Save model
        model_path = output_dir / 'xgboost_model.json'
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Step 7: Evaluate
        logger.info("Step 7: Evaluating model...")
        
        y_pred_proba = model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Accuracy
        accuracy = (y_pred == y_val).mean() * 100
        
        # Classification report
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)
        report = classification_report(
            y_val, y_pred, 
            target_names=class_names,
            digits=4
        )
        logger.info("\n" + report)
        
        # Confusion matrix
        logger.info("CONFUSION MATRIX")
        cm = confusion_matrix(y_val, y_pred)
        logger.info(f"\n{cm}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Validation Accuracy: {accuracy:.2f}%")
        logger.info(f"Best iteration: {model.best_iteration}")
        logger.info(f"Model saved to: {model_path}")
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
