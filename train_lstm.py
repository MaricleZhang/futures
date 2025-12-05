"""
LSTM模型训练脚本
用于训练ENAUSDC交易信号预测模型

Usage:
    python train_lstm.py --symbol ENAUSDC --start_date 2025-09-01 --end_date 2025-12-04

File: train_lstm.py
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import DataLoader as HistoricalDataLoader
from strategies.lstm_features import LSTMFeatureExtractor
from strategies.lstm_model import LSTMClassifier, LSTMTrainer
from utils.logger import Logger
import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM model for trading signals')
    
    parser.add_argument('--symbol', type=str, default='ENAUSDC',
                       help='Trading symbol (default: ENAUSDC)')
    parser.add_argument('--start_date', type=str, default='2025-06-01',
                       help='Start date for training data')
    parser.add_argument('--end_date', type=str, default='2025-12-05',
                       help='End date for training data')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Kline interval (default: 15m)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='LSTM sequence length (default: 60)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size (default: 128)')
    parser.add_argument('--output_dir', type=str, default='strategies/models',
                       help='Output directory for models')
    
    return parser.parse_args()


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
    
    logger.info("=" * 80)
    logger.info("LSTM MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Symbol:          {symbol}")
    logger.info(f"Period:          {args.start_date} to {args.end_date}")
    logger.info(f"Interval:        {args.interval}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Hidden Size:     {args.hidden_size}")
    logger.info(f"Batch Size:      {args.batch_size}")
    logger.info(f"Learning Rate:   {args.learning_rate}")
    logger.info(f"Max Epochs:      {args.epochs}")
    logger.info(f"Patience:        {args.patience}")
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
        
        # Step 2: Calculate features
        logger.info("Step 2: Calculating features...")
        feature_extractor = LSTMFeatureExtractor(sequence_length=args.sequence_length)
        df = feature_extractor.calculate_features(df)
        
        if df is None:
            logger.error("Failed to calculate features")
            return 1
        
        # Step 3: Prepare sequences
        logger.info("Step 3: Preparing sequences...")
        dl_config = config.DL_STRATEGY_CONFIG.get('training', {})
        future_periods = dl_config.get('future_periods', 5)
        threshold = dl_config.get('threshold', 0.002)
        
        X, y = feature_extractor.prepare_sequences(
            df, 
            future_periods=future_periods,
            threshold=threshold
        )
        
        if X is None or y is None:
            logger.error("Failed to prepare sequences")
            return 1
        
        logger.info(f"Prepared {len(X)} samples")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_names = ['跌', '持', '涨']
        logger.info("Class distribution:")
        for u, c in zip(unique, counts):
            logger.info(f"  {class_names[u]}: {c} ({c/len(y)*100:.1f}%)")
        
        # Step 4: Normalize features (fit global scaler)
        logger.info("Step 4: Normalizing features...")
        X = feature_extractor.normalize_features(X, fit=True)
        
        # Save scaler for inference
        scaler_path = Path(args.output_dir) / 'scaler.npz'
        feature_extractor.save_scaler(str(scaler_path))
        
        # Step 5: Split data
        logger.info("Step 5: Splitting train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        # Step 6: Initialize model
        logger.info("Step 6: Initializing model...")
        model = LSTMClassifier(
            input_size=feature_extractor.get_feature_count(),
            hidden_size=args.hidden_size,
            num_layers=2,
            num_classes=3,
            dropout=0.3
        )
        
        trainer = LSTMTrainer(model, learning_rate=args.learning_rate)
        
        # Step 7: Training loop
        logger.info("Step 7: Starting training...")
        logger.info("-" * 60)
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = output_dir / 'best_model.pth'
        
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                trainer.save_model(str(best_model_path))
                logger.info(f"  -> New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("-" * 60)
        logger.info(f"Best epoch: {best_epoch} | Best Val Accuracy: {best_val_acc:.2f}%")
        
        # Step 8: Final evaluation
        logger.info("Step 8: Final evaluation...")
        
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Predict on validation set
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(trainer.device)
                outputs = model(X_batch)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        # Classification report
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)
        report = classification_report(
            all_labels, all_preds, 
            target_names=class_names,
            digits=4
        )
        logger.info("\n" + report)
        
        # Confusion matrix
        logger.info("CONFUSION MATRIX")
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"\n{cm}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        logger.info(f"Target Accuracy:          65.00%")
        
        if best_val_acc >= 65.0:
            logger.info("✅ TARGET ACCURACY ACHIEVED!")
        else:
            logger.info(f"⚠️ Target not reached. Gap: {65.0 - best_val_acc:.2f}%")
            logger.info("Recommendations:")
            logger.info("  - Try different hyperparameters")
            logger.info("  - Increase training data")
            logger.info("  - Adjust prediction threshold")
        
        logger.info(f"\nModel saved to: {best_model_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
