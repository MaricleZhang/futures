"""
Train LSTM using cached local data (no network required)
Resamples 1-minute data to 15-minute candles

Usage:
    python train_lstm_cached.py
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.lstm_features import LSTMFeatureExtractor
from strategies.lstm_model import LSTMClassifier, LSTMTrainer
from utils.logger import Logger
import config


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to 15-minute candles"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled


def main():
    """Main training function"""
    # Initialize logger
    logger = Logger.get_logger()
    
    # Configuration
    data_path = Path('/Users/zhangjian/WorkSpace/futures/data/backtest/ENA_USDC_1m.csv')
    sequence_length = 60
    hidden_size = 128
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    patience = 15
    
    logger.info("=" * 80)
    logger.info("LSTM MODEL TRAINING (CACHED DATA)")
    logger.info("=" * 80)
    logger.info(f"Data Path:       {data_path}")
    logger.info(f"Sequence Length: {sequence_length}")
    logger.info(f"Hidden Size:     {hidden_size}")
    logger.info(f"Batch Size:      {batch_size}")
    logger.info(f"Learning Rate:   {learning_rate}")
    logger.info(f"Max Epochs:      {epochs}")
    logger.info(f"Patience:        {patience}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load cached data
        logger.info("Step 1: Loading cached 1m data...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} 1-minute candles")
        
        # Step 2: Resample to 15m
        logger.info("Step 2: Resampling to 15-minute candles...")
        df = resample_to_15m(df)
        logger.info(f"Resampled to {len(df)} 15-minute candles")
        
        # Step 3: Calculate features
        logger.info("Step 3: Calculating features...")
        feature_extractor = LSTMFeatureExtractor(sequence_length=sequence_length)
        df = feature_extractor.calculate_features(df)
        
        if df is None:
            logger.error("Failed to calculate features")
            return 1
        
        # Step 4: Prepare sequences
        logger.info("Step 4: Preparing sequences...")
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
        
        # Step 5: Normalize features
        logger.info("Step 5: Normalizing features...")
        X = feature_extractor.normalize_features(X)
        
        # Step 6: Split data
        logger.info("Step 6: Splitting train/validation sets...")
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
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Step 7: Initialize model
        logger.info("Step 7: Initializing model...")
        model = LSTMClassifier(
            input_size=feature_extractor.get_feature_count(),
            hidden_size=hidden_size,
            num_layers=2,
            num_classes=3,
            dropout=0.3
        )
        
        trainer = LSTMTrainer(model, learning_rate=learning_rate)
        
        # Step 8: Training loop
        logger.info("Step 8: Starting training...")
        logger.info("-" * 60)
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        output_dir = Path('strategies/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = output_dir / 'best_model.pth'
        
        for epoch in range(epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
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
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("-" * 60)
        logger.info(f"Best epoch: {best_epoch} | Best Val Accuracy: {best_val_acc:.2f}%")
        
        # Step 9: Final evaluation
        logger.info("Step 9: Final evaluation...")
        
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
