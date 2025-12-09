# Implementation Plan

- [x] 1. Set up XGBoost configuration and dependencies
  - [x] 1.1 Add XGBoost configuration to config.py
    - Add XGBOOST_STRATEGY_CONFIG with hyperparameters, paths, and training settings
    - _Requirements: 5.1_
  - [x] 1.2 Update requirements.txt with xgboost dependency
    - Add xgboost and hypothesis (for property testing)
    - _Requirements: 5.1_

- [x] 2. Implement XGBoost feature adapter
  - [x] 2.1 Create XGBoostFeatureAdapter class in strategies/xgboost_features.py
    - Implement extract_features() to convert OHLCV to flat feature matrix
    - Implement prepare_training_data() with label generation
    - Implement normalize_features(), save_scaler(), load_scaler()
    - Reuse LSTMFeatureExtractor for feature calculation
    - _Requirements: 1.1, 3.1_
  - [ ]* 2.2 Write property test for feature extraction consistency
    - **Property 1: Feature Extraction Consistency**
    - **Validates: Requirements 1.1**

- [-] 3. Implement XGBoost trainer with IC calculation
  - [x] 3.1 Create XGBoostTrainer class in strategies/xgboost_trainer.py
    - Implement train() with early stopping
    - Implement calculate_ic() using Spearman correlation
    - Implement evaluate() returning metrics dict (accuracy, precision, recall, f1, confusion_matrix)
    - Implement save_model() and load_model()
    - _Requirements: 1.2, 1.4, 1.5, 4.1, 4.2, 4.3, 5.2_
  - [ ]* 3.2 Write property test for IC calculation correctness
    - **Property 2: IC Calculation Correctness**
    - **Validates: Requirements 1.2, 4.3**
  - [ ]* 3.3 Write property test for model serialization round-trip
    - **Property 3: Model Serialization Round-Trip**
    - **Validates: Requirements 1.5**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement XGBoost strategy class
  - [x] 5.1 Create XGBoostStrategy class in strategies/xgboost_strategy.py
    - Inherit from BaseStrategy
    - Implement __init__() with model loading from strategies/models/{symbol}/
    - Implement generate_signal() with confidence threshold
    - Implement get_prediction_proba() returning [p_down, p_hold, p_up]
    - Implement monitor_position() for live trading
    - Log prediction probabilities and resulting signal
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.2, 3.4_
  - [x] 5.2 Write property test for signal generation logic
    - **Property 4: Signal Generation from Predictions**
    - **Validates: Requirements 2.2, 2.3**
  - [x] 5.3 Write property test for probability distribution validity
    - **Property 5: Probability Distribution Validity**
    - **Validates: Requirements 3.2**

- [x] 6. Implement training script
  - [x] 6.1 Create train_xgboost.py training script
    - Implement CLI argument parsing (symbol, start_date, end_date, interval, output_dir)
    - Implement auto-generated output directory path as strategies/models/{symbol}/
    - Implement training pipeline with IC validation and warning if IC <= 0.05
    - Output classification report and confusion matrix
    - Save model as xgboost_model.json and scaler as xgboost_scaler.npz
    - Output feature importance rankings
    - _Requirements: 6.1, 6.2, 6.3, 1.3, 4.1, 4.2, 5.3_
  - [ ]* 6.2 Write property test for output directory path generation
    - **Property 8: Output Directory Path Generation**
    - **Validates: Requirements 6.2**

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement model ensemble support
  - [ ] 8.1 Create ModelEnsemble class in strategies/model_ensemble.py
    - Implement weighted combination of XGBoost and LSTM predictions
    - Support configurable weights (weights must sum to 1)
    - Output combined probability distribution
    - _Requirements: 3.3_
  - [ ]* 8.2 Write property test for ensemble weighting correctness
    - **Property 6: Ensemble Weighting Correctness**
    - **Validates: Requirements 3.3**

- [ ] 9. Implement time-series cross-validation
  - [ ] 9.1 Add time-series cross-validation to XGBoostTrainer
    - Implement walk-forward validation
    - Ensure all training samples have timestamps strictly earlier than validation samples
    - _Requirements: 4.4_
  - [ ]* 9.2 Write property test for time-series cross-validation ordering
    - **Property 7: Time-Series Cross-Validation Ordering**
    - **Validates: Requirements 4.4**

- [x] 10. Integration and registration
  - [x] 10.1 Register XGBoost strategy in strategies/__init__.py
    - Add XGBoostStrategy to exports
    - _Requirements: 3.4_
  - [x] 10.2 Update trading_manager.py to support xgboost strategy type
    - Add 'xgboost' to strategy type selection
    - Import and instantiate XGBoostStrategy when strategy_type == 'xgboost'
    - _Requirements: 3.4_

- [ ] 11. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
