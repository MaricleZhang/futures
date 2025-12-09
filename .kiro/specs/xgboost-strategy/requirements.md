# Requirements Document

## Introduction

本文档定义了XGBoost交易策略的需求规格。该策略旨在使用XGBoost梯度提升算法预测交易信号，要求达到IC（信息系数）> 0.05的预测能力，并设计为可与现有LSTM策略进行模型融合。XGBoost策略将复用现有的特征工程模块，并提供与LSTM策略兼容的接口，以便后续实现集成学习。

## Glossary

- **XGBoost_Strategy**: 基于XGBoost算法的交易信号预测策略模块
- **IC (Information Coefficient)**: 信息系数，衡量预测值与实际收益之间的相关性，计算为预测值与实际收益的Spearman秩相关系数
- **Feature_Extractor**: 特征提取器，复用现有LSTM特征工程模块提取技术指标特征
- **Signal_Generator**: 信号生成器，将模型预测转换为交易信号（买入/卖出/观望）
- **Model_Trainer**: 模型训练器，负责XGBoost模型的训练、验证和保存
- **Prediction_Confidence**: 预测置信度，模型对预测结果的确信程度
- **Model_Ensemble**: 模型融合，将XGBoost与LSTM模型的预测结果进行组合

## Requirements

### Requirement 1

**User Story:** As a quantitative trader, I want to train an XGBoost model on historical price data, so that I can generate trading signals with IC > 0.05.

#### Acceptance Criteria

1. WHEN the Model_Trainer receives historical OHLCV data THEN the XGBoost_Strategy SHALL extract features using the Feature_Extractor and prepare training samples
2. WHEN training completes THEN the Model_Trainer SHALL calculate and report the IC value on the validation set
3. WHEN the validation IC is less than or equal to 0.05 THEN the Model_Trainer SHALL log a warning and suggest hyperparameter adjustments
4. WHEN the validation IC exceeds 0.05 THEN the Model_Trainer SHALL save the trained model to the designated output directory
5. WHEN serializing the trained model THEN the Model_Trainer SHALL save both the model file and the feature scaler parameters

### Requirement 2

**User Story:** As a quantitative trader, I want the XGBoost strategy to generate trading signals during live trading, so that I can execute trades based on model predictions.

#### Acceptance Criteria

1. WHEN the Signal_Generator receives new K-line data THEN the XGBoost_Strategy SHALL extract features and generate a prediction
2. WHEN the prediction confidence exceeds the configured threshold THEN the Signal_Generator SHALL return a trading signal (1 for buy, -1 for sell)
3. WHEN the prediction confidence is below the threshold THEN the Signal_Generator SHALL return 0 (hold/observe)
4. WHEN generating signals THEN the XGBoost_Strategy SHALL log the prediction probabilities and the resulting signal

### Requirement 3

**User Story:** As a quantitative trader, I want the XGBoost strategy to be compatible with the existing LSTM strategy, so that I can combine their predictions for improved accuracy.

#### Acceptance Criteria

1. WHEN initializing the XGBoost_Strategy THEN the system SHALL use the same Feature_Extractor interface as the LSTM strategy
2. WHEN generating predictions THEN the XGBoost_Strategy SHALL output probability distributions in the same format as the LSTM strategy (3 classes: down, hold, up)
3. WHEN the Model_Ensemble combines predictions THEN the system SHALL support configurable weighting between XGBoost and LSTM outputs
4. WHEN loading models THEN the XGBoost_Strategy SHALL follow the same directory structure convention as the LSTM strategy (strategies/models/{symbol}/)

### Requirement 4

**User Story:** As a quantitative trader, I want to evaluate the XGBoost model performance, so that I can assess its predictive power and reliability.

#### Acceptance Criteria

1. WHEN training completes THEN the Model_Trainer SHALL report classification metrics including accuracy, precision, recall, and F1-score for each class
2. WHEN training completes THEN the Model_Trainer SHALL report the confusion matrix
3. WHEN training completes THEN the Model_Trainer SHALL calculate and report the IC value using Spearman rank correlation
4. WHEN evaluating the model THEN the Model_Trainer SHALL perform time-series cross-validation to avoid look-ahead bias

### Requirement 5

**User Story:** As a quantitative trader, I want to configure XGBoost hyperparameters, so that I can optimize model performance for different market conditions.

#### Acceptance Criteria

1. WHEN initializing the Model_Trainer THEN the system SHALL accept configurable hyperparameters including max_depth, learning_rate, n_estimators, and subsample
2. WHEN training the model THEN the XGBoost_Strategy SHALL use early stopping based on validation loss to prevent overfitting
3. WHEN the configuration specifies feature importance analysis THEN the Model_Trainer SHALL output feature importance rankings after training

### Requirement 6

**User Story:** As a developer, I want to run the XGBoost training from command line, so that I can easily train models for different symbols and time periods.

#### Acceptance Criteria

1. WHEN the training script is executed THEN the system SHALL accept command-line arguments for symbol, start_date, end_date, interval, and output_dir
2. WHEN the output_dir is not specified THEN the system SHALL auto-generate the path as strategies/models/{symbol}/
3. WHEN training completes THEN the system SHALL save the model as xgboost_model.json and the scaler as xgboost_scaler.npz in the output directory
