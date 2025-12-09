"""
XGBoost Strategy Property-Based Tests

Property tests for validating correctness properties defined in the design document.
Uses hypothesis library for property-based testing.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume
import xgboost as xgb
import tempfile
import os


def generate_signal_from_probs(probs: np.ndarray, confidence_threshold: float) -> int:
    """
    Pure function that implements signal generation logic from XGBoostStrategy.
    
    This extracts the core signal generation logic for testability.
    
    Args:
        probs: Probability distribution [p_down, p_hold, p_up]
        confidence_threshold: Minimum confidence required for non-zero signal
        
    Returns:
        Signal: 1=buy, -1=sell, 0=hold
    """
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    # Generate signal
    # 0=down(sell), 1=hold(observe), 2=up(buy)
    signal = 0
    if confidence >= confidence_threshold:
        if pred_class == 2:  # up
            signal = 1  # buy
        elif pred_class == 0:  # down
            signal = -1  # sell
    
    return signal


# Strategy for generating valid probability distributions (sum to 1)
@st.composite
def probability_distribution(draw):
    """Generate a valid 3-class probability distribution that sums to 1."""
    # Generate 3 positive floats
    raw = draw(st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3, max_size=3
    ))
    # Normalize to sum to 1
    total = sum(raw)
    normalized = [x / total for x in raw]
    return np.array(normalized)


@st.composite
def confidence_threshold(draw):
    """Generate a valid confidence threshold between 0 and 1."""
    return draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False))


class TestSignalGenerationProperty:
    """
    **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
    **Validates: Requirements 2.2, 2.3**
    
    Property 4: Signal Generation from Predictions
    *For any* prediction probability distribution where the maximum probability exceeds 
    the confidence threshold, the Signal_Generator SHALL return a non-zero signal (1 or -1) 
    corresponding to the predicted class. *For any* prediction where the maximum probability 
    is at or below the threshold, the Signal_Generator SHALL return 0.
    """
    
    @given(probs=probability_distribution(), threshold=confidence_threshold())
    @settings(max_examples=100)
    def test_signal_generation_above_threshold_returns_nonzero_for_up_or_down(
        self, probs: np.ndarray, threshold: float
    ):
        """
        **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
        
        When max probability exceeds threshold AND predicted class is up(2) or down(0),
        signal should be non-zero (1 for up, -1 for down).
        """
        pred_class = np.argmax(probs)
        max_prob = probs[pred_class]
        
        # Only test cases where confidence exceeds threshold and class is not hold
        assume(max_prob >= threshold)
        assume(pred_class != 1)  # Not hold class
        
        signal = generate_signal_from_probs(probs, threshold)
        
        # Signal should be non-zero
        assert signal != 0, f"Expected non-zero signal for probs={probs}, threshold={threshold}"
        
        # Signal should match predicted class
        if pred_class == 2:  # up
            assert signal == 1, f"Expected buy signal (1) for up prediction, got {signal}"
        elif pred_class == 0:  # down
            assert signal == -1, f"Expected sell signal (-1) for down prediction, got {signal}"
    
    @given(probs=probability_distribution(), threshold=confidence_threshold())
    @settings(max_examples=100)
    def test_signal_generation_below_threshold_returns_zero(
        self, probs: np.ndarray, threshold: float
    ):
        """
        **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
        
        When max probability is below threshold, signal should be 0 (hold).
        """
        pred_class = np.argmax(probs)
        max_prob = probs[pred_class]
        
        # Only test cases where confidence is below threshold
        assume(max_prob < threshold)
        
        signal = generate_signal_from_probs(probs, threshold)
        
        assert signal == 0, f"Expected hold signal (0) for low confidence, got {signal}"
    
    @given(probs=probability_distribution(), threshold=confidence_threshold())
    @settings(max_examples=100)
    def test_signal_generation_hold_class_returns_zero(
        self, probs: np.ndarray, threshold: float
    ):
        """
        **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
        
        When predicted class is hold (1), signal should be 0 regardless of confidence.
        """
        pred_class = np.argmax(probs)
        
        # Only test cases where hold class has highest probability
        assume(pred_class == 1)
        
        signal = generate_signal_from_probs(probs, threshold)
        
        assert signal == 0, f"Expected hold signal (0) for hold prediction, got {signal}"
    
    @given(probs=probability_distribution(), threshold=confidence_threshold())
    @settings(max_examples=100)
    def test_signal_is_valid_value(self, probs: np.ndarray, threshold: float):
        """
        **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
        
        Signal should always be one of {-1, 0, 1}.
        """
        signal = generate_signal_from_probs(probs, threshold)
        
        assert signal in {-1, 0, 1}, f"Signal must be -1, 0, or 1, got {signal}"
    
    @given(probs=probability_distribution(), threshold=confidence_threshold())
    @settings(max_examples=100)
    def test_signal_consistency_with_prediction_class(
        self, probs: np.ndarray, threshold: float
    ):
        """
        **Feature: xgboost-strategy, Property 4: Signal Generation from Predictions**
        
        Comprehensive test: signal should be consistent with prediction class and confidence.
        - If confidence >= threshold and class is up(2): signal = 1
        - If confidence >= threshold and class is down(0): signal = -1
        - If confidence < threshold OR class is hold(1): signal = 0
        """
        pred_class = np.argmax(probs)
        max_prob = probs[pred_class]
        
        signal = generate_signal_from_probs(probs, threshold)
        
        if max_prob >= threshold and pred_class == 2:
            expected = 1
        elif max_prob >= threshold and pred_class == 0:
            expected = -1
        else:
            expected = 0
        
        assert signal == expected, (
            f"Signal mismatch: probs={probs}, threshold={threshold}, "
            f"pred_class={pred_class}, max_prob={max_prob}, "
            f"expected={expected}, got={signal}"
        )


# Strategy for generating valid feature matrices for XGBoost
@st.composite
def feature_matrix(draw):
    """Generate a valid feature matrix for XGBoost prediction.
    
    Generates a matrix with shape (n_samples, n_features) where:
    - n_samples: 1 to 5 samples
    - n_features: 18 features (matching XGBoostFeatureAdapter output)
    """
    n_samples = draw(st.integers(min_value=1, max_value=5))
    n_features = 18  # Standard feature count from XGBoostFeatureAdapter
    
    # Generate feature values in reasonable ranges
    features = []
    for _ in range(n_samples):
        row = draw(st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=n_features, max_size=n_features
        ))
        features.append(row)
    
    return np.array(features)


def create_mock_xgboost_model():
    """Create a simple trained XGBoost model for testing.
    
    Returns a model trained on synthetic data that outputs 3-class probabilities.
    """
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 100
    n_features = 18
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 3, n_samples)  # 3 classes: 0, 1, 2
    
    # Train a simple XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 3,
        'eta': 0.1,
        'seed': 42
    }
    
    model = xgb.train(params, dtrain, num_boost_round=10)
    return model


class TestProbabilityDistributionValidity:
    """
    **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
    **Validates: Requirements 3.2**
    
    Property 5: Probability Distribution Validity
    *For any* input features, the XGBoost model SHALL output a probability distribution 
    of exactly 3 elements (down, hold, up) that sum to 1.0 (within floating-point tolerance).
    """
    
    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Set up a mock XGBoost model for testing."""
        self.model = create_mock_xgboost_model()
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_has_three_elements(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        The output probability distribution must have exactly 3 elements.
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        # For each sample, check that we have exactly 3 probabilities
        for i in range(proba.shape[0]):
            assert proba[i].shape == (3,), (
                f"Expected 3 probabilities, got shape {proba[i].shape}"
            )
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_sums_to_one(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        The probability distribution must sum to 1.0 (within floating-point tolerance).
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        # For each sample, check that probabilities sum to 1
        for i in range(proba.shape[0]):
            prob_sum = np.sum(proba[i])
            assert np.isclose(prob_sum, 1.0, atol=1e-6), (
                f"Probabilities should sum to 1.0, got {prob_sum} for sample {i}"
            )
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_all_non_negative(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        All probability values must be non-negative.
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        # For each sample, check that all probabilities are non-negative
        for i in range(proba.shape[0]):
            assert np.all(proba[i] >= 0), (
                f"All probabilities should be non-negative, got {proba[i]} for sample {i}"
            )
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_all_at_most_one(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        All probability values must be at most 1.0.
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        # For each sample, check that all probabilities are at most 1
        for i in range(proba.shape[0]):
            assert np.all(proba[i] <= 1.0), (
                f"All probabilities should be at most 1.0, got {proba[i]} for sample {i}"
            )
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_no_nan_or_inf(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        Probability distribution must not contain NaN or Inf values.
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        # For each sample, check for NaN and Inf
        for i in range(proba.shape[0]):
            assert not np.any(np.isnan(proba[i])), (
                f"Probabilities should not contain NaN, got {proba[i]} for sample {i}"
            )
            assert not np.any(np.isinf(proba[i])), (
                f"Probabilities should not contain Inf, got {proba[i]} for sample {i}"
            )
    
    @given(features=feature_matrix())
    @settings(max_examples=100)
    def test_probability_distribution_complete_validity(self, features: np.ndarray):
        """
        **Feature: xgboost-strategy, Property 5: Probability Distribution Validity**
        
        Comprehensive test: probability distribution must be valid:
        - Exactly 3 elements
        - All values in [0, 1]
        - Sum to 1.0
        - No NaN or Inf
        """
        dmatrix = xgb.DMatrix(features)
        proba = self.model.predict(dmatrix)
        
        for i in range(proba.shape[0]):
            p = proba[i]
            
            # Check shape
            assert p.shape == (3,), f"Expected 3 probabilities, got shape {p.shape}"
            
            # Check no NaN or Inf
            assert not np.any(np.isnan(p)), f"Contains NaN: {p}"
            assert not np.any(np.isinf(p)), f"Contains Inf: {p}"
            
            # Check range [0, 1]
            assert np.all(p >= 0), f"Contains negative values: {p}"
            assert np.all(p <= 1.0), f"Contains values > 1: {p}"
            
            # Check sum to 1
            assert np.isclose(np.sum(p), 1.0, atol=1e-6), (
                f"Sum should be 1.0, got {np.sum(p)}"
            )
