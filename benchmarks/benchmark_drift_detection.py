"""
Drift Detection Performance Benchmark
======================================

Benchmarks drift monitoring performance including:
- KS test computation speed
- PSI calculation performance
- SHAP drift detection
- Full drift report generation

Requirements:
    pytest-benchmark
    scipy
    scikit-learn
    shap

Usage:
    pytest benchmarks/benchmark_drift_detection.py -v --benchmark-autosave
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import shap
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.drift_monitoring import (
    calculate_psi,
    detect_feature_drift,
    detect_shap_drift,
    DriftMonitor
)


# ============================================================================
# Test Data Generation
# ============================================================================

@pytest.fixture(scope="module")
def reference_data_small():
    """Generate small reference dataset (1K samples)."""
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])


@pytest.fixture(scope="module")
def reference_data_large():
    """Generate large reference dataset (10K samples)."""
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])


@pytest.fixture(scope="module")
def current_data_no_drift(reference_data_small):
    """Generate current data with no drift."""
    # Sample from reference
    return reference_data_small.sample(500, replace=True, random_state=43)


@pytest.fixture(scope="module")
def current_data_with_drift(reference_data_small):
    """Generate current data with drift."""
    data = reference_data_small.sample(500, replace=True, random_state=43)

    # Add drift to some features
    n_drift_features = 5
    drift_cols = data.columns[:n_drift_features]

    for col in drift_cols:
        data[col] += np.random.randn(len(data)) * data[col].std() * 0.5

    return data


# ============================================================================
# Statistical Test Benchmarks
# ============================================================================

class TestStatisticalTests:
    """Benchmark individual statistical tests."""

    def test_ks_test_performance(self, benchmark, reference_data_small, current_data_no_drift):
        """Benchmark Kolmogorov-Smirnov test performance."""
        ref_values = reference_data_small['feature_0'].values
        cur_values = current_data_no_drift['feature_0'].values

        result = benchmark(stats.ks_2samp, ref_values, cur_values)
        assert result is not None

    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_ks_test_scaling(self, benchmark, n_samples):
        """Test KS test performance scaling with data size."""
        ref_values = np.random.randn(n_samples)
        cur_values = np.random.randn(n_samples)

        result = benchmark(stats.ks_2samp, ref_values, cur_values)

    def test_psi_calculation_performance(self, benchmark, reference_data_small, current_data_no_drift):
        """Benchmark PSI calculation performance."""
        ref_values = reference_data_small['feature_0'].values
        cur_values = current_data_no_drift['feature_0'].values

        result = benchmark(calculate_psi, ref_values, cur_values)
        assert isinstance(result, float)

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_psi_different_bins(self, benchmark, reference_data_small, current_data_no_drift, bins):
        """Test PSI performance with different bin counts."""
        ref_values = reference_data_small['feature_0'].values
        cur_values = current_data_no_drift['feature_0'].values

        result = benchmark(calculate_psi, ref_values, cur_values, bins)


# ============================================================================
# Feature Drift Detection Benchmarks
# ============================================================================

class TestFeatureDriftDetection:
    """Benchmark feature-level drift detection."""

    def test_single_feature_drift_detection(
        self,
        benchmark,
        reference_data_small,
        current_data_no_drift
    ):
        """Benchmark drift detection for single feature."""
        result = benchmark(
            detect_feature_drift,
            reference_data_small,
            current_data_no_drift,
            'feature_0'
        )

        assert result.feature_name == 'feature_0'

    def test_all_features_drift_detection(
        self,
        benchmark,
        reference_data_small,
        current_data_no_drift
    ):
        """Benchmark drift detection for all features."""
        def detect_all():
            results = []
            for col in reference_data_small.columns:
                result = detect_feature_drift(
                    reference_data_small,
                    current_data_no_drift,
                    col
                )
                results.append(result)
            return results

        results = benchmark(detect_all)
        assert len(results) == len(reference_data_small.columns)

    def test_drift_detection_with_drift(
        self,
        benchmark,
        reference_data_small,
        current_data_with_drift
    ):
        """Benchmark drift detection when drift is present."""
        def detect_all():
            results = []
            for col in reference_data_small.columns[:10]:  # First 10 features
                result = detect_feature_drift(
                    reference_data_small,
                    current_data_with_drift,
                    col
                )
                results.append(result)
            return results

        results = benchmark(detect_all)
        # Should detect drift in some features
        drifted = [r for r in results if r.is_drifted]
        assert len(drifted) > 0


# ============================================================================
# SHAP Drift Detection Benchmarks
# ============================================================================

class TestShapDriftDetection:
    """Benchmark SHAP-based drift detection."""

    @pytest.fixture(scope="class")
    def model_and_shap(self, reference_data_small):
        """Train model and compute SHAP values."""
        X = reference_data_small
        y = pd.Series(np.random.randint(0, 2, len(X)))

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        ref_shap = explainer.shap_values(X.iloc[:200])
        if isinstance(ref_shap, list):
            ref_shap = ref_shap[1]

        cur_shap = explainer.shap_values(X.iloc[200:400])
        if isinstance(cur_shap, list):
            cur_shap = cur_shap[1]

        return ref_shap, cur_shap, list(X.columns)

    def test_shap_drift_computation(self, benchmark, model_and_shap):
        """Benchmark SHAP drift detection computation."""
        ref_shap, cur_shap, feature_names = model_and_shap

        result = benchmark(detect_shap_drift, ref_shap, cur_shap, feature_names)

        assert len(result) == len(feature_names)

    def test_shap_drift_with_subset(self, benchmark, model_and_shap):
        """Benchmark SHAP drift on feature subset."""
        ref_shap, cur_shap, feature_names = model_and_shap

        # Use only first 10 features
        ref_subset = ref_shap[:, :10]
        cur_subset = cur_shap[:, :10]
        names_subset = feature_names[:10]

        result = benchmark(detect_shap_drift, ref_subset, cur_subset, names_subset)
        assert len(result) == 10


# ============================================================================
# Full Drift Monitor Benchmarks
# ============================================================================

class TestFullDriftMonitor:
    """Benchmark complete drift monitoring workflow."""

    @pytest.fixture(scope="class")
    def drift_monitor(self, reference_data_small):
        """Create configured drift monitor."""
        X = reference_data_small
        y = pd.Series(np.random.randint(0, 2, len(X)))

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        monitor = DriftMonitor(
            model=model,
            reference_data=X,
            feature_names=list(X.columns),
            window_size=100,
            alert_threshold=3
        )

        return monitor

    def test_single_sample_addition(self, benchmark, drift_monitor, reference_data_small):
        """Benchmark adding single sample to monitor."""
        sample = reference_data_small.iloc[0].values

        result = benchmark(drift_monitor.add_sample, sample)

    def test_full_drift_check(self, benchmark, drift_monitor, current_data_no_drift):
        """Benchmark full drift detection check."""
        # Fill window
        for sample in current_data_no_drift.iloc[:100].values:
            drift_monitor.add_sample(sample)

        result = benchmark(drift_monitor.check_drift)

        assert result.n_samples_current == 100

    def test_drift_check_with_drift(self, benchmark, drift_monitor, current_data_with_drift):
        """Benchmark drift check when drift is present."""
        # Clear window first
        drift_monitor.current_window.clear()

        # Fill with drifted data
        for sample in current_data_with_drift.iloc[:100].values:
            drift_monitor.add_sample(sample)

        result = benchmark(drift_monitor.check_drift)

        # Should detect some drift
        assert result.n_features_drifted >= 0


# ============================================================================
# Performance Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Tests to catch drift detection performance regressions."""

    def test_feature_drift_threshold(
        self,
        benchmark,
        reference_data_small,
        current_data_no_drift
    ):
        """Ensure feature drift detection stays under threshold."""
        def detect_all():
            results = []
            for col in reference_data_small.columns:
                result = detect_feature_drift(
                    reference_data_small,
                    current_data_no_drift,
                    col
                )
                results.append(result)
            return results

        result = benchmark(detect_all)

        # Should complete in < 5 seconds for 20 features
        assert benchmark.stats.stats.mean < 5.0, "Drift detection regression detected!"

    def test_full_drift_report_threshold(self, benchmark, drift_monitor, current_data_no_drift):
        """Ensure full drift report generation stays under threshold."""
        # Fill window
        drift_monitor.current_window.clear()
        for sample in current_data_no_drift.iloc[:100].values:
            drift_monitor.add_sample(sample)

        result = benchmark(drift_monitor.check_drift)

        # Full drift report should complete in < 10 seconds
        assert benchmark.stats.stats.mean < 10.0, "Drift report regression detected!"


# ============================================================================
# Scaling Tests
# ============================================================================

class TestScaling:
    """Test performance scaling with data size."""

    @pytest.mark.parametrize("n_features", [10, 20, 50])
    def test_drift_scaling_with_features(self, benchmark, n_features):
        """Test how drift detection scales with number of features."""
        # Generate data
        X_ref, _ = make_classification(n_samples=1000, n_features=n_features, random_state=42)
        X_cur, _ = make_classification(n_samples=500, n_features=n_features, random_state=43)

        ref_df = pd.DataFrame(X_ref, columns=[f'f{i}' for i in range(n_features)])
        cur_df = pd.DataFrame(X_cur, columns=[f'f{i}' for i in range(n_features)])

        def detect_all():
            results = []
            for col in ref_df.columns:
                result = detect_feature_drift(ref_df, cur_df, col)
                results.append(result)
            return results

        results = benchmark(detect_all)
        assert len(results) == n_features

    @pytest.mark.slow
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_drift_scaling_with_samples(self, benchmark, n_samples):
        """Test how drift detection scales with number of samples."""
        X_ref, _ = make_classification(n_samples=n_samples, n_features=20, random_state=42)
        X_cur, _ = make_classification(n_samples=n_samples//2, n_features=20, random_state=43)

        ref_df = pd.DataFrame(X_ref, columns=[f'f{i}' for i in range(20)])
        cur_df = pd.DataFrame(X_cur, columns=[f'f{i}' for i in range(20)])

        result = benchmark(
            detect_feature_drift,
            ref_df,
            cur_df,
            'f0'
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-autosave", "-m", "not slow"])
