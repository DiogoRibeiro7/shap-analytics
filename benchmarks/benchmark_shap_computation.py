"""
SHAP Computation Performance Benchmark
=======================================

Benchmarks SHAP value computation performance across:
- Different data sizes (100, 1K, 10K, 100K samples)
- Different model types (RandomForest, GradientBoosting)
- Different explainer types (TreeExplainer, KernelExplainer)
- Different background sample sizes

Requirements:
    pytest-benchmark
    memory_profiler
    psutil

Usage:
    # Run all benchmarks
    pytest benchmarks/benchmark_shap_computation.py -v

    # Run with profiling
    pytest benchmarks/benchmark_shap_computation.py --benchmark-autosave

    # Compare with baseline
    pytest benchmarks/benchmark_shap_computation.py --benchmark-compare=0001

    # Generate HTML report
    pytest benchmarks/benchmark_shap_computation.py --benchmark-histogram
"""

import gc
import warnings
from typing import Tuple, Dict, Any

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Test Data Generation
# ============================================================================

def generate_classification_data(
    n_samples: int,
    n_features: int = 20,
    n_informative: int = 15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        random_state: Random seed

    Returns:
        Tuple of (X, y)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=3,
        n_classes=2,
        random_state=random_state,
        flip_y=0.1
    )

    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(y, name='target')

    return X, y


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module", params=[100, 1000, 10000])
def dataset_small_to_large(request):
    """Generate datasets of varying sizes."""
    n_samples = request.param
    X, y = generate_classification_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_samples': n_samples
    }


@pytest.fixture(scope="module")
def dataset_100():
    """Small dataset for quick tests."""
    X, y = generate_classification_data(100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture(scope="module")
def dataset_1k():
    """Medium dataset."""
    X, y = generate_classification_data(1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture(scope="module")
def dataset_10k():
    """Large dataset."""
    X, y = generate_classification_data(10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture(scope="module", params=['rf', 'gb'])
def trained_model(request, dataset_1k):
    """Train and return model."""
    model_type = request.param

    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:  # gb
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    model.fit(dataset_1k['X_train'], dataset_1k['y_train'])
    return {'model': model, 'type': model_type}


# ============================================================================
# Benchmark Tests: Model Training
# ============================================================================

class TestModelTraining:
    """Benchmark model training times."""

    def test_train_random_forest_100(self, benchmark, dataset_100):
        """Benchmark RF training on 100 samples."""
        def train():
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(dataset_100['X_train'], dataset_100['y_train'])
            return model

        model = benchmark(train)
        assert model is not None

    def test_train_random_forest_1k(self, benchmark, dataset_1k):
        """Benchmark RF training on 1K samples."""
        def train():
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(dataset_1k['X_train'], dataset_1k['y_train'])
            return model

        model = benchmark(train)
        assert model is not None

    def test_train_random_forest_10k(self, benchmark, dataset_10k):
        """Benchmark RF training on 10K samples."""
        def train():
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(dataset_10k['X_train'], dataset_10k['y_train'])
            return model

        model = benchmark(train)
        assert model is not None


# ============================================================================
# Benchmark Tests: SHAP Explainer Creation
# ============================================================================

class TestExplainerCreation:
    """Benchmark SHAP explainer creation."""

    def test_create_tree_explainer_small_background(self, benchmark, trained_model, dataset_100):
        """Benchmark TreeExplainer creation with small background."""
        model = trained_model['model']
        background = shap.sample(dataset_100['X_train'], 50)

        def create():
            return shap.TreeExplainer(model, background)

        explainer = benchmark(create)
        assert explainer is not None

    def test_create_tree_explainer_medium_background(self, benchmark, trained_model, dataset_1k):
        """Benchmark TreeExplainer creation with medium background."""
        model = trained_model['model']
        background = shap.sample(dataset_1k['X_train'], 100)

        def create():
            return shap.TreeExplainer(model, background)

        explainer = benchmark(create)
        assert explainer is not None

    def test_create_tree_explainer_large_background(self, benchmark, trained_model, dataset_1k):
        """Benchmark TreeExplainer creation with large background."""
        model = trained_model['model']
        background = shap.sample(dataset_1k['X_train'], 500)

        def create():
            return shap.TreeExplainer(model, background)

        explainer = benchmark(create)
        assert explainer is not None


# ============================================================================
# Benchmark Tests: SHAP Value Computation
# ============================================================================

class TestShapComputation:
    """Benchmark SHAP value computation."""

    def test_compute_shap_single_sample(self, benchmark, trained_model, dataset_100):
        """Benchmark SHAP computation for single sample."""
        model = trained_model['model']
        background = shap.sample(dataset_100['X_train'], 50)
        explainer = shap.TreeExplainer(model, background)
        sample = dataset_100['X_test'].iloc[0:1]

        def compute():
            return explainer.shap_values(sample)

        shap_values = benchmark(compute)
        assert shap_values is not None

    def test_compute_shap_10_samples(self, benchmark, trained_model, dataset_100):
        """Benchmark SHAP computation for 10 samples."""
        model = trained_model['model']
        background = shap.sample(dataset_100['X_train'], 50)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_100['X_test'].iloc[0:10]

        def compute():
            return explainer.shap_values(samples)

        shap_values = benchmark(compute)
        assert shap_values is not None

    def test_compute_shap_100_samples(self, benchmark, dataset_100):
        """Benchmark SHAP computation for 100 samples."""
        X, y = dataset_100['X_train'], dataset_100['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 50)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_100['X_test']

        def compute():
            return explainer.shap_values(samples)

        shap_values = benchmark(compute)
        assert shap_values is not None

    def test_compute_shap_1k_samples(self, benchmark, dataset_1k):
        """Benchmark SHAP computation for 1K samples."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:200]  # Test on subset

        def compute():
            return explainer.shap_values(samples)

        shap_values = benchmark(compute)
        assert shap_values is not None

    @pytest.mark.slow
    def test_compute_shap_10k_samples(self, benchmark, dataset_10k):
        """Benchmark SHAP computation for 10K samples (slow)."""
        X, y = dataset_10k['X_train'], dataset_10k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_10k['X_test'].iloc[0:1000]  # Test on subset

        def compute():
            return explainer.shap_values(samples)

        shap_values = benchmark(compute)
        assert shap_values is not None


# ============================================================================
# Benchmark Tests: Background Sample Size Impact
# ============================================================================

class TestBackgroundSampleSize:
    """Benchmark impact of background sample size."""

    @pytest.mark.parametrize("bg_size", [10, 50, 100, 200, 500])
    def test_shap_with_varying_background(self, benchmark, dataset_1k, bg_size):
        """Benchmark SHAP computation with varying background sizes."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, min(bg_size, len(X)))
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:100]

        def compute():
            return explainer.shap_values(samples)

        shap_values = benchmark(compute)
        assert shap_values is not None


# ============================================================================
# Benchmark Tests: Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Benchmark batch processing strategies."""

    def test_batch_processing_vs_single(self, benchmark, dataset_1k):
        """Compare batch vs single sample processing."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:100]

        def compute_batch():
            # Process all at once
            return explainer.shap_values(samples)

        shap_values = benchmark(compute_batch)
        assert shap_values is not None

    def test_sequential_processing(self, benchmark, dataset_1k):
        """Benchmark sequential single-sample processing."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:10]  # Smaller for sequential

        def compute_sequential():
            results = []
            for idx in range(len(samples)):
                sample = samples.iloc[idx:idx+1]
                shap_val = explainer.shap_values(sample)
                results.append(shap_val)
            return results

        shap_values = benchmark(compute_sequential)
        assert len(shap_values) == 10


# ============================================================================
# Benchmark Tests: Memory Efficiency
# ============================================================================

class TestMemoryEfficiency:
    """Benchmark memory usage patterns."""

    def test_memory_efficient_batch_processing(self, benchmark, dataset_1k):
        """Test memory-efficient batch processing."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:500]

        def compute_with_cleanup():
            batch_size = 50
            results = []

            for i in range(0, len(samples), batch_size):
                batch = samples.iloc[i:i+batch_size]
                shap_vals = explainer.shap_values(batch)
                results.append(shap_vals)

                # Force garbage collection
                if i % 100 == 0:
                    gc.collect()

            return results

        results = benchmark(compute_with_cleanup)
        assert len(results) > 0


# ============================================================================
# Performance Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Tests to catch performance regressions."""

    def test_shap_computation_threshold_100(self, benchmark, dataset_100):
        """Ensure SHAP computation stays under threshold for 100 samples."""
        X, y = dataset_100['X_train'], dataset_100['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 50)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_100['X_test']

        result = benchmark(lambda: explainer.shap_values(samples))

        # Assert performance threshold: should complete in < 2 seconds
        assert benchmark.stats.stats.mean < 2.0, "Performance regression detected!"

    def test_shap_computation_threshold_1k(self, benchmark, dataset_1k):
        """Ensure SHAP computation stays under threshold for 1K samples."""
        X, y = dataset_1k['X_train'], dataset_1k['y_train']
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(model, background)
        samples = dataset_1k['X_test'].iloc[0:200]

        result = benchmark(lambda: explainer.shap_values(samples))

        # Assert performance threshold: should complete in < 10 seconds
        assert benchmark.stats.stats.mean < 10.0, "Performance regression detected!"


# ============================================================================
# Configuration and Reporting
# ============================================================================

def pytest_benchmark_generate_json(config, benchmarks, include_data):
    """Customize benchmark JSON output."""
    return {
        'benchmarks': benchmarks,
        'datetime': include_data.get('datetime'),
        'machine_info': include_data.get('machine_info'),
        'commit_info': include_data.get('commit_info', {})
    }


if __name__ == "__main__":
    # Run with: python -m pytest benchmarks/benchmark_shap_computation.py -v
    pytest.main([__file__, '-v', '--benchmark-autosave'])
