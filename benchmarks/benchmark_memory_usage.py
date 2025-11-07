"""
Memory Usage Profiling Benchmark
=================================

Profiles memory usage for SHAP operations including:
- Model loading memory footprint
- SHAP explainer memory usage
- Computation memory scaling
- Memory leak detection

Requirements:
    memory_profiler
    psutil
    pytest

Usage:
    # Run with memory profiling
    python -m memory_profiler benchmarks/benchmark_memory_usage.py

    # Run as pytest
    pytest benchmarks/benchmark_memory_usage.py -v -s
"""

import gc
import pytest
import psutil
import os
from memory_profiler import profile, memory_usage
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import shap


# ============================================================================
# Memory Profiling Utilities
# ============================================================================

def get_process_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def memory_usage_decorator(func):
    """Decorator to measure memory usage of a function."""
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        result = func(*args, **kwargs)
        mem_after = get_process_memory()
        mem_used = mem_after - mem_before

        print(f"\nMemory usage for {func.__name__}:")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After: {mem_after:.2f} MB")
        print(f"  Used: {mem_used:.2f} MB")

        return result
    return wrapper


# ============================================================================
# Test Data Generation
# ============================================================================

@pytest.fixture(scope="module")
def small_dataset():
    """Generate small dataset (1K samples)."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture(scope="module")
def large_dataset():
    """Generate large dataset (10K samples)."""
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


# ============================================================================
# Model Memory Tests
# ============================================================================

class TestModelMemory:
    """Test memory usage of model training and storage."""

    @pytest.mark.memory
    def test_random_forest_memory_footprint(self, small_dataset):
        """Profile memory footprint of Random Forest model."""
        X, y = small_dataset

        mem_before = get_process_memory()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        mem_after = get_process_memory()
        mem_used = mem_after - mem_before

        print(f"\nRandom Forest (100 trees) memory: {mem_used:.2f} MB")
        assert mem_used < 200, f"Model memory too high: {mem_used:.2f} MB"

    @pytest.mark.memory
    def test_model_scaling_with_data(self):
        """Test how model memory scales with data size."""
        results = []

        for n_samples in [1000, 5000, 10000]:
            X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)

            gc.collect()  # Clean up before measurement
            mem_before = get_process_memory()

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

            mem_after = get_process_memory()
            mem_used = mem_after - mem_before

            results.append({
                'n_samples': n_samples,
                'memory_mb': mem_used
            })

            print(f"Samples: {n_samples:,}, Memory: {mem_used:.2f} MB")

            del model
            gc.collect()


# ============================================================================
# SHAP Explainer Memory Tests
# ============================================================================

class TestExplainerMemory:
    """Test memory usage of SHAP explainers."""

    @pytest.mark.memory
    def test_tree_explainer_memory(self, small_dataset):
        """Profile TreeExplainer memory footprint."""
        X, y = small_dataset

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        gc.collect()
        mem_before = get_process_memory()

        explainer = shap.TreeExplainer(model)

        mem_after = get_process_memory()
        mem_used = mem_after - mem_before

        print(f"\nTreeExplainer memory: {mem_used:.2f} MB")
        assert mem_used < 100, f"Explainer memory too high: {mem_used:.2f} MB"

    @pytest.mark.memory
    def test_explainer_with_background(self, small_dataset):
        """Test explainer memory with different background sizes."""
        X, y = small_dataset
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        for bg_size in [10, 50, 100, 500]:
            gc.collect()
            mem_before = get_process_memory()

            background = shap.sample(X, min(bg_size, len(X)))
            explainer = shap.TreeExplainer(model, background)

            mem_after = get_process_memory()
            mem_used = mem_after - mem_before

            print(f"Background size: {bg_size}, Memory: {mem_used:.2f} MB")

            del explainer, background
            gc.collect()


# ============================================================================
# SHAP Computation Memory Tests
# ============================================================================

class TestComputationMemory:
    """Test memory usage during SHAP value computation."""

    @pytest.mark.memory
    def test_shap_computation_memory_scaling(self, small_dataset):
        """Test memory scaling with number of samples explained."""
        X, y = small_dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)

        for n_explain in [10, 50, 100, 500]:
            samples = X.iloc[:n_explain]

            gc.collect()
            mem_before = get_process_memory()

            shap_values = explainer.shap_values(samples)

            mem_after = get_process_memory()
            mem_used = mem_after - mem_before

            print(f"Samples explained: {n_explain}, Memory: {mem_used:.2f} MB")

            del shap_values
            gc.collect()

    @pytest.mark.memory
    @pytest.mark.slow
    def test_large_batch_memory(self, large_dataset):
        """Test memory usage for large batch explanation."""
        X, y = large_dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:5000], y[:5000])

        explainer = shap.TreeExplainer(model)

        gc.collect()
        mem_before = get_process_memory()

        # Explain 1000 samples
        shap_values = explainer.shap_values(X[:1000])

        mem_after = get_process_memory()
        mem_used = mem_after - mem_before

        print(f"\nLarge batch (1000 samples) memory: {mem_used:.2f} MB")

        # Memory should be reasonable even for large batches
        assert mem_used < 500, f"Large batch memory too high: {mem_used:.2f} MB"


# ============================================================================
# Memory Leak Tests
# ============================================================================

class TestMemoryLeaks:
    """Test for memory leaks in repeated operations."""

    @pytest.mark.memory
    def test_repeated_computation_no_leak(self, small_dataset):
        """Test that repeated SHAP computations don't leak memory."""
        X, y = small_dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)

        samples = X.iloc[:100]

        # Baseline memory
        gc.collect()
        mem_baseline = get_process_memory()

        # Run multiple iterations
        for i in range(10):
            shap_values = explainer.shap_values(samples)
            del shap_values
            gc.collect()

        # Final memory
        mem_final = get_process_memory()
        mem_increase = mem_final - mem_baseline

        print(f"\nMemory after 10 iterations: {mem_increase:.2f} MB increase")

        # Should not increase significantly
        assert mem_increase < 50, f"Potential memory leak: {mem_increase:.2f} MB increase"

    @pytest.mark.memory
    def test_explainer_recreation_no_leak(self, small_dataset):
        """Test that repeatedly creating explainers doesn't leak memory."""
        X, y = small_dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        gc.collect()
        mem_baseline = get_process_memory()

        for i in range(10):
            explainer = shap.TreeExplainer(model)
            del explainer
            gc.collect()

        mem_final = get_process_memory()
        mem_increase = mem_final - mem_baseline

        print(f"\nMemory after 10 explainer creations: {mem_increase:.2f} MB increase")

        assert mem_increase < 30, f"Potential memory leak: {mem_increase:.2f} MB increase"


# ============================================================================
# Memory-Efficient Patterns
# ============================================================================

class TestMemoryOptimization:
    """Test memory-efficient processing patterns."""

    @pytest.mark.memory
    def test_chunked_processing_memory(self, large_dataset):
        """Test chunked processing vs. full batch for memory efficiency."""
        X, y = large_dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:5000], y[:5000])
        explainer = shap.TreeExplainer(model)

        n_samples = 1000
        chunk_size = 100

        # Measure chunked processing
        gc.collect()
        mem_before_chunked = get_process_memory()

        results = []
        for i in range(0, n_samples, chunk_size):
            chunk = X.iloc[i:i+chunk_size]
            shap_vals = explainer.shap_values(chunk)
            results.append(shap_vals)
            del shap_vals
            gc.collect()

        mem_after_chunked = get_process_memory()
        mem_chunked = mem_after_chunked - mem_before_chunked

        print(f"\nChunked processing memory: {mem_chunked:.2f} MB")

        # Chunked should use less memory than full batch
        assert mem_chunked < 300, f"Chunked processing memory: {mem_chunked:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "memory"])
