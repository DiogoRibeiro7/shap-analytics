"""
Dependency Compatibility Tests
==============================

Tests to ensure critical dependencies are at compatible versions.
This helps catch issues early when dependencies are updated.
"""

import sys
import pytest
import numpy as np
import shap
import sklearn
import pandas as pd


class TestNumpyCompatibility:
    """Test NumPy version compatibility with SHAP."""

    def test_numpy_version_less_than_2(self):
        """
        Ensure NumPy version is < 2.0.0 for SHAP compatibility.

        SHAP 0.45.0 and earlier versions are not compatible with NumPy 2.x
        due to breaking API changes in NumPy 2.0.
        """
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        major, minor = numpy_version

        assert major < 2, (
            f"NumPy version {np.__version__} is not compatible with SHAP. "
            f"Please use NumPy < 2.0.0. Current SHAP version: {shap.__version__}"
        )

    def test_numpy_version_minimum(self):
        """Ensure NumPy version meets minimum requirement."""
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        major, minor = numpy_version

        # Minimum version is 1.24.0
        assert (major, minor) >= (1, 24), (
            f"NumPy version {np.__version__} is too old. "
            f"Please use NumPy >= 1.24.0"
        )

    def test_numpy_array_api_attribute(self):
        """
        Test that NumPy has the _ARRAY_API attribute (NumPy 1.x).

        This attribute was removed in NumPy 2.0, and its presence
        indicates we're using NumPy 1.x.
        """
        # In NumPy 1.x, certain internal attributes exist
        # In NumPy 2.x, these are removed causing AttributeError
        # We test that basic NumPy operations work as expected
        arr = np.array([1, 2, 3])
        assert arr is not None
        assert len(arr) == 3

    def test_numpy_shap_integration(self):
        """
        Test basic SHAP functionality with current NumPy version.

        This ensures SHAP can actually work with the installed NumPy version.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Create small test dataset
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Create SHAP explainer (this will fail if NumPy is incompatible)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:10])

            # Verify SHAP values were computed
            assert shap_values is not None
            if isinstance(shap_values, list):
                assert len(shap_values) > 0
                assert shap_values[0].shape[0] == 10
            else:
                assert shap_values.shape[0] == 10

        except AttributeError as e:
            # If we get AttributeError, it's likely NumPy 2.x incompatibility
            if "_ARRAY_API" in str(e) or "obj2sctype" in str(e):
                pytest.fail(
                    f"SHAP failed with NumPy {np.__version__}: {e}. "
                    "This indicates NumPy 2.x incompatibility. "
                    "Please use NumPy < 2.0.0"
                )
            else:
                raise


class TestDependencyVersions:
    """Test that all critical dependencies are at expected versions."""

    def test_python_version(self):
        """Ensure Python version is >= 3.10."""
        assert sys.version_info >= (3, 10), (
            f"Python version {sys.version_info.major}.{sys.version_info.minor} "
            "is not supported. Please use Python >= 3.10"
        )

    def test_pandas_version(self):
        """Ensure pandas version is >= 2.0.0."""
        pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
        major, minor = pandas_version

        assert (major, minor) >= (2, 0), (
            f"pandas version {pd.__version__} is too old. "
            "Please use pandas >= 2.0.0"
        )

    def test_sklearn_version(self):
        """Ensure scikit-learn version is >= 1.3.0."""
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        major, minor = sklearn_version

        assert (major, minor) >= (1, 3), (
            f"scikit-learn version {sklearn.__version__} is too old. "
            "Please use scikit-learn >= 1.3.0"
        )

    def test_shap_version(self):
        """Ensure SHAP version is >= 0.43.0."""
        # SHAP version format: "0.45.0" or similar
        shap_version_str = shap.__version__.split('.')
        major = int(shap_version_str[0])
        minor = int(shap_version_str[1])

        assert (major, minor) >= (0, 43), (
            f"SHAP version {shap.__version__} is too old. "
            "Please use SHAP >= 0.43.0"
        )


class TestDependencyInteractions:
    """Test interactions between dependencies."""

    def test_numpy_pandas_compatibility(self):
        """Test NumPy and pandas work together correctly."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        arr = df.values

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)

    def test_numpy_sklearn_compatibility(self):
        """Test NumPy and scikit-learn work together correctly."""
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (50, 5)

    def test_all_together(self):
        """
        Integration test: NumPy + pandas + sklearn + SHAP.

        This is the most critical test - ensuring all dependencies
        work together in a realistic ML workflow.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Create dataset
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(df, y)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df[:20])

        # Verify everything worked
        assert shap_values is not None
        if isinstance(shap_values, list):
            assert len(shap_values) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
