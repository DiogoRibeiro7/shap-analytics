"""
Basic SHAP Analytics Usage Example
===================================

This example demonstrates the fundamental usage of SHAP (SHapley Additive exPlanations)
for explaining machine learning model predictions using the breast cancer dataset.

Features:
- Data loading and preprocessing
- Model training with Random Forest
- SHAP value computation
- Basic visualization
- Feature importance analysis

Requirements:
- scikit-learn
- shap
- numpy
- pandas
- matplotlib

Usage:
    python examples/basic_usage.py
"""

import logging
import time
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# Suppress matplotlib backend warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Configure matplotlib backend for server environments
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("Matplotlib not available - visualizations will be skipped")

# Configure logging for production-ready output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load and prepare the breast cancer dataset.

    Returns:
        Tuple containing:
        - X: Feature DataFrame
        - y: Target Series
        - feature_names: List of feature names

    Raises:
        RuntimeError: If data loading fails
    """
    try:
        logger.info("Loading breast cancer dataset...")
        data = load_breast_cancer()

        # Convert to pandas for better handling
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y, list(data.feature_names)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise RuntimeError(f"Data loading error: {e}")


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility

    Returns:
        Trained RandomForestClassifier

    Raises:
        RuntimeError: If training fails
    """
    try:
        logger.info(f"Training Random Forest with {n_estimators} estimators...")
        start_time = time.time()

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )
        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"Model trained in {training_time:.2f} seconds")

        return model

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise RuntimeError(f"Training error: {e}")


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Test accuracy score
    """
    try:
        logger.info("Evaluating model performance...")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        return accuracy

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return 0.0


def compute_shap_values(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_samples: int = 100,
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Compute SHAP values for model explanations.

    Args:
        model: Trained model
        X_train: Training data for background distribution
        X_explain: Data to explain (typically test set or subset)
        max_samples: Maximum number of background samples (for performance)

    Returns:
        Tuple of (explainer, shap_values)

    Raises:
        RuntimeError: If SHAP computation fails

    Note:
        For large datasets, use a sample of training data as background
        to improve performance.
    """
    try:
        logger.info("Computing SHAP values...")
        start_time = time.time()

        # Use a sample of training data as background for better performance
        if len(X_train) > max_samples:
            logger.info(f"Sampling {max_samples} background samples for efficiency")
            background = shap.sample(X_train, max_samples, random_state=42)
        else:
            background = X_train

        # Create TreeExplainer (fast for tree-based models)
        explainer = shap.TreeExplainer(model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_explain)

        computation_time = time.time() - start_time
        logger.info(f"SHAP values computed in {computation_time:.2f} seconds")
        logger.info(f"SHAP values shape: {np.array(shap_values).shape}")

        return explainer, shap_values

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        raise RuntimeError(f"SHAP error: {e}")


def analyze_feature_importance(
    shap_values: np.ndarray, feature_names: list, top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze global feature importance using SHAP values.

    Args:
        shap_values: Computed SHAP values
        feature_names: List of feature names
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importance rankings
    """
    try:
        logger.info(f"\nAnalyzing top {top_n} feature importances...")

        # For binary classification, use class 1 (malignant)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)

        # Display top features
        logger.info("\nTop Features by SHAP Importance:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']:<30} {row['importance']:.6f}")

        return importance_df

    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return pd.DataFrame()


def create_visualizations(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    save_plots: bool = True,
    output_dir: str = "examples",
) -> None:
    """
    Create and optionally save SHAP visualizations.

    Args:
        explainer: SHAP explainer object
        shap_values: Computed SHAP values
        X_explain: Data used for explanations
        save_plots: Whether to save plots to disk
        output_dir: Directory to save plots

    Note:
        Visualizations are displayed and optionally saved as PNG files.
        Skips visualization if matplotlib is not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping visualizations")
        return

    try:
        logger.info("\nGenerating SHAP visualizations...")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # For binary classification, use class 1
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values

        # 1. Summary Plot (shows feature importance and effects)
        logger.info("Creating summary plot...")
        try:
            shap.summary_plot(shap_values_plot, X_explain, show=False, max_display=15)
            if save_plots:
                plot_path = output_path / "shap_summary_plot.png"
                plt.savefig(str(plot_path), bbox_inches="tight", dpi=150)
                logger.info(f"  Saved: {plot_path}")
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to create summary plot: {e}")
            plt.close()

        # 2. Bar Plot (feature importance)
        logger.info("Creating feature importance bar plot...")
        try:
            shap.summary_plot(
                shap_values_plot, X_explain, plot_type="bar", show=False, max_display=15
            )
            if save_plots:
                plot_path = output_path / "shap_bar_plot.png"
                plt.savefig(str(plot_path), bbox_inches="tight", dpi=150)
                logger.info(f"  Saved: {plot_path}")
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to create bar plot: {e}")
            plt.close()

        # 3. Waterfall Plot (single prediction explanation)
        logger.info("Creating waterfall plot for first prediction...")
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_plot[0],
                    base_values=explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
                    data=X_explain.iloc[0],
                    feature_names=X_explain.columns.tolist(),
                ),
                show=False,
            )
            if save_plots:
                plot_path = output_path / "shap_waterfall_plot.png"
                plt.savefig(str(plot_path), bbox_inches="tight", dpi=150)
                logger.info(f"  Saved: {plot_path}")
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to create waterfall plot: {e}")
            plt.close()

        logger.info("Visualizations complete!")

    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        # Ensure all plots are closed
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")


def main():
    """
    Main execution flow for basic SHAP usage example.
    """
    try:
        logger.info("=" * 80)
        logger.info("SHAP Analytics - Basic Usage Example")
        logger.info("=" * 80)

        # Step 1: Load and prepare data
        X, y, feature_names = load_and_prepare_data()

        # Step 2: Split data
        logger.info("\nSplitting data into train/test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Step 3: Train model
        model = train_model(X_train, y_train)

        # Step 4: Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        if accuracy < 0.9:
            logger.warning(f"Model accuracy ({accuracy:.4f}) is below expected threshold (0.90)")

        # Step 5: Compute SHAP values
        # Use a subset of test data for faster computation in this example
        X_explain = X_test.head(100)
        explainer, shap_values = compute_shap_values(model, X_train, X_explain, max_samples=100)

        # Step 6: Analyze feature importance
        importance_df = analyze_feature_importance(shap_values, feature_names, top_n=10)

        # Step 7: Create visualizations
        create_visualizations(explainer, shap_values, X_explain, save_plots=True)

        logger.info("\n" + "=" * 80)
        logger.info("Example completed successfully!")
        logger.info("=" * 80)

        return importance_df

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
