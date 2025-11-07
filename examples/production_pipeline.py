"""
End-to-End Production ML Pipeline with SHAP Analytics
=====================================================

This example demonstrates a complete production ML pipeline integrating
SHAP explanations at multiple stages: training, evaluation, deployment, and monitoring.

Pipeline Stages:
1. Data ingestion and validation
2. Feature engineering
3. Model training with hyperparameter tuning
4. Model evaluation with SHAP explanations
5. Model versioning and serialization
6. Deployment preparation
7. Monitoring and explainability tracking

Features:
- MLOps best practices
- Automated feature validation
- Model comparison with SHAP
- Explainability auditing
- Production-ready error handling
- Comprehensive logging

Requirements:
- scikit-learn
- shap
- numpy
- pandas
- joblib

Usage:
    python examples/production_pipeline.py
"""

import logging
import time
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration and Data Classes
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""

    project_name: str = "shap_breast_cancer"
    model_dir: str = "models"
    metrics_dir: str = "metrics"
    explainability_dir: str = "explainability"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    confusion_matrix: List[List[int]]


@dataclass
class ExplainabilityMetrics:
    """Container for explainability metrics."""

    top_features: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    shap_summary_stats: Dict[str, float]
    explanation_quality_score: float


@dataclass
class ModelArtifact:
    """Container for model artifact metadata."""

    model_id: str
    model_type: str
    version: str
    timestamp: str
    metrics: ModelMetrics
    explainability: ExplainabilityMetrics
    config: Dict[str, Any]
    file_path: str


class ProductionPipeline:
    """
    End-to-end production ML pipeline with SHAP integration.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.explainer = None

        # Create directories
        Path(config.model_dir).mkdir(exist_ok=True)
        Path(config.metrics_dir).mkdir(exist_ok=True)
        Path(config.explainability_dir).mkdir(exist_ok=True)

        logger.info(f"Initialized pipeline: {config.project_name}")

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and validate input data.

        Returns:
            Tuple of (features, target)

        Raises:
            ValueError: If data validation fails
        """
        logger.info("Loading and validating data...")

        try:
            # Load data
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name="target")

            # Data validation checks
            logger.info("Performing data validation...")

            # Check for missing values
            if X.isnull().any().any():
                raise ValueError("Dataset contains missing values")

            # Check for infinite values
            if np.isinf(X.values).any():
                raise ValueError("Dataset contains infinite values")

            # Check feature variance
            low_variance = X.var() < 1e-10
            if low_variance.any():
                logger.warning(
                    f"Features with low variance: {low_variance[low_variance].index.tolist()}"
                )

            # Check target distribution
            class_counts = y.value_counts()
            minority_ratio = class_counts.min() / class_counts.sum()
            logger.info(f"Target distribution: {class_counts.to_dict()}")
            logger.info(f"Minority class ratio: {minority_ratio:.2%}")

            if minority_ratio < 0.05:
                logger.warning("Severe class imbalance detected")

            logger.info(f"Data validated: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y

        except Exception as e:
            logger.error(f"Data loading/validation failed: {e}")
            raise

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering.

        Args:
            X: Input features

        Returns:
            Engineered features
        """
        logger.info("Engineering features...")

        # For this example, we'll keep original features
        # In production, you might add:
        # - Feature interactions
        # - Polynomial features
        # - Domain-specific transformations

        X_engineered = X.copy()

        # Example: Add some statistical features
        X_engineered["mean_features"] = X.mean(axis=1)
        X_engineered["std_features"] = X.std(axis=1)
        X_engineered["max_features"] = X.max(axis=1)

        logger.info(f"Features after engineering: {X_engineered.shape[1]}")

        return X_engineered

    def train_and_evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "random_forest",
    ) -> Tuple[Any, ModelMetrics]:
        """
        Train and evaluate a model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model to train

        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training {model_type} model...")
        start_time = time.time()

        # Initialize model
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"Model trained in {training_time:.2f}s")

        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=self.config.cv_folds, scoring="accuracy", n_jobs=-1
        )

        # Validation predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, zero_division=0),
            recall=recall_score(y_val, y_pred, zero_division=0),
            f1_score=f1_score(y_val, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_val, y_pred_proba),
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            confusion_matrix=confusion_matrix(y_val, y_pred).tolist(),
        )

        # Log metrics
        logger.info(f"Validation Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Validation F1 Score: {metrics.f1_score:.4f}")
        logger.info(f"CV Accuracy: {metrics.cv_mean:.4f} (+/- {metrics.cv_std:.4f})")

        return model, metrics

    def compute_explainability_metrics(
        self, model: Any, X_explain: pd.DataFrame, top_n: int = 10
    ) -> ExplainabilityMetrics:
        """
        Compute explainability metrics using SHAP.

        Args:
            model: Trained model
            X_explain: Data to explain
            top_n: Number of top features

        Returns:
            ExplainabilityMetrics object
        """
        logger.info("Computing explainability metrics...")

        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_explain)

            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Calculate feature importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(X_explain.columns, mean_abs_shap))

            # Top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [
                {"feature": feat, "importance": float(imp), "rank": i + 1}
                for i, (feat, imp) in enumerate(sorted_features[:top_n])
            ]

            # SHAP summary statistics
            shap_summary_stats = {
                "mean_abs_shap": float(np.abs(shap_values).mean()),
                "std_abs_shap": float(np.abs(shap_values).std()),
                "max_abs_shap": float(np.abs(shap_values).max()),
                "min_abs_shap": float(np.abs(shap_values).min()),
            }

            # Explanation quality score (higher = more explainable)
            # Based on concentration of importance in top features
            top_importance_ratio = sum([f["importance"] for f in top_features]) / sum(
                feature_importance.values()
            )
            explanation_quality_score = float(top_importance_ratio)

            metrics = ExplainabilityMetrics(
                top_features=top_features,
                feature_importance=feature_importance,
                shap_summary_stats=shap_summary_stats,
                explanation_quality_score=explanation_quality_score,
            )

            logger.info(f"Explainability quality score: {explanation_quality_score:.4f}")
            logger.info(f"Top 3 features: {[f['feature'] for f in top_features[:3]]}")

            self.explainer = explainer  # Store for later use

            return metrics

        except Exception as e:
            logger.error(f"Explainability computation failed: {e}")
            raise

    def compare_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> ModelArtifact:
        """
        Compare multiple models and select the best.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Best model artifact
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPARING MODELS")
        logger.info("=" * 80)

        models_to_compare = ["random_forest", "gradient_boosting"]
        results = []

        for model_type in models_to_compare:
            logger.info(f"\nEvaluating {model_type}...")

            # Train model
            model, metrics = self.train_and_evaluate_model(
                X_train, y_train, X_val, y_val, model_type
            )

            # Compute explainability
            explainability = self.compute_explainability_metrics(
                model, X_val.sample(min(200, len(X_val)))
            )

            # Store results
            results.append(
                {
                    "model_type": model_type,
                    "model": model,
                    "metrics": metrics,
                    "explainability": explainability,
                }
            )

        # Select best model (based on F1 score and explainability)
        best_result = max(
            results,
            key=lambda x: x["metrics"].f1_score * 0.7
            + x["explainability"].explanation_quality_score * 0.3,
        )

        logger.info("\n" + "=" * 80)
        logger.info(f"BEST MODEL: {best_result['model_type']}")
        logger.info(f"F1 Score: {best_result['metrics'].f1_score:.4f}")
        logger.info(
            f"Explainability Score: {best_result['explainability'].explanation_quality_score:.4f}"
        )
        logger.info("=" * 80)

        # Create artifact
        artifact = ModelArtifact(
            model_id=f"{self.config.project_name}_{int(time.time())}",
            model_type=best_result["model_type"],
            version="1.0.0",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=best_result["metrics"],
            explainability=best_result["explainability"],
            config=asdict(self.config),
            file_path="",  # Will be set during save
        )

        self.model = best_result["model"]

        return artifact

    def save_model_artifact(self, artifact: ModelArtifact, X_sample: pd.DataFrame) -> str:
        """
        Save model artifact with metadata.

        Args:
            artifact: Model artifact to save
            X_sample: Sample data for explainer

        Returns:
            Path to saved artifact
        """
        logger.info("Saving model artifact...")

        # Save model
        model_path = os.path.join(self.config.model_dir, f"{artifact.model_id}.joblib")
        joblib.dump(
            {
                "model": self.model,
                "explainer": self.explainer,
                "feature_names": X_sample.columns.tolist(),
            },
            model_path,
        )

        artifact.file_path = model_path

        # Save metadata
        metadata_path = os.path.join(self.config.metrics_dir, f"{artifact.model_id}_metadata.json")
        metadata = {
            "model_id": artifact.model_id,
            "model_type": artifact.model_type,
            "version": artifact.version,
            "timestamp": artifact.timestamp,
            "metrics": asdict(artifact.metrics),
            "explainability": asdict(artifact.explainability),
            "config": artifact.config,
            "file_path": model_path,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

        return model_path

    def run_pipeline(self) -> ModelArtifact:
        """
        Execute the complete pipeline.

        Returns:
            Final model artifact
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING PRODUCTION PIPELINE")
        logger.info("=" * 80)

        pipeline_start = time.time()

        try:
            # Stage 1: Load and validate data
            X, y = self.load_and_validate_data()

            # Stage 2: Feature engineering
            X = self.engineer_features(X)

            # Stage 3: Split data
            logger.info("\nSplitting data...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=self.config.val_size / (1 - self.config.test_size),
                random_state=self.config.random_state,
                stratify=y_temp,
            )

            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Stage 4: Model comparison and selection
            artifact = self.compare_models(X_train, y_train, X_val, y_val)

            # Stage 5: Final evaluation on test set
            logger.info("\nFinal evaluation on test set...")
            y_test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            logger.info(f"Test Set Accuracy: {test_accuracy:.4f}")

            # Stage 6: Save artifact
            model_path = self.save_model_artifact(artifact, X_train)

            # Pipeline summary
            pipeline_time = time.time() - pipeline_start

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total time: {pipeline_time:.2f}s")
            logger.info(f"Model ID: {artifact.model_id}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Validation F1: {artifact.metrics.f1_score:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info("=" * 80)

            return artifact

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """
    Main entry point for production pipeline.
    """
    # Configure pipeline
    config = PipelineConfig(
        project_name="shap_breast_cancer_production", test_size=0.2, val_size=0.1, cv_folds=5
    )

    # Initialize and run pipeline
    pipeline = ProductionPipeline(config)
    artifact = pipeline.run_pipeline()

    return artifact


if __name__ == "__main__":
    main()
