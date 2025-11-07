"""
Real-Time Drift Detection with SHAP Analytics
==============================================

This example demonstrates how to monitor data drift and model performance degradation
in production using SHAP values as drift indicators.

Features:
- Statistical drift detection (KS test, PSI)
- SHAP-based drift monitoring
- Feature distribution tracking
- Automated alerting thresholds
- Windowed monitoring for real-time systems
- Performance metrics correlation

Concepts:
- Data Drift: Changes in input feature distributions
- Prediction Drift: Changes in model output distributions
- SHAP Drift: Changes in feature importance patterns
- Performance Drift: Degradation in model accuracy

Requirements:
- scikit-learn
- shap
- numpy
- pandas
- scipy

Usage:
    python examples/drift_monitoring.py
"""

import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Drift Detection
# ============================================================================


@dataclass
class DriftMetrics:
    """Container for drift detection metrics."""

    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    mean_shift: float
    std_shift: float
    is_drifted: bool
    drift_severity: str  # 'none', 'low', 'medium', 'high'


@dataclass
class ShapDriftMetrics:
    """Container for SHAP-based drift metrics."""

    feature_name: str
    importance_shift: float
    correlation_change: float
    is_drifted: bool


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""

    timestamp: str
    n_samples_reference: int
    n_samples_current: int
    n_features_drifted: int
    drifted_features: List[str]
    feature_metrics: List[DriftMetrics]
    shap_metrics: List[ShapDriftMetrics]
    overall_drift_score: float
    alert_level: str  # 'green', 'yellow', 'red'


# ============================================================================
# Drift Detection Functions
# ============================================================================


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures the shift in distribution between two datasets.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Small change
    PSI >= 0.2: Significant change

    Args:
        reference: Reference (baseline) data
        current: Current (production) data
        bins: Number of bins for histogram

    Returns:
        PSI score
    """
    try:
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Calculate frequencies
        ref_freq, _ = np.histogram(reference, bins=breakpoints)
        cur_freq, _ = np.histogram(current, bins=breakpoints)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_freq = ref_freq + epsilon
        cur_freq = cur_freq + epsilon

        # Normalize to percentages
        ref_pct = ref_freq / len(reference)
        cur_pct = cur_freq / len(current)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return 0.0


def detect_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature: str,
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> DriftMetrics:
    """
    Detect drift for a single feature using multiple statistical tests.

    Args:
        reference_data: Baseline data
        current_data: Current production data
        feature: Feature name to analyze
        ks_threshold: P-value threshold for KS test
        psi_threshold: Threshold for PSI score

    Returns:
        DriftMetrics object with detection results
    """
    try:
        ref_values = reference_data[feature].values
        cur_values = current_data[feature].values

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(ref_values, cur_values)

        # Population Stability Index
        psi = calculate_psi(ref_values, cur_values)

        # Distribution statistics
        mean_shift = (np.mean(cur_values) - np.mean(ref_values)) / (np.std(ref_values) + 1e-10)
        std_shift = np.std(cur_values) / (np.std(ref_values) + 1e-10)

        # Determine drift status
        is_drifted = (ks_pval < ks_threshold) or (psi > psi_threshold)

        # Classify severity
        if psi < 0.1 and ks_pval >= ks_threshold:
            severity = "none"
        elif psi < 0.2 and ks_pval >= 0.01:
            severity = "low"
        elif psi < 0.3 or ks_pval < 0.01:
            severity = "medium"
        else:
            severity = "high"

        return DriftMetrics(
            feature_name=feature,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            psi_score=float(psi),
            mean_shift=float(mean_shift),
            std_shift=float(std_shift),
            is_drifted=is_drifted,
            drift_severity=severity,
        )

    except Exception as e:
        logger.error(f"Drift detection failed for feature {feature}: {e}")
        return DriftMetrics(
            feature_name=feature,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            psi_score=0.0,
            mean_shift=0.0,
            std_shift=1.0,
            is_drifted=False,
            drift_severity="none",
        )


def detect_shap_drift(
    reference_shap: np.ndarray,
    current_shap: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.15,
) -> List[ShapDriftMetrics]:
    """
    Detect drift in SHAP values (feature importance changes).

    Args:
        reference_shap: Reference SHAP values
        current_shap: Current SHAP values
        feature_names: List of feature names
        threshold: Importance shift threshold

    Returns:
        List of SHAP drift metrics
    """
    try:
        # Calculate mean absolute SHAP values (importance)
        ref_importance = np.abs(reference_shap).mean(axis=0)
        cur_importance = np.abs(current_shap).mean(axis=0)

        # Normalize importances
        ref_importance_norm = ref_importance / (ref_importance.sum() + 1e-10)
        cur_importance_norm = cur_importance / (cur_importance.sum() + 1e-10)

        shap_metrics = []

        for i, feature in enumerate(feature_names):
            # Calculate importance shift
            importance_shift = abs(cur_importance_norm[i] - ref_importance_norm[i])

            # Calculate correlation change (how SHAP values correlate)
            ref_corr = np.corrcoef(reference_shap[:, i], np.arange(len(reference_shap)))[0, 1]
            cur_corr = np.corrcoef(current_shap[:, i], np.arange(len(current_shap)))[0, 1]
            correlation_change = (
                abs(cur_corr - ref_corr) if not (np.isnan(ref_corr) or np.isnan(cur_corr)) else 0.0
            )

            is_drifted = importance_shift > threshold

            shap_metrics.append(
                ShapDriftMetrics(
                    feature_name=feature,
                    importance_shift=float(importance_shift),
                    correlation_change=float(correlation_change),
                    is_drifted=is_drifted,
                )
            )

        return shap_metrics

    except Exception as e:
        logger.error(f"SHAP drift detection failed: {e}")
        return []


class DriftMonitor:
    """
    Real-time drift monitoring system using windowed data.
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        reference_data: pd.DataFrame,
        feature_names: List[str],
        window_size: int = 1000,
        alert_threshold: int = 3,
    ):
        """
        Initialize drift monitor.

        Args:
            model: Trained model
            reference_data: Baseline data for comparison
            feature_names: List of feature names
            window_size: Size of sliding window for monitoring
            alert_threshold: Number of drifted features to trigger alert
        """
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        # Create SHAP explainer
        logger.info("Initializing SHAP explainer for drift monitoring...")
        self.explainer = shap.TreeExplainer(model)

        # Compute reference SHAP values
        logger.info("Computing reference SHAP values...")
        ref_sample = reference_data.sample(min(500, len(reference_data)), random_state=42)
        self.reference_shap = self.explainer.shap_values(ref_sample)
        if isinstance(self.reference_shap, list):
            self.reference_shap = self.reference_shap[1]  # Use positive class

        # Sliding window for current data
        self.current_window = deque(maxlen=window_size)

        # Metrics history
        self.drift_history = []

        logger.info(f"Drift monitor initialized with window size {window_size}")

    def add_sample(self, sample: np.ndarray) -> Optional[DriftReport]:
        """
        Add a new sample and check for drift if window is full.

        Args:
            sample: New data sample

        Returns:
            DriftReport if drift check performed, None otherwise
        """
        self.current_window.append(sample)

        # Only check drift when window is full
        if len(self.current_window) == self.window_size:
            return self.check_drift()

        return None

    def check_drift(self) -> DriftReport:
        """
        Perform comprehensive drift check on current window.

        Returns:
            DriftReport with drift detection results
        """
        logger.info("Performing drift detection...")
        start_time = time.time()

        # Convert window to DataFrame
        current_data = pd.DataFrame(list(self.current_window), columns=self.feature_names)

        # Feature-level drift detection
        feature_metrics = []
        drifted_features = []

        for feature in self.feature_names:
            metrics = detect_feature_drift(self.reference_data, current_data, feature)
            feature_metrics.append(metrics)

            if metrics.is_drifted:
                drifted_features.append(feature)

        # SHAP-based drift detection
        logger.info("Computing current SHAP values...")
        current_shap = self.explainer.shap_values(current_data.sample(min(200, len(current_data))))
        if isinstance(current_shap, list):
            current_shap = current_shap[1]

        shap_metrics = detect_shap_drift(self.reference_shap, current_shap, self.feature_names)

        # Calculate overall drift score (weighted combination)
        psi_scores = [m.psi_score for m in feature_metrics]
        overall_drift = np.mean(psi_scores)

        # Determine alert level
        n_drifted = len(drifted_features)
        if n_drifted == 0:
            alert_level = "green"
        elif n_drifted < self.alert_threshold:
            alert_level = "yellow"
        else:
            alert_level = "red"

        # Create report
        report = DriftReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_samples_reference=len(self.reference_data),
            n_samples_current=len(current_data),
            n_features_drifted=n_drifted,
            drifted_features=drifted_features,
            feature_metrics=feature_metrics,
            shap_metrics=shap_metrics,
            overall_drift_score=float(overall_drift),
            alert_level=alert_level,
        )

        # Store in history
        self.drift_history.append(report)

        elapsed = time.time() - start_time
        logger.info(f"Drift detection completed in {elapsed:.2f}s")

        return report


def print_drift_report(report: DriftReport):
    """
    Print comprehensive drift report.

    Args:
        report: DriftReport object to display
    """
    logger.info("\n" + "=" * 80)
    logger.info("DRIFT DETECTION REPORT")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {report.timestamp}")
    logger.info(f"Alert Level: {report.alert_level.upper()}")
    logger.info(f"Overall Drift Score: {report.overall_drift_score:.4f}")
    logger.info(f"Features Drifted: {report.n_features_drifted}/{len(report.feature_metrics)}")

    if report.drifted_features:
        logger.info("\nDrifted Features:")
        for feature in report.drifted_features:
            # Find metrics for this feature
            metrics = next(m for m in report.feature_metrics if m.feature_name == feature)
            logger.info(f"  - {feature}")
            logger.info(f"    PSI: {metrics.psi_score:.4f}, KS p-value: {metrics.ks_pvalue:.4f}")
            logger.info(
                f"    Severity: {metrics.drift_severity}, Mean Shift: {metrics.mean_shift:.2f}"
            )

    # SHAP drift
    shap_drifted = [m for m in report.shap_metrics if m.is_drifted]
    if shap_drifted:
        logger.info(f"\nSHAP Importance Changes ({len(shap_drifted)} features):")
        for metric in shap_drifted[:5]:  # Top 5
            logger.info(f"  - {metric.feature_name}: Shift = {metric.importance_shift:.4f}")

    logger.info("=" * 80 + "\n")


def simulate_production_data(
    reference_data: pd.DataFrame, n_samples: int, drift_severity: float = 0.0
) -> pd.DataFrame:
    """
    Simulate production data with optional drift injection.

    Args:
        reference_data: Original data
        n_samples: Number of samples to generate
        drift_severity: Amount of drift to inject (0=none, 1=severe)

    Returns:
        Simulated production data
    """
    # Sample from reference with replacement
    production_data = reference_data.sample(n_samples, replace=True)

    # Inject drift if specified
    if drift_severity > 0:
        logger.info(f"Injecting drift with severity {drift_severity:.2f}")

        # Add gaussian noise scaled by drift severity
        noise = (
            np.random.randn(*production_data.shape) * drift_severity * production_data.std().values
        )
        production_data = production_data + noise

        # Shift distributions for some features
        n_shift_features = int(len(production_data.columns) * drift_severity)
        shift_features = np.random.choice(production_data.columns, n_shift_features, replace=False)

        for feature in shift_features:
            shift = drift_severity * production_data[feature].std()
            production_data[feature] += shift

    return production_data


def main():
    """
    Main execution for drift monitoring example.
    """
    try:
        logger.info("=" * 80)
        logger.info("SHAP Analytics - Drift Monitoring Example")
        logger.info("=" * 80)

        # Load data
        logger.info("\nLoading breast cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        # Split data (reference = training set)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        logger.info("\nTraining model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Initialize drift monitor
        monitor = DriftMonitor(
            model=model,
            reference_data=X_train,
            feature_names=list(data.feature_names),
            window_size=100,
            alert_threshold=3,
        )

        # Simulate production monitoring
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 1: No Drift (Normal Operation)")
        logger.info("=" * 80)

        # Generate normal production data (no drift)
        normal_data = simulate_production_data(X_test, n_samples=100, drift_severity=0.0)

        for sample in normal_data.values:
            report = monitor.add_sample(sample)

        if report:
            print_drift_report(report)

        # Reset window for next scenario
        monitor.current_window.clear()

        # Scenario 2: Moderate drift
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 2: Moderate Drift")
        logger.info("=" * 80)

        drifted_data = simulate_production_data(X_test, n_samples=100, drift_severity=0.3)

        for sample in drifted_data.values:
            report = monitor.add_sample(sample)

        if report:
            print_drift_report(report)

        # Reset window for next scenario
        monitor.current_window.clear()

        # Scenario 3: Severe drift
        logger.info("\n" + "=" * 80)
        logger.info("SCENARIO 3: Severe Drift")
        logger.info("=" * 80)

        severe_drift_data = simulate_production_data(X_test, n_samples=100, drift_severity=0.7)

        for sample in severe_drift_data.values:
            report = monitor.add_sample(sample)

        if report:
            print_drift_report(report)

        logger.info("\n" + "=" * 80)
        logger.info("Drift monitoring example completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
