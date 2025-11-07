"""
Custom SHAP Visualizations with Plotly Dashboards
=================================================

This example demonstrates advanced interactive visualizations for SHAP explanations
using Plotly, creating production-ready dashboards for ML explainability.

Features:
- Interactive feature importance charts
- Force plots for individual predictions
- Dependence plots with interactions
- Multi-dimensional SHAP analysis
- Real-time dashboard with Plotly Dash
- Export capabilities for reports

Visualization Types:
1. Feature Importance Bar Charts
2. SHAP Value Distribution Plots
3. Feature Dependence Scatter Plots
4. Decision Path Waterfall Charts
5. Correlation Heatmaps
6. Interactive Dashboard

Requirements:
- plotly
- dash (optional, for interactive dashboard)
- pandas
- numpy
- scikit-learn
- shap

Usage:
    # Generate static visualizations
    python examples/custom_visualization.py

    # Launch interactive dashboard
    python examples/custom_visualization.py --dashboard
"""

import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
# Visualization Functions
# ============================================================================


def create_feature_importance_bar(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 15,
    title: str = "SHAP Feature Importance",
) -> go.Figure:
    """
    Create interactive bar chart for feature importance.

    Args:
        feature_names: List of feature names
        importance_values: Mean absolute SHAP values
        top_n: Number of top features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Create DataFrame and sort
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    # Create bar chart
    fig = go.Figure(
        [
            go.Bar(
                x=df["importance"][::-1],
                y=df["feature"][::-1],
                orientation="h",
                marker=dict(
                    color=df["importance"][::-1],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Importance"),
                ),
                text=df["importance"][::-1].round(4),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Arial, sans-serif")),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        height=max(400, top_n * 30),
        template="plotly_white",
        hovermode="closest",
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50),
    )

    return fig


def create_shap_beeswarm(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    top_n: int = 15,
    title: str = "SHAP Value Distribution",
) -> go.Figure:
    """
    Create beeswarm plot showing SHAP value distributions.

    Args:
        shap_values: SHAP values array
        features: Feature DataFrame
        top_n: Number of features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Calculate feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    # Add traces for each feature
    for idx in top_indices:
        feature_name = features.columns[idx]
        feature_values = features.iloc[:, idx].values
        shap_vals = shap_values[:, idx]

        # Normalize feature values for color
        feature_norm = (feature_values - feature_values.min()) / (
            feature_values.max() - feature_values.min() + 1e-10
        )

        fig.add_trace(
            go.Scatter(
                x=shap_vals,
                y=[feature_name] * len(shap_vals),
                mode="markers",
                marker=dict(
                    size=5,
                    color=feature_norm,
                    colorscale="RdBu",
                    showscale=False,
                    opacity=0.6,
                    line=dict(width=0.5, color="white"),
                ),
                name=feature_name,
                hovertemplate=f"<b>{feature_name}</b><br>SHAP: %{{x:.4f}}<br>Feature Value: %{{customdata:.2f}}<extra></extra>",
                customdata=feature_values,
            )
        )

    fig.update_layout(
        height=max(400, top_n * 40),
        template="plotly_white",
        xaxis_title="SHAP Value",
        yaxis_title="",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=200, r=50, t=80, b=50),
    )

    return fig


def create_dependence_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_idx: int,
    interaction_idx: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create SHAP dependence plot for a feature.

    Args:
        shap_values: SHAP values array
        features: Feature DataFrame
        feature_idx: Index of feature to plot
        interaction_idx: Index of interaction feature for coloring
        title: Chart title

    Returns:
        Plotly figure object
    """
    feature_name = features.columns[feature_idx]
    feature_values = features.iloc[:, feature_idx].values
    shap_vals = shap_values[:, feature_idx]

    # Determine interaction feature
    if interaction_idx is None:
        # Find feature with highest correlation to SHAP values
        correlations = [
            abs(np.corrcoef(features.iloc[:, i], shap_vals)[0, 1]) if i != feature_idx else 0
            for i in range(features.shape[1])
        ]
        interaction_idx = np.argmax(correlations)

    interaction_name = features.columns[interaction_idx]
    interaction_values = features.iloc[:, interaction_idx].values

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=feature_values,
            y=shap_vals,
            mode="markers",
            marker=dict(
                size=6,
                color=interaction_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text=f"<b>{interaction_name}</b>", side="right")),
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=f"<b>{feature_name}</b>: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<br>{interaction_name}: %{{marker.color:.2f}}<extra></extra>",
        )
    )

    if title is None:
        title = f"SHAP Dependence: {feature_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=feature_name,
        yaxis_title=f"SHAP value for {feature_name}",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )

    return fig


def create_waterfall_chart(
    shap_values: np.ndarray,
    features: pd.Series,
    base_value: float,
    max_display: int = 10,
    title: str = "SHAP Waterfall - Individual Prediction",
) -> go.Figure:
    """
    Create waterfall chart for individual prediction explanation.

    Args:
        shap_values: SHAP values for single prediction
        features: Feature values for single prediction
        base_value: Expected value (base)
        max_display: Maximum features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Get top contributing features
    importance_order = np.argsort(np.abs(shap_values))[::-1]
    top_indices = importance_order[:max_display]

    # Build waterfall data
    feature_names = []
    contributions = []
    cumulative = [base_value]

    for idx in top_indices:
        feature_names.append(f"{features.index[idx]}<br>= {features.iloc[idx]:.2f}")
        contributions.append(shap_values[idx])
        cumulative.append(cumulative[-1] + shap_values[idx])

    # Add final prediction
    feature_names.append("<b>Prediction</b>")
    contributions.append(0)

    # Create waterfall chart
    fig = go.Figure()

    # Add bars
    for i, (name, contrib) in enumerate(zip(feature_names[:-1], contributions[:-1])):
        color = "rgb(255, 120, 120)" if contrib < 0 else "rgb(120, 180, 255)"

        fig.add_trace(
            go.Waterfall(
                name=name,
                orientation="v",
                measure=["relative"] * len(contributions),
                x=feature_names,
                y=contributions,
                connector={"line": {"color": "rgb(100, 100, 100)"}},
                decreasing={"marker": {"color": "rgb(255, 120, 120)"}},
                increasing={"marker": {"color": "rgb(120, 180, 255)"}},
                totals={"marker": {"color": "rgb(100, 200, 100)"}},
                hovertemplate="<b>%{x}</b><br>Contribution: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title="Model Output Value",
        height=500,
        template="plotly_white",
        showlegend=False,
        xaxis={"categoryorder": "array", "categoryarray": feature_names},
    )

    return fig


def create_shap_correlation_heatmap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "SHAP Value Correlation Heatmap",
) -> go.Figure:
    """
    Create correlation heatmap of SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Select top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    # Calculate correlation matrix
    shap_subset = shap_values[:, top_indices]
    corr_matrix = np.corrcoef(shap_subset.T)

    top_feature_names = [feature_names[i] for i in top_indices]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=top_feature_names,
            y=top_feature_names,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=max(600, top_n * 30),
        width=max(600, top_n * 30),
        template="plotly_white",
        xaxis={"tickangle": -45},
        margin=dict(l=150, r=50, t=100, b=150),
    )

    return fig


def create_multi_class_comparison(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    top_n: int = 10,
    title: str = "Feature Importance by Class",
) -> go.Figure:
    """
    Create comparison of feature importance across classes.

    Args:
        shap_values: SHAP values for each class (list of arrays)
        feature_names: List of feature names
        class_names: List of class names
        top_n: Number of features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Calculate importance for each class
    for i, class_name in enumerate(class_names):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        fig.add_trace(
            go.Bar(
                name=f"Class {class_name}",
                x=importance_df["feature"],
                y=importance_df["importance"],
                hovertemplate="<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP Value|",
        barmode="group",
        height=500,
        template="plotly_white",
        hovermode="closest",
        xaxis={"tickangle": -45},
    )

    return fig


def create_decision_plot(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    features: pd.DataFrame,
    sample_indices: Optional[List[int]] = None,
    max_samples: int = 20,
    title: str = "SHAP Decision Plot",
) -> go.Figure:
    """
    Create decision plot showing cumulative SHAP contributions.

    Decision plots show how features contribute to moving the prediction
    from the base value to the final prediction for multiple samples.

    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        features: Feature DataFrame
        sample_indices: Specific samples to plot (if None, selects random)
        max_samples: Maximum number of samples to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Select samples to plot
        if sample_indices is None:
            n_samples = min(max_samples, len(features))
            sample_indices = np.random.choice(len(features), n_samples, replace=False)

        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # Binary classification

        fig = go.Figure()

        # Calculate cumulative SHAP for each sample
        for idx in sample_indices:
            # Sort features by SHAP value for this sample
            shap_sample = shap_values[idx]
            sort_idx = np.argsort(np.abs(shap_sample))

            # Cumulative sum
            cumsum = np.concatenate([[base_value], base_value + np.cumsum(shap_sample[sort_idx])])
            feature_order = ["Base"] + [features.columns[i] for i in sort_idx]

            fig.add_trace(
                go.Scatter(
                    x=cumsum,
                    y=list(range(len(cumsum))),
                    mode="lines+markers",
                    name=f"Sample {idx}",
                    line=dict(width=1.5),
                    marker=dict(size=4),
                    hovertemplate="<b>%{customdata}</b><br>Value: %{x:.4f}<extra></extra>",
                    customdata=feature_order,
                    opacity=0.7,
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="Model Output Value",
            yaxis_title="Feature (ordered by |SHAP|)",
            height=600,
            template="plotly_white",
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
        )

        return fig

    except Exception as e:
        logger.error(f"Decision plot creation failed: {e}")
        return go.Figure()


def create_force_plot_html(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    features: pd.DataFrame,
    sample_idx: int = 0,
    output_file: Optional[str] = None,
) -> str:
    """
    Generate interactive SHAP force plot as HTML.

    Force plots visualize how features push the prediction from the base value
    to the final output for a single prediction.

    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        features: Feature DataFrame
        sample_idx: Index of sample to visualize
        output_file: Path to save HTML file (if None, returns HTML string)

    Returns:
        HTML string or file path
    """
    try:
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
            shap_vals = (
                shap_values[sample_idx]
                if len(shap_values.shape) == 2
                else shap_values[1][sample_idx]
            )
        else:
            shap_vals = shap_values[sample_idx]

        # Generate force plot
        force_plot = shap.force_plot(
            base_value, shap_vals, features.iloc[sample_idx], matplotlib=False
        )

        # Save or return HTML
        html_str = f"<html><head></head><body>{shap.getjs()}{force_plot.html()}</body></html>"

        if output_file:
            with open(output_file, "w") as f:
                f.write(html_str)
            logger.info(f"Force plot saved to {output_file}")
            return output_file

        return html_str

    except Exception as e:
        logger.error(f"Force plot generation failed: {e}")
        return ""


def create_interaction_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature1_idx: int,
    feature2_idx: int,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create 2D interaction plot between two features.

    Shows how the interaction between two features affects SHAP values.

    Args:
        shap_values: SHAP values array
        features: Feature DataFrame
        feature1_idx: Index of first feature
        feature2_idx: Index of second feature
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        feature1_name = features.columns[feature1_idx]
        feature2_name = features.columns[feature2_idx]

        feature1_vals = features.iloc[:, feature1_idx].values
        feature2_vals = features.iloc[:, feature2_idx].values
        shap1_vals = shap_values[:, feature1_idx]

        # Create 2D scatter
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=feature1_vals,
                y=feature2_vals,
                mode="markers",
                marker=dict(
                    size=8,
                    color=shap1_vals,
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(title=f"SHAP for<br>{feature1_name}", titleside="right"),
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=f"<b>{feature1_name}</b>: %{{x:.2f}}<br><b>{feature2_name}</b>: %{{y:.2f}}<br>SHAP: %{{marker.color:.4f}}<extra></extra>",
            )
        )

        if title is None:
            title = f"SHAP Interaction: {feature1_name} vs {feature2_name}"

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=feature1_name,
            yaxis_title=feature2_name,
            template="plotly_white",
            height=500,
            hovermode="closest",
        )

        return fig

    except Exception as e:
        logger.error(f"Interaction plot creation failed: {e}")
        return go.Figure()


def create_partial_dependence_plot(
    model: Any,
    features: pd.DataFrame,
    feature_idx: int,
    num_points: int = 50,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create partial dependence plot using SHAP.

    Shows the marginal effect of a feature on model predictions.

    Args:
        model: Trained model
        features: Feature DataFrame
        feature_idx: Index of feature to analyze
        num_points: Number of points for PD curve
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        feature_name = features.columns[feature_idx]
        feature_vals = features.iloc[:, feature_idx].values

        # Create grid of feature values
        feature_range = np.linspace(feature_vals.min(), feature_vals.max(), num_points)

        # Calculate predictions for each point
        predictions = []
        for val in feature_range:
            # Create dataset with feature fixed at val
            X_temp = features.copy()
            X_temp.iloc[:, feature_idx] = val

            # Get mean prediction
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_temp)[:, 1].mean()
            else:
                pred = model.predict(X_temp).mean()

            predictions.append(pred)

        # Create line plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=feature_range,
                y=predictions,
                mode="lines",
                line=dict(color="blue", width=3),
                name="Partial Dependence",
                hovertemplate=f"<b>{feature_name}</b>: %{{x:.2f}}<br>Prediction: %{{y:.4f}}<extra></extra>",
            )
        )

        # Add rug plot showing data distribution
        fig.add_trace(
            go.Scatter(
                x=feature_vals,
                y=[predictions[0]] * len(feature_vals),
                mode="markers",
                marker=dict(symbol="line-ns", size=10, color="rgba(0,0,0,0.3)", line=dict(width=1)),
                name="Data Distribution",
                hovertemplate=f"<b>{feature_name}</b>: %{{x:.2f}}<extra></extra>",
            )
        )

        if title is None:
            title = f"Partial Dependence: {feature_name}"

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=feature_name,
            yaxis_title="Partial Dependence (Mean Prediction)",
            template="plotly_white",
            height=500,
            hovermode="closest",
            showlegend=True,
        )

        return fig

    except Exception as e:
        logger.error(f"Partial dependence plot creation failed: {e}")
        return go.Figure()


def create_shap_violin_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "SHAP Value Distributions (Violin Plot)",
) -> go.Figure:
    """
    Create violin plot showing SHAP value distributions for top features.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of features to display
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Get top features by importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

        fig = go.Figure()

        # Add violin for each feature
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_vals = shap_values[:, idx]

            fig.add_trace(
                go.Violin(
                    y=shap_vals,
                    name=feature_name,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor="lightblue",
                    opacity=0.6,
                    hovertemplate=f"<b>{feature_name}</b><br>SHAP: %{{y:.4f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            yaxis_title="SHAP Value",
            xaxis_title="Feature",
            height=500,
            template="plotly_white",
            showlegend=False,
            xaxis={"tickangle": -45},
        )

        return fig

    except Exception as e:
        logger.error(f"Violin plot creation failed: {e}")
        return go.Figure()


class ShapVisualizationSuite:
    """
    Comprehensive visualization suite for SHAP analytics.
    """

    def __init__(self, model, X_train: pd.DataFrame, X_explain: pd.DataFrame):
        """
        Initialize visualization suite.

        Args:
            model: Trained model
            X_train: Training data for background
            X_explain: Data to explain
        """
        logger.info("Initializing SHAP visualization suite...")

        self.model = model
        self.X_train = X_train
        self.X_explain = X_explain
        self.feature_names = list(X_explain.columns)

        # Create explainer
        logger.info("Creating SHAP explainer...")
        background = shap.sample(X_train, min(100, len(X_train)))
        self.explainer = shap.TreeExplainer(model, background)

        # Compute SHAP values
        logger.info("Computing SHAP values...")
        self.shap_values = self.explainer.shap_values(X_explain)

        # Handle binary classification
        if isinstance(self.shap_values, list):
            self.shap_values_class1 = self.shap_values[1]
            self.base_value = self.explainer.expected_value[1]
        else:
            self.shap_values_class1 = self.shap_values
            self.base_value = self.explainer.expected_value

        logger.info("Visualization suite ready!")

    def generate_all_plots(self, output_dir: str = "examples/visualizations"):
        """
        Generate all visualization plots.

        Args:
            output_dir: Directory to save plots
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        logger.info("Generating all visualizations...")

        # 1. Feature Importance
        logger.info("Creating feature importance chart...")
        mean_abs_shap = np.abs(self.shap_values_class1).mean(axis=0)
        fig1 = create_feature_importance_bar(self.feature_names, mean_abs_shap, top_n=15)
        fig1.write_html(os.path.join(output_dir, "feature_importance.html"))

        # 2. Beeswarm plot
        logger.info("Creating beeswarm plot...")
        fig2 = create_shap_beeswarm(self.shap_values_class1, self.X_explain, top_n=15)
        fig2.write_html(os.path.join(output_dir, "beeswarm_plot.html"))

        # 3. Dependence plots (top 3 features)
        logger.info("Creating dependence plots...")
        top_feature_indices = np.argsort(mean_abs_shap)[-3:][::-1]
        for idx in top_feature_indices:
            fig = create_dependence_plot(self.shap_values_class1, self.X_explain, idx)
            feature_name = self.feature_names[idx].replace(" ", "_")
            fig.write_html(os.path.join(output_dir, f"dependence_{feature_name}.html"))

        # 4. Waterfall charts (first 3 predictions)
        logger.info("Creating waterfall charts...")
        for i in range(min(3, len(self.X_explain))):
            fig = create_waterfall_chart(
                self.shap_values_class1[i],
                self.X_explain.iloc[i],
                self.base_value,
                title=f"SHAP Waterfall - Prediction {i + 1}",
            )
            fig.write_html(os.path.join(output_dir, f"waterfall_prediction_{i + 1}.html"))

        # 5. Correlation heatmap
        logger.info("Creating correlation heatmap...")
        fig5 = create_shap_correlation_heatmap(
            self.shap_values_class1, self.feature_names, top_n=15
        )
        fig5.write_html(os.path.join(output_dir, "correlation_heatmap.html"))

        # 6. Decision plot (new)
        logger.info("Creating decision plot...")
        fig6 = create_decision_plot(
            self.explainer, self.shap_values_class1, self.X_explain, max_samples=10
        )
        fig6.write_html(os.path.join(output_dir, "decision_plot.html"))

        # 7. Force plots (new) - first 3 predictions
        logger.info("Creating force plots...")
        for i in range(min(3, len(self.X_explain))):
            output_file = os.path.join(output_dir, f"force_plot_{i + 1}.html")
            create_force_plot_html(
                self.explainer,
                self.shap_values_class1,
                self.X_explain,
                sample_idx=i,
                output_file=output_file,
            )

        # 8. Interaction plots (new) - top 2 features
        logger.info("Creating interaction plots...")
        if len(top_feature_indices) >= 2:
            idx1, idx2 = top_feature_indices[0], top_feature_indices[1]
            fig8 = create_interaction_plot(self.shap_values_class1, self.X_explain, idx1, idx2)
            fig8.write_html(os.path.join(output_dir, "interaction_plot.html"))

        # 9. Partial dependence plots (new) - top 3 features
        logger.info("Creating partial dependence plots...")
        for idx in top_feature_indices:
            fig = create_partial_dependence_plot(self.model, self.X_explain, idx)
            feature_name = self.feature_names[idx].replace(" ", "_")
            fig.write_html(os.path.join(output_dir, f"partial_dependence_{feature_name}.html"))

        # 10. Violin plot (new)
        logger.info("Creating violin plot...")
        fig10 = create_shap_violin_plot(self.shap_values_class1, self.feature_names, top_n=10)
        fig10.write_html(os.path.join(output_dir, "violin_plot.html"))

        logger.info(f"\nAll visualizations saved to {output_dir}")
        logger.info(f"Total plots generated: {len(os.listdir(output_dir))}")


def main():
    """
    Main execution for visualization example.
    """
    logger.info("=" * 80)
    logger.info("SHAP Analytics - Custom Visualization Example")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split and train
    logger.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Create visualization suite
    logger.info("\nCreating visualizations...")
    X_explain = X_test.head(100)
    viz_suite = ShapVisualizationSuite(model, X_train, X_explain)

    # Generate all plots
    viz_suite.generate_all_plots()

    logger.info("\n" + "=" * 80)
    logger.info("Visualization example completed!")
    logger.info("Open the HTML files in examples/visualizations/ to view interactive plots")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
