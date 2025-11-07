"""
Tests for SHAP Analytics CLI.

This module provides comprehensive tests for all CLI commands including:
- compute: SHAP value computation
- validate: Background sample validation
- monitor: Feature drift monitoring
- serve: FastAPI server
- export: Result export
- completion: Shell completion generation
"""

import json
import tempfile

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import shap

from click.testing import CliRunner
from sklearn.ensemble import RandomForestClassifier

from shap_analytics.cli import cli


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def runner():
    """Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = np.random.randint(0, 2, n_samples)

    return X, y


@pytest.fixture
def sample_model(sample_data):
    """Trained sample model."""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_shap_values(sample_model, sample_data):
    """Computed SHAP values."""
    X, _ = sample_data
    X_train = X[:80]
    X_test = X[80:]

    explainer = shap.TreeExplainer(sample_model)
    shap_values = explainer(X_test)

    return shap_values, X_test


@pytest.fixture
def config_file(temp_dir):
    """Sample configuration file."""
    config = {"background_size": 50, "threshold": 0.15, "bins": 15}

    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return config_path


# ============================================================================
# CLI Group Tests
# ============================================================================


def test_cli_help(runner):
    """Test CLI help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "SHAP Analytics CLI" in result.output
    assert "compute" in result.output
    assert "validate" in result.output
    assert "monitor" in result.output
    assert "serve" in result.output
    assert "export" in result.output


def test_cli_version(runner):
    """Test CLI version display."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_verbose_flag(runner):
    """Test verbose flag."""
    result = runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0


# ============================================================================
# Compute Command Tests
# ============================================================================


def test_compute_help(runner):
    """Test compute command help."""
    result = runner.invoke(cli, ["compute", "--help"])
    assert result.exit_code == 0
    assert "Compute SHAP values" in result.output
    assert "--model" in result.output
    assert "--train-data" in result.output
    assert "--test-data" in result.output


def test_compute_missing_arguments(runner):
    """Test compute command with missing arguments."""
    result = runner.invoke(cli, ["compute"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_compute_success(runner, temp_dir, sample_data, sample_model):
    """Test successful SHAP computation."""
    X, y = sample_data
    X_train = X[:80]
    X_test = X[80:]

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    output_path = temp_dir / "shap.joblib"

    joblib.dump(sample_model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    # Run command
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(output_path),
            "--background-size",
            "10",
        ],
    )

    # Check result
    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_path.exists()

    # Verify output content
    result_data = joblib.load(output_path)
    assert "shap_values" in result_data
    assert "feature_names" in result_data
    assert "metadata" in result_data


def test_compute_with_config(runner, temp_dir, sample_data, sample_model, config_file):
    """Test compute with configuration file."""
    X, y = sample_data
    X_train = X[:80]
    X_test = X[80:]

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    output_path = temp_dir / "shap.joblib"

    joblib.dump(sample_model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    # Run command with config
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(output_path),
            "--config",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"


def test_compute_with_verify(runner, temp_dir, sample_data, sample_model):
    """Test compute with verification."""
    X, y = sample_data
    X_train = X[:80]
    X_test = X[80:]

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    output_path = temp_dir / "shap.joblib"

    joblib.dump(sample_model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    # Run command with verify
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(output_path),
            "--verify",
        ],
    )

    assert result.exit_code == 0


def test_compute_dimension_mismatch(runner, temp_dir, sample_model):
    """Test compute with mismatched dimensions."""
    # Create data with different dimensions
    X_train = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])
    X_test = pd.DataFrame(np.random.randn(20, 3), columns=[f"f{i}" for i in range(3)])

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    output_path = temp_dir / "shap.joblib"

    joblib.dump(sample_model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    # Run command
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code != 0
    assert "mismatch" in result.output.lower()


# ============================================================================
# Validate Command Tests
# ============================================================================


def test_validate_help(runner):
    """Test validate command help."""
    result = runner.invoke(cli, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Validate background sample" in result.output


def test_validate_success(runner, temp_dir, sample_data):
    """Test successful validation."""
    X, _ = sample_data
    data_path = temp_dir / "data.csv"
    X.to_csv(data_path, index=False)

    result = runner.invoke(
        cli, ["validate", "--data", str(data_path), "--sample-size", "50", "--threshold", "0.1"]
    )

    assert result.exit_code == 0


def test_validate_with_config(runner, temp_dir, sample_data, config_file):
    """Test validate with configuration."""
    X, _ = sample_data
    data_path = temp_dir / "data.csv"
    X.to_csv(data_path, index=False)

    result = runner.invoke(cli, ["validate", "--data", str(data_path), "--config", str(config_file)])

    assert result.exit_code == 0


def test_validate_missing_data(runner, temp_dir):
    """Test validate with missing data file."""
    result = runner.invoke(cli, ["validate", "--data", str(temp_dir / "missing.csv")])

    assert result.exit_code != 0


# ============================================================================
# Monitor Command Tests
# ============================================================================


def test_monitor_help(runner):
    """Test monitor command help."""
    result = runner.invoke(cli, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "Monitor feature drift" in result.output


def test_monitor_success(runner, temp_dir, sample_data):
    """Test successful drift monitoring."""
    X, _ = sample_data
    X_train = X[:50]
    X_new = X[50:]

    train_path = temp_dir / "train.csv"
    new_path = temp_dir / "new.csv"

    X_train.to_csv(train_path, index=False)
    X_new.to_csv(new_path, index=False)

    result = runner.invoke(
        cli,
        [
            "monitor",
            "--train-data",
            str(train_path),
            "--new-data",
            str(new_path),
            "--threshold",
            "0.2",
        ],
    )

    assert result.exit_code == 0


def test_monitor_with_output(runner, temp_dir, sample_data):
    """Test monitor with output file."""
    X, _ = sample_data
    X_train = X[:50]
    X_new = X[50:]

    train_path = temp_dir / "train.csv"
    new_path = temp_dir / "new.csv"
    output_path = temp_dir / "drift_report.json"

    X_train.to_csv(train_path, index=False)
    X_new.to_csv(new_path, index=False)

    result = runner.invoke(
        cli,
        [
            "monitor",
            "--train-data",
            str(train_path),
            "--new-data",
            str(new_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    # Verify report content
    with open(output_path) as f:
        report = json.load(f)
    assert "summary" in report
    assert "drift_scores" in report


def test_monitor_dimension_mismatch(runner, temp_dir):
    """Test monitor with dimension mismatch."""
    X_train = pd.DataFrame(np.random.randn(50, 5))
    X_new = pd.DataFrame(np.random.randn(30, 3))

    train_path = temp_dir / "train.csv"
    new_path = temp_dir / "new.csv"

    X_train.to_csv(train_path, index=False)
    X_new.to_csv(new_path, index=False)

    result = runner.invoke(
        cli, ["monitor", "--train-data", str(train_path), "--new-data", str(new_path)]
    )

    assert result.exit_code != 0
    assert "mismatch" in result.output.lower()


# ============================================================================
# Export Command Tests
# ============================================================================


def test_export_help(runner):
    """Test export command help."""
    result = runner.invoke(cli, ["export", "--help"])
    assert result.exit_code == 0
    assert "Export SHAP values" in result.output


def test_export_csv(runner, temp_dir, sample_shap_values):
    """Test export to CSV."""
    shap_values, X_test = sample_shap_values

    # Save SHAP values
    input_path = temp_dir / "shap.joblib"
    result_data = {
        "shap_values": shap_values,
        "feature_names": list(X_test.columns),
        "test_data": X_test,
        "metadata": {"n_samples": len(X_test)},
    }
    joblib.dump(result_data, input_path)

    output_path = temp_dir / "output.csv"

    result = runner.invoke(
        cli, ["export", "--input", str(input_path), "--output", str(output_path), "--format", "csv"]
    )

    assert result.exit_code == 0
    assert output_path.exists()

    # Verify CSV content
    df = pd.read_csv(output_path)
    assert len(df.columns) == len(X_test.columns)


def test_export_parquet(runner, temp_dir, sample_shap_values):
    """Test export to Parquet."""
    shap_values, X_test = sample_shap_values

    # Save SHAP values
    input_path = temp_dir / "shap.joblib"
    result_data = {
        "shap_values": shap_values,
        "feature_names": list(X_test.columns),
        "test_data": X_test,
    }
    joblib.dump(result_data, input_path)

    output_path = temp_dir / "output.parquet"

    result = runner.invoke(
        cli,
        ["export", "--input", str(input_path), "--output", str(output_path), "--format", "parquet"],
    )

    assert result.exit_code == 0
    assert output_path.exists()


def test_export_json(runner, temp_dir, sample_shap_values):
    """Test export to JSON."""
    shap_values, X_test = sample_shap_values

    # Save SHAP values
    input_path = temp_dir / "shap.joblib"
    result_data = {
        "shap_values": shap_values,
        "feature_names": list(X_test.columns),
        "test_data": X_test,
    }
    joblib.dump(result_data, input_path)

    output_path = temp_dir / "output.json"

    result = runner.invoke(
        cli, ["export", "--input", str(input_path), "--output", str(output_path), "--format", "json"]
    )

    assert result.exit_code == 0
    assert output_path.exists()


def test_export_with_metadata(runner, temp_dir, sample_shap_values):
    """Test export with metadata."""
    shap_values, X_test = sample_shap_values

    # Save SHAP values with metadata
    input_path = temp_dir / "shap.joblib"
    result_data = {
        "shap_values": shap_values,
        "feature_names": list(X_test.columns),
        "metadata": {"model_type": "RandomForest", "n_samples": len(X_test)},
    }
    joblib.dump(result_data, input_path)

    output_path = temp_dir / "output.csv"

    result = runner.invoke(
        cli,
        [
            "export",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-metadata",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    # Check metadata file
    metadata_path = output_path.with_suffix(".metadata.json")
    assert metadata_path.exists()


# ============================================================================
# Serve Command Tests
# ============================================================================


def test_serve_help(runner):
    """Test serve command help."""
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "FastAPI server" in result.output


@patch("shap_analytics.cli.uvicorn")
def test_serve_basic(mock_uvicorn, runner):
    """Test serve command (mocked)."""
    mock_uvicorn.run = Mock()

    result = runner.invoke(cli, ["serve", "--port", "8001", "--workers", "2"])

    # Verify uvicorn.run was called
    mock_uvicorn.run.assert_called_once()
    call_kwargs = mock_uvicorn.run.call_args[1]
    assert call_kwargs["port"] == 8001
    assert call_kwargs["workers"] == 2


# ============================================================================
# Completion Command Tests
# ============================================================================


def test_completion_help(runner):
    """Test completion command help."""
    result = runner.invoke(cli, ["completion", "--help"])
    assert result.exit_code == 0
    assert "shell completion" in result.output.lower()


def test_completion_bash(runner):
    """Test bash completion generation."""
    result = runner.invoke(cli, ["completion", "--shell", "bash"])
    assert result.exit_code == 0
    assert "bash" in result.output.lower()
    assert "_shap_analytics_completion" in result.output


def test_completion_zsh(runner):
    """Test zsh completion generation."""
    result = runner.invoke(cli, ["completion", "--shell", "zsh"])
    assert result.exit_code == 0
    assert "zsh" in result.output.lower() or "#compdef" in result.output


def test_completion_fish(runner):
    """Test fish completion generation."""
    result = runner.invoke(cli, ["completion", "--shell", "fish"])
    assert result.exit_code == 0
    assert "fish" in result.output.lower()


def test_completion_powershell(runner):
    """Test powershell completion generation."""
    result = runner.invoke(cli, ["completion", "--shell", "powershell"])
    assert result.exit_code == 0
    assert "powershell" in result.output.lower() or "Register-ArgumentCompleter" in result.output


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_file_not_found_error(runner, temp_dir):
    """Test file not found error handling."""
    result = runner.invoke(cli, ["validate", "--data", str(temp_dir / "nonexistent.csv")])

    assert result.exit_code != 0


def test_invalid_config_format(runner, temp_dir):
    """Test invalid config file format."""
    # Create invalid config
    config_path = temp_dir / "config.txt"
    config_path.write_text("invalid config")

    X = pd.DataFrame(np.random.randn(50, 3))
    data_path = temp_dir / "data.csv"
    X.to_csv(data_path, index=False)

    result = runner.invoke(cli, ["validate", "--data", str(data_path), "--config", str(config_path)])

    assert result.exit_code != 0


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_full_workflow(runner, temp_dir, sample_data, sample_model):
    """Test complete workflow: compute -> validate -> monitor -> export."""
    X, y = sample_data
    X_train = X[:50]
    X_test = X[50:70]
    X_new = X[70:]

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    new_path = temp_dir / "new.csv"
    shap_path = temp_dir / "shap.joblib"
    export_path = temp_dir / "results.csv"

    joblib.dump(sample_model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    X_new.to_csv(new_path, index=False)

    # Step 1: Validate
    result = runner.invoke(cli, ["validate", "--data", str(train_path)])
    assert result.exit_code == 0

    # Step 2: Compute SHAP values
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(shap_path),
        ],
    )
    assert result.exit_code == 0
    assert shap_path.exists()

    # Step 3: Monitor drift
    result = runner.invoke(
        cli, ["monitor", "--train-data", str(train_path), "--new-data", str(new_path)]
    )
    assert result.exit_code == 0

    # Step 4: Export results
    result = runner.invoke(cli, ["export", "--input", str(shap_path), "--output", str(export_path)])
    assert result.exit_code == 0
    assert export_path.exists()


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.slow
def test_compute_large_dataset(runner, temp_dir):
    """Test compute with larger dataset."""
    # Generate larger dataset
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    X_test = pd.DataFrame(np.random.randn(200, 20), columns=[f"f{i}" for i in range(20)])
    y_train = np.random.randint(0, 2, 1000)

    # Train model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save files
    model_path = temp_dir / "model.joblib"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    output_path = temp_dir / "shap.joblib"

    joblib.dump(model, model_path)
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)

    # Run command
    result = runner.invoke(
        cli,
        [
            "compute",
            "--model",
            str(model_path),
            "--train-data",
            str(train_path),
            "--test-data",
            str(test_path),
            "--output",
            str(output_path),
            "--background-size",
            "100",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
