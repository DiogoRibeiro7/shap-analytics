"""
Command-line Interface for SHAP Analytics.

This module provides a comprehensive CLI for SHAP value computation,
validation, monitoring, serving, and exporting using the Click framework.

Commands:
    - compute: Compute SHAP values from file
    - validate: Validate background samples
    - monitor: Run drift monitoring
    - serve: Start FastAPI server
    - export: Export results to various formats

Usage:
    shap-analytics compute --model model.joblib --data data.csv
    shap-analytics validate --data data.csv --sample-size 100
    shap-analytics monitor --train train.csv --new new.csv
    shap-analytics serve --host 0.0.0.0 --port 8000
    shap-analytics export --input shap_values.joblib --output results.csv

Enable shell completion:
    # Bash
    eval "$(_SHAP_ANALYTICS_COMPLETE=bash_source shap-analytics)"

    # Zsh
    eval "$(_SHAP_ANALYTICS_COMPLETE=zsh_source shap-analytics)"

    # Fish
    eval (env _SHAP_ANALYTICS_COMPLETE=fish_source shap-analytics)
"""

import json
import sys
import time

from pathlib import Path
from typing import Any, Optional

import click
import joblib
import numpy as np
import pandas as pd

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .shap_explain import (
    compute_shap_values,
    monitor_feature_drift,
    validate_background_sample,
    verify_shap_reconstruction,
)
from .utils.io_utils import export_shap_report, load_compressed, save_compressed
from .utils.logging_utils import setup_structured_logger

# Setup rich console for beautiful output
console = Console()
logger = setup_structured_logger(__name__, enable_json=False)

# Version info
__version__ = "0.1.0"


# ============================================================================
# Helper Functions
# ============================================================================


def load_config(config_path: Optional[str]) -> dict[str, Any]:
    """
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise click.FileError(str(config_file), "Configuration file not found")

    try:
        with open(config_file) as f:
            if config_file.suffix in {".yaml", ".yml"}:
                try:
                    import yaml

                    return yaml.safe_load(f)
                except ImportError:
                    raise click.ClickException(
                        "YAML support requires PyYAML. Install with: pip install pyyaml"
                    )
            elif config_file.suffix == ".json":
                return json.load(f)
            else:
                raise click.ClickException(
                    f"Unsupported config format: {config_file.suffix}. Use .json or .yaml"
                )
    except Exception as e:
        raise click.ClickException(f"Failed to load config: {e}")


def load_data_file(
    file_path: str,
    file_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from various file formats.

    Args:
        file_path: Path to data file.
        file_format: File format override (csv, parquet, feather).

    Returns:
        DataFrame with loaded data.

    Raises:
        click.ClickException: If file cannot be loaded.
    """
    path = Path(file_path)
    if not path.exists():
        raise click.FileError(str(path), "Data file not found")

    # Determine format
    fmt = file_format or path.suffix[1:].lower()

    try:
        with console.status(f"[bold blue]Loading {path.name}..."):
            if fmt == "csv":
                df = pd.read_csv(path)
            elif fmt == "parquet":
                df = pd.read_parquet(path)
            elif fmt == "feather":
                df = pd.read_feather(path)
            elif fmt == "json":
                df = pd.read_json(path)
            else:
                raise click.ClickException(
                    f"Unsupported format: {fmt}. Use csv, parquet, feather, or json"
                )

        console.print(f"[green]✓[/green] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        raise click.ClickException(f"Failed to load data: {e}")


def load_model_file(file_path: str) -> Any:
    """
    Load model from joblib file.

    Args:
        file_path: Path to model file.

    Returns:
        Loaded model object.

    Raises:
        click.ClickException: If model cannot be loaded.
    """
    path = Path(file_path)
    if not path.exists():
        raise click.FileError(str(path), "Model file not found")

    try:
        with console.status(f"[bold blue]Loading model from {path.name}..."):
            model = joblib.load(path)
        console.print(f"[green]✓[/green] Model loaded: {type(model).__name__}")
        return model
    except Exception as e:
        raise click.ClickException(f"Failed to load model: {e}")


def handle_error(e: Exception, verbose: bool = False) -> None:
    """
    Handle errors with user-friendly messages.

    Args:
        e: Exception to handle.
        verbose: Whether to show full traceback.
    """
    console.print(f"[bold red]Error:[/bold red] {str(e)}")

    if verbose:
        import traceback

        console.print("\n[yellow]Traceback:[/yellow]")
        console.print(traceback.format_exc())

    sys.exit(1)


# ============================================================================
# CLI Group
# ============================================================================


@click.group()
@click.version_option(version=__version__, prog_name="shap-analytics")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output with detailed logging.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    SHAP Analytics CLI - Professional SHAP value computation and analysis toolkit.

    This CLI provides commands for computing SHAP values, validating data,
    monitoring drift, serving APIs, and exporting results.

    Examples:

        # Compute SHAP values
        shap-analytics compute --model model.joblib --data test.csv

        # Validate background samples
        shap-analytics validate --data train.csv --sample-size 100

        # Monitor feature drift
        shap-analytics monitor --train train.csv --new new.csv

        # Start API server
        shap-analytics serve --port 8000

        # Export results
        shap-analytics export --input shap.joblib --output results.csv

    For detailed help on any command, use:
        shap-analytics COMMAND --help
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


# ============================================================================
# Compute Command
# ============================================================================


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model file (joblib format).",
)
@click.option(
    "--train-data",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data file (for background distribution).",
)
@click.option(
    "--test-data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to test data file (to compute SHAP values for).",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for computed SHAP values (joblib format).",
)
@click.option(
    "--background-size",
    "-b",
    default=100,
    type=int,
    help="Number of background samples for SHAP computation.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "parquet", "feather", "json"], case_sensitive=False),
    help="Input file format override.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML).",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Verify SHAP reconstruction after computation.",
)
@click.pass_context
def compute(
    ctx: click.Context,
    model: str,
    train_data: str,
    test_data: str,
    output: str,
    background_size: int,
    format: Optional[str],
    config: Optional[str],
    verify: bool,
) -> None:
    """
    Compute SHAP values for model predictions.

    This command loads a trained model and computes SHAP values for test data
    using a background distribution sampled from training data.

    Examples:

        # Basic usage
        shap-analytics compute -m model.joblib -t train.csv -d test.csv -o shap.joblib

        # With custom background size
        shap-analytics compute -m model.joblib -t train.csv -d test.csv -o shap.joblib -b 200

        # With verification
        shap-analytics compute -m model.joblib -t train.csv -d test.csv -o shap.joblib --verify

        # With config file
        shap-analytics compute -m model.joblib -t train.csv -d test.csv -o shap.joblib -c config.json
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        cfg = load_config(config)
        background_size = cfg.get("background_size", background_size)

        console.rule("[bold blue]SHAP Value Computation")

        # Load model
        model_obj = load_model_file(model)

        # Load data
        X_train = load_data_file(train_data, format)
        X_test = load_data_file(test_data, format)

        # Validate dimensions
        if X_train.shape[1] != X_test.shape[1]:
            raise click.ClickException(
                f"Feature count mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
            )

        # Compute SHAP values with progress
        console.print(f"\n[bold]Computing SHAP values (background_size={background_size})...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing SHAP values...", total=100)

            start_time = time.time()
            shap_values = compute_shap_values(
                model_obj, X_train, X_test, background_size=background_size
            )
            progress.update(task, advance=100)

            elapsed = time.time() - start_time

        console.print(f"[green]✓[/green] SHAP values computed in {elapsed:.2f}s")

        # Verify reconstruction if requested
        if verify:
            console.print("\n[bold]Verifying SHAP reconstruction...[/bold]")
            is_valid = verify_shap_reconstruction(shap_values, X_test, model_obj)
            if is_valid:
                console.print("[green]✓[/green] SHAP reconstruction verified")
            else:
                console.print("[yellow]⚠[/yellow] SHAP reconstruction check failed")

        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with console.status(f"[bold blue]Saving results to {output_path.name}..."):
            result = {
                "shap_values": shap_values,
                "feature_names": list(X_test.columns),
                "test_data": X_test,
                "metadata": {
                    "model_type": type(model_obj).__name__,
                    "background_size": background_size,
                    "n_samples": len(X_test),
                    "n_features": X_test.shape[1],
                    "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
            joblib.dump(result, output_path)

        console.print(f"[green]✓[/green] Results saved to {output_path}")

        # Summary table
        table = Table(title="Computation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model Type", type(model_obj).__name__)
        table.add_row("Samples", str(len(X_test)))
        table.add_row("Features", str(X_test.shape[1]))
        table.add_row("Background Size", str(background_size))
        table.add_row("Computation Time", f"{elapsed:.2f}s")
        table.add_row("Output File", str(output_path))

        console.print("\n")
        console.print(table)

    except Exception as e:
        handle_error(e, verbose)


# ============================================================================
# Validate Command
# ============================================================================


@cli.command()
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data file.",
)
@click.option(
    "--sample-size",
    "-s",
    default=100,
    type=int,
    help="Number of samples to use for validation.",
)
@click.option(
    "--threshold",
    "-t",
    default=0.1,
    type=float,
    help="Maximum allowed normalized difference threshold.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "parquet", "feather", "json"], case_sensitive=False),
    help="Input file format override.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML).",
)
@click.pass_context
def validate(
    ctx: click.Context,
    data: str,
    sample_size: int,
    threshold: float,
    format: Optional[str],
    config: Optional[str],
) -> None:
    """
    Validate background sample representativeness.

    This command checks if a background sample is statistically representative
    of the full training data by comparing means and standard deviations.

    Examples:

        # Basic validation
        shap-analytics validate -d train.csv

        # Custom sample size and threshold
        shap-analytics validate -d train.csv -s 200 -t 0.15

        # With config file
        shap-analytics validate -d train.csv -c config.json
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        cfg = load_config(config)
        sample_size = cfg.get("sample_size", sample_size)
        threshold = cfg.get("threshold", threshold)

        console.rule("[bold blue]Background Sample Validation")

        # Load data
        X_train = load_data_file(data, format)

        # Validate
        console.print(f"\n[bold]Validating background sample...[/bold]")
        console.print(f"Sample size: {sample_size}")
        console.print(f"Threshold: {threshold}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating sample...", total=None)

            is_valid = validate_background_sample(
                X_train, sample_size=sample_size, threshold=threshold
            )

            progress.update(task, completed=100)

        if is_valid:
            console.print("\n[green]✓[/green] Background sample is valid and representative")
        else:
            console.print(
                "\n[yellow]⚠[/yellow] Background sample validation failed - "
                "consider increasing sample size"
            )
            sys.exit(1)

    except Exception as e:
        handle_error(e, verbose)


# ============================================================================
# Monitor Command
# ============================================================================


@cli.command()
@click.option(
    "--train-data",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data file (reference distribution).",
)
@click.option(
    "--new-data",
    "-n",
    required=True,
    type=click.Path(exists=True),
    help="Path to new data file (current distribution).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for drift report (JSON format).",
)
@click.option(
    "--threshold",
    "-t",
    default=0.2,
    type=float,
    help="Drift threshold for alerting (0-1 scale).",
)
@click.option(
    "--bins",
    "-b",
    default=20,
    type=int,
    help="Number of histogram bins for distribution estimation.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "parquet", "feather", "json"], case_sensitive=False),
    help="Input file format override.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML).",
)
@click.pass_context
def monitor(
    ctx: click.Context,
    train_data: str,
    new_data: str,
    output: Optional[str],
    threshold: float,
    bins: int,
    format: Optional[str],
    config: Optional[str],
) -> None:
    """
    Monitor feature drift between training and new data.

    This command uses Jensen-Shannon divergence to detect distribution shifts
    that might affect SHAP value reliability.

    Examples:

        # Basic drift monitoring
        shap-analytics monitor -t train.csv -n new.csv

        # Save drift report
        shap-analytics monitor -t train.csv -n new.csv -o drift_report.json

        # Custom threshold and bins
        shap-analytics monitor -t train.csv -n new.csv --threshold 0.3 --bins 30
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        cfg = load_config(config)
        threshold = cfg.get("threshold", threshold)
        bins = cfg.get("bins", bins)

        console.rule("[bold blue]Feature Drift Monitoring")

        # Load data
        X_train = load_data_file(train_data, format)
        X_new = load_data_file(new_data, format)

        # Validate dimensions
        if X_train.shape[1] != X_new.shape[1]:
            raise click.ClickException(
                f"Feature count mismatch: train={X_train.shape[1]}, new={X_new.shape[1]}"
            )

        # Monitor drift
        console.print(f"\n[bold]Monitoring drift (threshold={threshold}, bins={bins})...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing drift scores...", total=len(X_train.columns))

            drift_scores = monitor_feature_drift(X_train, X_new, threshold=threshold, bins=bins)

            progress.update(task, completed=len(X_train.columns))

        # Analyze results
        high_drift = {k: v for k, v in drift_scores.items() if v > threshold}
        avg_drift = np.mean(list(drift_scores.values()))

        # Display results
        console.print("\n[bold]Drift Analysis:[/bold]")
        console.print(f"Features analyzed: {len(drift_scores)}")
        console.print(f"High drift features: {len(high_drift)}")
        console.print(f"Average drift score: {avg_drift:.4f}")

        if high_drift:
            console.print("\n[yellow]⚠ High drift detected in features:[/yellow]")
            table = Table(show_header=True)
            table.add_column("Feature", style="cyan")
            table.add_column("Drift Score", style="red")

            for feature, score in sorted(high_drift.items(), key=lambda x: x[1], reverse=True):
                table.add_row(feature, f"{score:.4f}")

            console.print(table)
        else:
            console.print("\n[green]✓[/green] No significant drift detected")

        # Save report if requested
        if output:
            report = {
                "summary": {
                    "features_analyzed": len(drift_scores),
                    "high_drift_features": len(high_drift),
                    "average_drift": float(avg_drift),
                    "threshold": threshold,
                    "bins": bins,
                },
                "drift_scores": {k: float(v) for k, v in drift_scores.items()},
                "high_drift_features": {k: float(v) for k, v in high_drift.items()},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            console.print(f"\n[green]✓[/green] Drift report saved to {output_path}")

    except Exception as e:
        handle_error(e, verbose)


# ============================================================================
# Serve Command
# ============================================================================


@cli.command()
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    type=int,
    help="Port to bind the server to.",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="Number of worker processes.",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default="info",
    help="Logging level.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML).",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    workers: int,
    reload: bool,
    log_level: str,
    config: Optional[str],
) -> None:
    """
    Start FastAPI server for SHAP explanations.

    This command launches a production-ready REST API server for serving
    SHAP explanations with health checks, metrics, and monitoring.

    Examples:

        # Start server on default port
        shap-analytics serve

        # Custom host and port
        shap-analytics serve --host 127.0.0.1 --port 5000

        # Development mode with auto-reload
        shap-analytics serve --reload

        # Production with multiple workers
        shap-analytics serve --workers 4
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        cfg = load_config(config)
        host = cfg.get("host", host)
        port = cfg.get("port", port)
        workers = cfg.get("workers", workers)
        log_level = cfg.get("log_level", log_level)

        console.rule("[bold blue]Starting SHAP Analytics API Server")

        console.print(f"\n[bold]Server Configuration:[/bold]")
        console.print(f"Host: {host}")
        console.print(f"Port: {port}")
        console.print(f"Workers: {workers}")
        console.print(f"Log Level: {log_level}")
        console.print(f"Reload: {reload}")

        console.print(
            f"\n[green]✓[/green] Server will be available at: [link]http://{host}:{port}[/link]"
        )
        console.print("Press Ctrl+C to stop\n")

        # Import and run uvicorn
        try:
            import uvicorn
        except ImportError:
            raise click.ClickException(
                "uvicorn is required to run the server. Install with: pip install uvicorn"
            )

        # Run server
        uvicorn.run(
            "shap_analytics.examples.api_deployment:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=log_level,
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        handle_error(e, verbose)


# ============================================================================
# Export Command
# ============================================================================


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to input SHAP values file (joblib format).",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for exported results.",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "parquet", "json"], case_sensitive=False),
    default="csv",
    help="Export format.",
)
@click.option(
    "--include-metadata",
    is_flag=True,
    help="Include metadata in export.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON or YAML).",
)
@click.pass_context
def export(
    ctx: click.Context,
    input_file: str,
    output: str,
    format: str,
    include_metadata: bool,
    config: Optional[str],
) -> None:
    """
    Export SHAP values to various formats.

    This command exports computed SHAP values to CSV, Parquet, or JSON formats
    for analysis, visualization, or integration with other tools.

    Examples:

        # Export to CSV
        shap-analytics export -i shap.joblib -o results.csv

        # Export to Parquet
        shap-analytics export -i shap.joblib -o results.parquet -f parquet

        # Export with metadata
        shap-analytics export -i shap.joblib -o results.json -f json --include-metadata
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load configuration
        cfg = load_config(config)
        format = cfg.get("format", format)

        console.rule("[bold blue]Exporting SHAP Values")

        # Load SHAP values
        with console.status(f"[bold blue]Loading SHAP values..."):
            result = joblib.load(input_file)

        shap_values = result["shap_values"]
        feature_names = result.get("feature_names", [])
        metadata = result.get("metadata", {})

        console.print(f"[green]✓[/green] Loaded SHAP values")
        console.print(f"Samples: {len(shap_values)}")
        console.print(f"Features: {len(feature_names)}")

        # Export
        console.print(f"\n[bold]Exporting to {format.upper()} format...[/bold]")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with console.status(f"[bold blue]Writing to {output_path.name}..."):
            exported_path = export_shap_report(
                shap_values, feature_names, output_path, format=format
            )

        console.print(f"[green]✓[/green] SHAP values exported to {exported_path}")

        # Export metadata if requested
        if include_metadata and metadata:
            metadata_path = output_path.with_suffix(".metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            console.print(f"[green]✓[/green] Metadata exported to {metadata_path}")

        # Summary
        console.print(f"\n[bold green]Export complete![/bold green]")

    except Exception as e:
        handle_error(e, verbose)


# ============================================================================
# Completion Command
# ============================================================================


@cli.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish", "powershell"], case_sensitive=False),
    required=True,
    help="Shell to generate completion for.",
)
@click.option(
    "--install",
    is_flag=True,
    help="Install completion script (requires admin/sudo).",
)
def completion(shell: str, install: bool) -> None:
    """
    Generate shell completion scripts.

    This command generates shell completion scripts for bash, zsh, fish, or powershell.
    Use --install flag to automatically install the completion script.

    Examples:

        # Generate bash completion
        shap-analytics completion --shell bash

        # Install bash completion
        shap-analytics completion --shell bash --install

        # Generate zsh completion
        shap-analytics completion --shell zsh

    Manual Installation:

        Bash:
            shap-analytics completion --shell bash > ~/.shap-analytics-complete.sh
            echo 'source ~/.shap-analytics-complete.sh' >> ~/.bashrc

        Zsh:
            shap-analytics completion --shell zsh > ~/.shap-analytics-complete.zsh
            echo 'source ~/.shap-analytics-complete.zsh' >> ~/.zshrc

        Fish:
            shap-analytics completion --shell fish > ~/.config/fish/completions/shap-analytics.fish

        PowerShell:
            shap-analytics completion --shell powershell > shap-analytics.ps1
            # Add to PowerShell profile
    """
    shell_lower = shell.lower()

    # Generate completion script
    if shell_lower == "bash":
        script = """
# Bash completion for shap-analytics
_shap_analytics_completion() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _SHAP_ANALYTICS_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

complete -F _shap_analytics_completion -o default shap-analytics
"""
    elif shell_lower == "zsh":
        script = """
#compdef shap-analytics

_shap_analytics_completion() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[shap-analytics] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) _SHAP_ANALYTICS_COMPLETE=zsh_complete shap-analytics)}")

    for type key descr in ${response}; do
        if [[ "$type" == "plain" ]]; then
            if [[ "$descr" == "_" ]]; then
                completions+=("$key")
            else
                completions_with_descriptions+=("$key":"$descr")
            fi
        elif [[ "$type" == "dir" ]]; then
            _path_files -/
        elif [[ "$type" == "file" ]]; then
            _path_files -f
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

compdef _shap_analytics_completion shap-analytics
"""
    elif shell_lower == "fish":
        script = """
# Fish completion for shap-analytics
function __shap_analytics_completion
    set -l response (env _SHAP_ANALYTICS_COMPLETE=fish_complete COMP_WORDS=(commandline -cp) COMP_CWORD=(commandline -t) shap-analytics)

    for completion in $response
        set -l metadata (string split "," $completion)

        if test $metadata[1] = "dir"
            __fish_complete_directories $metadata[2]
        else if test $metadata[1] = "file"
            __fish_complete_path $metadata[2]
        else if test $metadata[1] = "plain"
            echo $metadata[2]
        end
    end
end

complete --no-files --command shap-analytics --arguments '(__shap_analytics_completion)'
"""
    elif shell_lower == "powershell":
        script = """
# PowerShell completion for shap-analytics
Register-ArgumentCompleter -Native -CommandName shap-analytics -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $env:_SHAP_ANALYTICS_COMPLETE = "powershell_complete"
    $env:COMP_WORDS = $commandAst.ToString()
    $env:COMP_CWORD = $cursorPosition
    shap-analytics | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
    Remove-Item Env:\\_SHAP_ANALYTICS_COMPLETE
    Remove-Item Env:\\COMP_WORDS
    Remove-Item Env:\\COMP_CWORD
}
"""
    else:
        console.print(f"[red]Unsupported shell: {shell}[/red]")
        sys.exit(1)

    if install:
        console.print(f"[yellow]Auto-installation not yet supported.[/yellow]")
        console.print(f"Please manually install the completion script using the instructions above.")
        console.print(f"\nGenerated script:\n")

    console.print(script)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
