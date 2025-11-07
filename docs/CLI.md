# SHAP Analytics CLI Documentation

## Overview

The SHAP Analytics CLI provides command-line access to SHAP value computation, validation, monitoring, serving, and export functionality. Built with the Click framework, it offers a Unix-philosophy-compliant interface for batch processing and automation.

## Installation

```bash
# Install with pip
pip install shap-analytics

# Install with poetry
poetry add shap-analytics

# Install from source
git clone https://github.com/diogoribeiro7/shap-analytics.git
cd shap-analytics
poetry install
```

## Quick Start

```bash
# Show help
shap-analytics --help

# Show version
shap-analytics --version

# Get help for a specific command
shap-analytics compute --help
```

## Commands

### 1. `compute` - Compute SHAP Values

Compute SHAP values for model predictions using background distribution from training data.

**Usage:**
```bash
shap-analytics compute \
  --model model.joblib \
  --train-data train.csv \
  --test-data test.csv \
  --output shap_values.joblib
```

**Options:**
- `--model, -m`: Path to trained model file (joblib format) [required]
- `--train-data, -t`: Path to training data file (for background distribution) [required]
- `--test-data, -d`: Path to test data file (to compute SHAP values for) [required]
- `--output, -o`: Output path for computed SHAP values [required]
- `--background-size, -b`: Number of background samples (default: 100)
- `--format, -f`: Input file format (csv, parquet, feather, json)
- `--config, -c`: Path to configuration file (JSON or YAML)
- `--verify`: Verify SHAP reconstruction after computation

**Examples:**

```bash
# Basic usage
shap-analytics compute \
  -m model.joblib \
  -t train.csv \
  -d test.csv \
  -o shap.joblib

# With custom background size
shap-analytics compute \
  -m model.joblib \
  -t train.csv \
  -d test.csv \
  -o shap.joblib \
  --background-size 200

# With verification
shap-analytics compute \
  -m model.joblib \
  -t train.csv \
  -d test.csv \
  -o shap.joblib \
  --verify

# With config file
shap-analytics compute \
  -m model.joblib \
  -t train.csv \
  -d test.csv \
  -o shap.joblib \
  --config config.json

# With parquet input
shap-analytics compute \
  -m model.joblib \
  -t train.parquet \
  -d test.parquet \
  -o shap.joblib \
  --format parquet
```

**Configuration File Example (config.json):**
```json
{
  "background_size": 200,
  "threshold": 0.15,
  "bins": 25
}
```

---

### 2. `validate` - Validate Background Samples

Validate that background samples are statistically representative of the full training data.

**Usage:**
```bash
shap-analytics validate --data train.csv
```

**Options:**
- `--data, -d`: Path to training data file [required]
- `--sample-size, -s`: Number of samples to use for validation (default: 100)
- `--threshold, -t`: Maximum allowed normalized difference (default: 0.1)
- `--format, -f`: Input file format (csv, parquet, feather, json)
- `--config, -c`: Path to configuration file

**Examples:**

```bash
# Basic validation
shap-analytics validate -d train.csv

# Custom sample size and threshold
shap-analytics validate \
  -d train.csv \
  --sample-size 200 \
  --threshold 0.15

# With config file
shap-analytics validate \
  -d train.csv \
  --config config.json
```

---

### 3. `monitor` - Monitor Feature Drift

Monitor feature drift between training and new data using Jensen-Shannon divergence.

**Usage:**
```bash
shap-analytics monitor \
  --train-data train.csv \
  --new-data new.csv
```

**Options:**
- `--train-data, -t`: Path to training data file (reference distribution) [required]
- `--new-data, -n`: Path to new data file (current distribution) [required]
- `--output, -o`: Output path for drift report (JSON format)
- `--threshold`: Drift threshold for alerting (default: 0.2)
- `--bins, -b`: Number of histogram bins (default: 20)
- `--format, -f`: Input file format
- `--config, -c`: Path to configuration file

**Examples:**

```bash
# Basic drift monitoring
shap-analytics monitor \
  -t train.csv \
  -n new.csv

# Save drift report
shap-analytics monitor \
  -t train.csv \
  -n new.csv \
  --output drift_report.json

# Custom threshold and bins
shap-analytics monitor \
  -t train.csv \
  -n new.csv \
  --threshold 0.3 \
  --bins 30

# With verbose output
shap-analytics --verbose monitor \
  -t train.csv \
  -n new.csv
```

**Drift Report Example:**
```json
{
  "summary": {
    "features_analyzed": 30,
    "high_drift_features": 3,
    "average_drift": 0.125,
    "threshold": 0.2,
    "bins": 20
  },
  "drift_scores": {
    "feature_1": 0.05,
    "feature_2": 0.35,
    "feature_3": 0.12
  },
  "high_drift_features": {
    "feature_2": 0.35
  },
  "timestamp": "2025-11-07 10:30:45"
}
```

---

### 4. `serve` - Start FastAPI Server

Start a production-ready FastAPI server for SHAP explanations.

**Usage:**
```bash
shap-analytics serve
```

**Options:**
- `--host, -h`: Host to bind the server to (default: 0.0.0.0)
- `--port, -p`: Port to bind the server to (default: 8000)
- `--workers, -w`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development
- `--log-level`: Logging level (debug, info, warning, error, critical)
- `--config, -c`: Path to configuration file

**Examples:**

```bash
# Start server on default port
shap-analytics serve

# Custom host and port
shap-analytics serve --host 127.0.0.1 --port 5000

# Development mode with auto-reload
shap-analytics serve --reload

# Production with multiple workers
shap-analytics serve --workers 4 --log-level info

# With config file
shap-analytics serve --config server_config.json
```

**API Endpoints:**
- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `GET /metrics` - API metrics and monitoring
- `POST /explain` - Generate SHAP explanations
- `GET /model/info` - Get model information
- `POST /explain/batch` - Batch explanations

**Testing the API:**
```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Explain prediction
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, ...]}'
```

---

### 5. `export` - Export SHAP Values

Export computed SHAP values to various formats (CSV, Parquet, JSON).

**Usage:**
```bash
shap-analytics export \
  --input shap_values.joblib \
  --output results.csv
```

**Options:**
- `--input, -i`: Path to input SHAP values file (joblib format) [required]
- `--output, -o`: Output path for exported results [required]
- `--format, -f`: Export format (csv, parquet, json) (default: csv)
- `--include-metadata`: Include metadata in export
- `--config, -c`: Path to configuration file

**Examples:**

```bash
# Export to CSV
shap-analytics export \
  -i shap.joblib \
  -o results.csv

# Export to Parquet
shap-analytics export \
  -i shap.joblib \
  -o results.parquet \
  --format parquet

# Export to JSON with metadata
shap-analytics export \
  -i shap.joblib \
  -o results.json \
  --format json \
  --include-metadata
```

---

### 6. `completion` - Shell Completion

Generate shell completion scripts for bash, zsh, fish, or PowerShell.

**Usage:**
```bash
shap-analytics completion --shell bash
```

**Options:**
- `--shell`: Shell to generate completion for (bash, zsh, fish, powershell) [required]
- `--install`: Install completion script (requires admin/sudo)

**Examples:**

```bash
# Generate bash completion
shap-analytics completion --shell bash

# Generate zsh completion
shap-analytics completion --shell zsh

# Generate fish completion
shap-analytics completion --shell fish

# Generate PowerShell completion
shap-analytics completion --shell powershell
```

**Manual Installation:**

**Bash:**
```bash
shap-analytics completion --shell bash > ~/.shap-analytics-complete.sh
echo 'source ~/.shap-analytics-complete.sh' >> ~/.bashrc
source ~/.bashrc
```

**Zsh:**
```bash
shap-analytics completion --shell zsh > ~/.shap-analytics-complete.zsh
echo 'source ~/.shap-analytics-complete.zsh' >> ~/.zshrc
source ~/.zshrc
```

**Fish:**
```bash
shap-analytics completion --shell fish > ~/.config/fish/completions/shap-analytics.fish
```

**PowerShell:**
```powershell
shap-analytics completion --shell powershell > shap-analytics.ps1
# Add to PowerShell profile
```

---

## Global Options

These options are available for all commands:

- `--verbose, -v`: Enable verbose output with detailed logging
- `--help`: Show help message
- `--version`: Show version information

**Examples:**
```bash
# Verbose mode
shap-analytics --verbose compute -m model.joblib -t train.csv -d test.csv -o shap.joblib

# Show help
shap-analytics --help
shap-analytics compute --help

# Show version
shap-analytics --version
```

---

## Configuration Files

The CLI supports configuration files in JSON or YAML format. Configuration files can contain default values for command options.

**Example config.json:**
```json
{
  "background_size": 200,
  "threshold": 0.15,
  "bins": 25,
  "host": "0.0.0.0",
  "port": 8000,
  "workers": 4,
  "log_level": "info",
  "format": "parquet"
}
```

**Example config.yaml:**
```yaml
background_size: 200
threshold: 0.15
bins: 25
host: 0.0.0.0
port: 8000
workers: 4
log_level: info
format: parquet
```

---

## File Formats

The CLI supports multiple file formats for input and output:

### Input Formats
- **CSV** (`.csv`) - Comma-separated values
- **Parquet** (`.parquet`) - Apache Parquet columnar format
- **Feather** (`.feather`) - Arrow-based binary format
- **JSON** (`.json`) - JavaScript Object Notation

### Model Formats
- **Joblib** (`.joblib`, `.pkl`) - Serialized scikit-learn models

### Output Formats
- **CSV** (`.csv`) - For SHAP value exports
- **Parquet** (`.parquet`) - For SHAP value exports
- **JSON** (`.json`) - For SHAP value exports and reports
- **Joblib** (`.joblib`) - For SHAP computation results

---

## Batch Processing Examples

### Pipeline Example 1: Full Workflow

```bash
#!/bin/bash
# Complete SHAP analysis pipeline

# Step 1: Validate training data
shap-analytics validate --data train.csv

# Step 2: Compute SHAP values
shap-analytics compute \
  --model model.joblib \
  --train-data train.csv \
  --test-data test.csv \
  --output shap_values.joblib \
  --verify

# Step 3: Monitor drift
shap-analytics monitor \
  --train-data train.csv \
  --new-data new_batch.csv \
  --output drift_report.json

# Step 4: Export results
shap-analytics export \
  --input shap_values.joblib \
  --output results.csv \
  --include-metadata

echo "Pipeline complete!"
```

### Pipeline Example 2: Automated Monitoring

```bash
#!/bin/bash
# Automated drift monitoring with alerting

THRESHOLD=0.3
TRAIN_DATA="reference_data.csv"
NEW_DATA="latest_data.csv"
REPORT="drift_report_$(date +%Y%m%d).json"

# Run drift monitoring
shap-analytics monitor \
  --train-data "$TRAIN_DATA" \
  --new-data "$NEW_DATA" \
  --threshold "$THRESHOLD" \
  --output "$REPORT"

# Check for high drift
if grep -q "high_drift_features" "$REPORT"; then
    echo "WARNING: High drift detected!"
    # Send alert (e.g., email, Slack, etc.)
fi
```

### Pipeline Example 3: Batch Export

```bash
#!/bin/bash
# Export SHAP values to multiple formats

INPUT="shap_values.joblib"

# Export to CSV
shap-analytics export -i "$INPUT" -o results.csv -f csv

# Export to Parquet
shap-analytics export -i "$INPUT" -o results.parquet -f parquet

# Export to JSON with metadata
shap-analytics export -i "$INPUT" -o results.json -f json --include-metadata

echo "Exported to multiple formats"
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: SHAP Analysis

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  shap-analysis:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install shap-analytics

      - name: Validate training data
        run: |
          shap-analytics validate --data data/train.csv

      - name: Compute SHAP values
        run: |
          shap-analytics compute \
            --model models/model.joblib \
            --train-data data/train.csv \
            --test-data data/test.csv \
            --output artifacts/shap_values.joblib

      - name: Monitor drift
        run: |
          shap-analytics monitor \
            --train-data data/train.csv \
            --new-data data/new.csv \
            --output artifacts/drift_report.json

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: shap-artifacts
          path: artifacts/
```

---

## Docker Integration

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install shap-analytics

# Copy data and models
COPY data/ /app/data/
COPY models/ /app/models/

# Run SHAP computation
CMD ["shap-analytics", "compute", \
     "--model", "models/model.joblib", \
     "--train-data", "data/train.csv", \
     "--test-data", "data/test.csv", \
     "--output", "data/shap_values.joblib"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  shap-compute:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    command: >
      shap-analytics compute
      --model /app/models/model.joblib
      --train-data /app/data/train.csv
      --test-data /app/data/test.csv
      --output /app/output/shap_values.joblib

  shap-server:
    build: .
    ports:
      - "8000:8000"
    command: >
      shap-analytics serve
      --host 0.0.0.0
      --port 8000
      --workers 4
```

---

## Error Handling

The CLI provides user-friendly error messages for common issues:

### File Not Found
```bash
$ shap-analytics validate --data missing.csv
Error: Data file not found: missing.csv
```

### Dimension Mismatch
```bash
$ shap-analytics compute -m model.joblib -t train.csv -d test.csv -o out.joblib
Error: Feature count mismatch: train=30, test=25
```

### Invalid Format
```bash
$ shap-analytics export -i shap.joblib -o out.txt -f txt
Error: Unsupported format: txt. Use csv, parquet, or json
```

### Verbose Mode
For detailed debugging, use `--verbose`:
```bash
$ shap-analytics --verbose compute -m model.joblib -t train.csv -d test.csv -o out.joblib
```

---

## Best Practices

1. **Use Configuration Files**: Store common settings in config files for consistency
2. **Enable Verification**: Use `--verify` flag for critical computations
3. **Monitor Drift**: Regularly check for data drift to maintain model reliability
4. **Batch Processing**: Process multiple files in scripts for automation
5. **Export Multiple Formats**: Export to different formats for various use cases
6. **Use Appropriate Background Size**: Balance accuracy and computation time
7. **Version Control**: Track configuration files and scripts in version control
8. **CI/CD Integration**: Automate SHAP analysis in your deployment pipeline

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure shap-analytics is properly installed:
```bash
pip install shap-analytics
# or
poetry install
```

### Issue: "Memory Error"
**Solution**: Reduce background size or process data in batches:
```bash
shap-analytics compute ... --background-size 50
```

### Issue: "Server Won't Start"
**Solution**: Check if port is already in use:
```bash
# Use different port
shap-analytics serve --port 8001

# Or kill process using the port
lsof -ti:8000 | xargs kill -9
```

---

## Performance Tips

1. **Optimize Background Size**: Start with 100 samples, increase if needed
2. **Use Parquet**: For large datasets, use Parquet format for faster I/O
3. **Parallel Processing**: Use multiple workers for the server
4. **Batch Operations**: Process multiple samples together
5. **Cache Models**: Reuse loaded models when possible

---

## Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Click Documentation](https://click.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GitHub Repository](https://github.com/diogoribeiro7/shap-analytics)
- [Issue Tracker](https://github.com/diogoribeiro7/shap-analytics/issues)

---

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/diogoribeiro7/shap-analytics/issues
- Documentation: https://diogoribeiro7.github.io/shap-analytics
- Email: dfr@esmad.ipp.pt
