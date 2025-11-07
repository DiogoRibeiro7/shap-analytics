# SHAP Analytics

**Professional SHAP value computation, analysis, and deployment toolkit**

[![CI Status](https://github.com/diogoribeiro7/shap-analytics/workflows/CI/badge.svg)](https://github.com/diogoribeiro7/shap-analytics/actions)
[![codecov](https://codecov.io/gh/diogoribeiro7/shap-analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/diogoribeiro7/shap-analytics)
[![PyPI version](https://badge.fury.io/py/shap-analytics.svg)](https://badge.fury.io/py/shap-analytics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://diogoribeiro7.github.io/shap-analytics)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## Overview

SHAP Analytics is a comprehensive Python library for computing, analyzing, visualizing, and deploying SHAP (SHapley Additive exPlanations) values in production machine learning systems. It provides robust tools for feature importance analysis, drift monitoring, background sample validation, and automated explanation generation with enterprise-grade quality controls.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Key Features

###  Core Capabilities

- **SHAP Computation**: Efficient SHAP value calculation with TreeExplainer support
- **Validation Tools**: Background sample validation and SHAP reconstruction verification
- **Drift Monitoring**: Feature drift detection using Jensen-Shannon divergence
- **Interactive Dashboards**: Create rich visualizations with Plotly integration
- **REST API**: FastAPI-based service for SHAP computation and serving
- **Registry System**: Track and manage SHAP explanations with metadata
- **MLflow Integration**: Experiment tracking and model versioning
- **Type Safety**: Fully typed codebase with mypy strict mode
- **Production Ready**: Docker support, structured logging, comprehensive testing

### Module Overview

#### ­ƒö¼ shap_explain.py - Core SHAP Computation
- `compute_shap_values()` - TreeExplainer with background sampling
- `validate_background_sample()` - Statistical validation
- `monitor_feature_drift()` - Distribution shift detection
- `verify_shap_reconstruction()` - Probability reconstruction check

#### ­ƒôè shap_expansion.py - Extensions & Deployment
- Interactive Plotly dashboards
- Model registry with versioning
- Drift detection and alerting
- FastAPI scaffolding
- CI/CD integration helpers

#### ­ƒÜÇ shap_future.py - Experimental Features
- Adaptive model retraining
- SHAP caching with TTL
- Performance benchmarking
- Advanced visualizations

## Installation

### Requirements

**Important**: This project requires **NumPy < 2.0.0** for compatibility with SHAP 0.45.0. NumPy 2.x introduced breaking API changes that are not yet supported by the SHAP library.

Minimum requirements:
- Python >= 3.10
- NumPy >= 1.24.0, < 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- SHAP >= 0.43.0

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/diogoribeiro7/shap-analytics.git
cd shap-analytics

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Using pip

```bash
pip install shap-analytics

# Or from source
pip install git+https://github.com/diogoribeiro7/shap-analytics.git
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access API at http://localhost:8080
curl http://localhost:8080/health
```

## Quick Start

### Basic Usage

```python
from shap_analytics import compute_shap_values, validate_background_sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate background sample
is_valid = validate_background_sample(X_train, sample_size=100)

# Compute SHAP values
shap_values = compute_shap_values(model, X_train, X_test)

# Access SHAP values for class 1
shap_values_class1 = shap_values[..., 1]
print(f"SHAP values shape: {shap_values_class1.shape}")
```

### API Service

```python
from shap_analytics import serve_shap_api
import uvicorn

# Create FastAPI app
app = serve_shap_api(model, X_train, background_size=100)

# Run server
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### CLI Usage

```bash
# Run example
shap-analytics

# Or directly
python -m shap_analytics.shap_explain
```

## Documentation

Full documentation is available at [https://diogoribeiro7.github.io/shap-analytics](https://diogoribeiro7.github.io/shap-analytics)

- [Usage Guide](https://diogoribeiro7.github.io/shap-analytics/usage/)
- [Architecture](https://diogoribeiro7.github.io/shap-analytics/architecture/)
- [API Reference](https://diogoribeiro7.github.io/shap-analytics/api_reference/)
- [Future Work](https://diogoribeiro7.github.io/shap-analytics/future_work/)

### Build Documentation Locally

```bash
poetry run mkdocs serve
# Open http://localhost:8000
```

## Project Structure

```
shap_analytics/
Ôö£ÔöÇÔöÇ src/shap_analytics/
Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôö£ÔöÇÔöÇ shap_explain.py          # Core SHAP computation
Ôöé   Ôö£ÔöÇÔöÇ shap_expansion.py        # Visualization & extensions
Ôöé   Ôö£ÔöÇÔöÇ shap_future.py           # Experimental features
Ôöé   Ôö£ÔöÇÔöÇ shap_backlog.py          # Implementation roadmap
Ôöé   ÔööÔöÇÔöÇ utils/
Ôöé       Ôö£ÔöÇÔöÇ __init__.py
Ôöé       Ôö£ÔöÇÔöÇ common.py            # Shared utilities
Ôöé       Ôö£ÔöÇÔöÇ io_utils.py          # I/O operations
Ôöé       ÔööÔöÇÔöÇ logging_utils.py     # Structured logging
Ôö£ÔöÇÔöÇ tests/
Ôöé   Ôö£ÔöÇÔöÇ conftest.py              # Pytest fixtures
Ôöé   Ôö£ÔöÇÔöÇ test_shap_explain.py     # Core tests
Ôöé   Ôö£ÔöÇÔöÇ test_shap_expansion.py   # Extension tests
Ôöé   Ôö£ÔöÇÔöÇ test_shap_future.py      # Experimental tests
Ôöé   ÔööÔöÇÔöÇ test_utils.py            # Utility tests
Ôö£ÔöÇÔöÇ docs/
Ôöé   Ôö£ÔöÇÔöÇ index.md
Ôöé   Ôö£ÔöÇÔöÇ usage.md
Ôöé   Ôö£ÔöÇÔöÇ architecture.md
Ôöé   Ôö£ÔöÇÔöÇ api_reference.md
Ôöé   ÔööÔöÇÔöÇ future_work.md
Ôö£ÔöÇÔöÇ .github/workflows/
Ôöé   Ôö£ÔöÇÔöÇ ci.yml                   # CI/CD pipeline
Ôöé   Ôö£ÔöÇÔöÇ todo.yml                 # TODO to GitHub Issues
Ôöé   ÔööÔöÇÔöÇ docs.yml                 # Documentation deployment
Ôö£ÔöÇÔöÇ pyproject.toml               # Poetry configuration
Ôö£ÔöÇÔöÇ Dockerfile                   # Multi-stage Docker build
Ôö£ÔöÇÔöÇ compose.yaml                 # Docker Compose configuration
Ôö£ÔöÇÔöÇ mkdocs.yml                   # Documentation configuration
Ôö£ÔöÇÔöÇ setup.sh                     # Setup script
Ôö£ÔöÇÔöÇ .pre-commit-config.yaml      # Pre-commit hooks
Ôö£ÔöÇÔöÇ .mypy.ini                    # MyPy configuration
Ôö£ÔöÇÔöÇ README.md
Ôö£ÔöÇÔöÇ LICENSE
Ôö£ÔöÇÔöÇ CONTRIBUTING.md
Ôö£ÔöÇÔöÇ CODE_OF_CONDUCT.md
Ôö£ÔöÇÔöÇ CHANGELOG.md
ÔööÔöÇÔöÇ CITATION.cff
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/diogoribeiro7/shap-analytics.git
cd shap-analytics

# Run setup script
./setup.sh

# Or manually:
poetry install
pre-commit install
```

### Quality Checks

```bash
# Type checking
poetry run mypy --strict src/

# Linting
poetry run ruff check src/

# Code formatting
poetry run ruff format src/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

### Running the Application

```bash
# Activate virtual environment
poetry shell

# Run example
python -m shap_analytics.shap_explain

# Start API server
uvicorn shap_analytics.shap_expansion:app --reload

# View documentation
mkdocs serve
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=src/shap_analytics --cov-report=html

# Run specific test file
poetry run pytest tests/test_shap_explain.py -v

# Run tests in parallel
poetry run pytest -n auto
```

### Test Coverage

The test suite includes **140+ test cases** with 80%+ coverage:

| Module | Coverage | Tests |
|--------|----------|-------|
| shap_explain.py | 90% | 40+ |
| shap_expansion.py | 85% | 50+ |
| shap_future.py | 82% | 30+ |
| utils/ | 92% | 20+ |

### Test Categories

```bash
# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Skip slow tests
poetry run pytest -m "not slow"
```

## Docker

### Build and Run

```bash
# Build image
docker build -t shap-analytics:latest .

# Run container
docker run -p 8080:8080 shap-analytics:latest

# Using Docker Compose (recommended)
docker-compose up -d

# View logs
docker-compose logs -f shap-api

# Stop services
docker-compose down
```

### Docker Services

The `compose.yaml` includes:

- **shap-api**: FastAPI service on port 8080
- **redis**: Cache server on port 6379
- **mlflow**: Tracking server on port 5000

### Environment Variables

See `.env.example` for configuration options.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed
- Use conventional commits for commit messages

## Roadmap

See [Future Work](docs/future_work.md) for planned features and enhancements.

**Upcoming Features:**

- Async SHAP computation
- Redis caching integration
- XGBoost/LightGBM support
- GPU acceleration
- Distributed computing with Dask
- Kubernetes deployment

## Citation

If you use SHAP Analytics in your research, please cite:

```bibtex
@software{shap_analytics_2025,
  title = {SHAP Analytics: Professional SHAP Value Computation and Analysis},
  author = {SHAP Analytics Contributors},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/diogoribeiro7/shap-analytics}
}
```

See [CITATION.cff](CITATION.cff) for more details.

## Acknowledgments

Built on top of the excellent [SHAP library](https://github.com/slundberg/shap) by Scott Lundberg.

## Support

- **Documentation**: [https://diogoribeiro7.github.io/shap-analytics](https://diogoribeiro7.github.io/shap-analytics)
- **Issues**: [GitHub Issues](https://github.com/diogoribeiro7/shap-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diogoribeiro7/shap-analytics/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
