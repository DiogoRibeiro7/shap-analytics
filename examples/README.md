# SHAP Analytics Examples

This directory contains practical, production-ready examples demonstrating various use cases of SHAP (SHapley Additive exPlanations) for ML explainability in enterprise environments.

## What's New ✨

### Recent Enhancements

**New Visualization Types** (custom_visualization.py):
- 🎨 **Decision Plots** - Visualize cumulative SHAP contributions across multiple samples
- 🎨 **Force Plots** - Interactive HTML force plots showing feature push/pull effects
- 🎨 **Interaction Plots** - 2D scatter plots for analyzing feature interactions
- 🎨 **Partial Dependence Plots** - Marginal effects with data distribution rug plots
- 🎨 **Violin Plots** - Statistical distributions with box plots for SHAP values

**API Improvements** (api_deployment.py):
- 📊 **Metrics Endpoint** - `/metrics` endpoint for monitoring request counts, success rates, and latency
- 📊 **Automatic Request Tracking** - Built-in middleware tracks all requests and performance
- 📊 **Observability Ready** - Metrics formatted for integration with monitoring tools

**Reliability Improvements** (basic_usage.py):
- 🛡️ **Robust Backend Handling** - Works in headless/server environments without display
- 🛡️ **Graceful Degradation** - Continues running even if matplotlib unavailable
- 🛡️ **Cross-Platform Paths** - Uses pathlib for Windows/Linux/Mac compatibility

## Overview

Each example is designed to be:
- **Production-ready** - Includes error handling, logging, and best practices
- **Well-documented** - Comprehensive docstrings and inline comments
- **Runnable** - Minimal setup required, works with common datasets
- **Educational** - Demonstrates enterprise-grade patterns and techniques

## Examples

### 1. Basic Usage (`basic_usage.py`)

**Purpose**: Introduction to fundamental SHAP concepts and workflows.

**Features**:
- Data loading and validation with sklearn datasets
- Model training with Random Forest
- SHAP value computation using TreeExplainer
- Feature importance analysis
- Basic visualizations (summary, bar, waterfall plots)
- **IMPROVED:** Robust matplotlib backend handling (works in server environments)
- **IMPROVED:** Better error handling with graceful degradation
- **IMPROVED:** Path management with pathlib for cross-platform compatibility
- Performance considerations for large datasets

**When to use**:
- Learning SHAP basics
- Quick model explanations
- Feature importance analysis
- Educational purposes

**Usage**:
```bash
python examples/basic_usage.py
```

**Output**:
- Console logs with feature importance rankings
- PNG files: `shap_summary_plot.png`, `shap_bar_plot.png`, `shap_waterfall_plot.png`

**Key Concepts**:
- TreeExplainer for fast tree-based model explanations
- Background sampling for performance optimization
- Global feature importance vs. local explanations

---

### 2. API Deployment (`api_deployment.py`)

**Purpose**: Deploy SHAP explanations as a production REST API using FastAPI.

**Features**:
- RESTful API with FastAPI framework
- Request/response validation with Pydantic
- Model loading and caching
- Health checks and monitoring endpoints
- **NEW: Metrics endpoint** for observability (request counts, success rates, latency)
- Batch prediction support
- Docker-ready configuration
- CORS and security middleware
- Comprehensive error handling
- Automatic request tracking and performance monitoring

**When to use**:
- Serving explanations in production
- Building ML platforms with explainability
- Integrating SHAP into web applications
- Creating explanation APIs for enterprise systems

**Usage**:
```bash
# Run locally
python examples/api_deployment.py

# Run with uvicorn (production)
uvicorn examples.api_deployment:app --host 0.0.0.0 --port 8000 --workers 4

# Run with Docker
docker build -t shap-api -f Dockerfile .
docker run -p 8000:8000 shap-api
```

**API Endpoints**:
- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - **NEW:** API metrics (requests, success rate, latency)
- `GET /model/info` - Model metadata
- `POST /explain` - Single prediction explanation
- `POST /explain/batch` - Batch explanations

**Example Request**:
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, ...]
  }'
```

**Key Concepts**:
- Async API patterns for ML serving
- Model artifact management
- API versioning and monitoring
- Production deployment patterns

---

### 3. Drift Monitoring (`drift_monitoring.py`)

**Purpose**: Detect data drift and model degradation in production using statistical tests and SHAP.

**Features**:
- Multi-method drift detection (Kolmogorov-Smirnov, PSI)
- SHAP-based feature importance drift
- Sliding window monitoring for real-time systems
- Automated alerting with severity levels
- Comprehensive drift reports
- Simulated production scenarios

**When to use**:
- Production ML monitoring
- Detecting model degradation
- Feature distribution shifts
- Continuous model validation
- Automated retraining triggers

**Usage**:
```bash
python examples/drift_monitoring.py
```

**Drift Detection Methods**:
1. **Kolmogorov-Smirnov Test**: Statistical test for distribution differences
2. **Population Stability Index (PSI)**: Industry-standard drift metric
3. **SHAP Importance Drift**: Changes in feature importance patterns
4. **Distribution Statistics**: Mean/std shifts

**Alert Levels**:
- **Green**: No significant drift detected
- **Yellow**: Minor drift (< threshold), monitor closely
- **Red**: Significant drift detected, action required

**Key Concepts**:
- Statistical drift detection
- SHAP as a drift indicator
- Windowed monitoring for streaming data
- Alert threshold tuning

---

### 4. Production Pipeline (`production_pipeline.py`)

**Purpose**: Complete end-to-end ML pipeline integrating SHAP at every stage.

**Features**:
- Data ingestion and validation
- Feature engineering
- Model training with hyperparameter tuning
- Model comparison and selection
- SHAP-based model evaluation
- Model versioning and serialization
- Metadata tracking
- Explainability auditing

**When to use**:
- Building MLOps pipelines
- Automated model training workflows
- Model comparison and selection
- Enterprise ML systems
- Audit-ready model deployment

**Usage**:
```bash
python examples/production_pipeline.py
```

**Pipeline Stages**:
1. **Data Validation**: Check for missing values, data quality issues
2. **Feature Engineering**: Transform and create features
3. **Model Training**: Train multiple model types
4. **Model Evaluation**: Performance metrics + SHAP analysis
5. **Model Selection**: Choose best model based on performance and explainability
6. **Model Serialization**: Save model with metadata
7. **Deployment Preparation**: Generate deployment artifacts

**Output Artifacts**:
- `models/` - Serialized models (joblib)
- `metrics/` - Model metadata (JSON)
- `explainability/` - SHAP explanations

**Key Concepts**:
- MLOps pipeline design
- Model governance and tracking
- Explainability as a model quality metric
- Artifact management

---

### 5. Custom Visualization (`custom_visualization.py`)

**Purpose**: Create advanced, interactive SHAP visualizations using Plotly for dashboards and reports.

**Features**:
- Interactive feature importance charts
- Beeswarm plots for value distributions
- Dependence plots with interaction detection
- Waterfall charts for individual predictions
- Correlation heatmaps
- Multi-class comparison visualizations
- **NEW: Decision plots** - Cumulative SHAP contributions for multiple samples
- **NEW: Force plots** - Interactive HTML force plots for individual predictions
- **NEW: Interaction plots** - 2D feature interaction analysis
- **NEW: Partial dependence plots** - Marginal effects with data distribution
- **NEW: Violin plots** - SHAP value distributions with box plots
- Export to HTML for sharing

**When to use**:
- Creating executive dashboards
- Model documentation and reports
- Interactive exploration of explanations
- Stakeholder presentations
- Regulatory compliance reporting

**Usage**:
```bash
python examples/custom_visualization.py
```

**Visualizations Generated**:
1. **Feature Importance Bar Chart** - Global feature rankings
2. **Beeswarm Plot** - Distribution of SHAP values with feature values
3. **Dependence Plots** - Feature effects with interactions
4. **Waterfall Charts** - Individual prediction breakdowns
5. **Correlation Heatmap** - SHAP value correlations
6. **Decision Plot** - Cumulative feature contributions across samples
7. **Force Plots** - Interactive push/pull visualization for predictions
8. **Interaction Plots** - 2D scatter showing feature interactions
9. **Partial Dependence** - Marginal effect curves with rug plots
10. **Violin Plots** - Statistical distributions of SHAP values

**Output**:
- `examples/visualizations/*.html` - Interactive HTML plots

**Key Concepts**:
- Interactive visualizations for explainability
- Plotly for production dashboards
- Customizable chart templates
- Export formats for reporting

---

## Installation

### Basic Requirements

```bash
pip install -r examples/requirements.txt
```

### Optional Dependencies

For API deployment:
```bash
pip install fastapi uvicorn
```

For interactive dashboards:
```bash
pip install dash plotly
```

For advanced visualizations:
```bash
pip install plotly kaleido
```

## Quick Start

### Run All Examples

```bash
# Basic usage
python examples/basic_usage.py

# Drift monitoring
python examples/drift_monitoring.py

# Production pipeline
python examples/production_pipeline.py

# Custom visualizations
python examples/custom_visualization.py

# API deployment (separate terminal)
python examples/api_deployment.py
```

### Docker Deployment

Build and run the API example:
```bash
docker build -t shap-analytics-api -f Dockerfile .
docker run -p 8000:8000 -e API_WORKERS=4 shap-analytics-api
```

## Common Use Cases

### Use Case 1: Model Development
1. Start with `basic_usage.py` to understand feature importance
2. Use `custom_visualization.py` to create detailed analysis
3. Run `production_pipeline.py` to build complete workflow

### Use Case 2: Production Deployment
1. Use `production_pipeline.py` to train and validate model
2. Deploy with `api_deployment.py` for real-time explanations
3. Monitor with `drift_monitoring.py` for ongoing validation

### Use Case 3: Model Monitoring
1. Deploy model with `api_deployment.py`
2. Implement `drift_monitoring.py` for continuous monitoring
3. Generate reports with `custom_visualization.py`

### Use Case 4: Regulatory Compliance
1. Use `production_pipeline.py` for auditable training
2. Generate documentation with `custom_visualization.py`
3. Track explanations with `api_deployment.py` logging

## Performance Considerations

### Dataset Size
- **Small (<10K samples)**: Use full dataset for SHAP computation
- **Medium (10K-100K)**: Sample background data (100-500 samples)
- **Large (>100K)**: Use stratified sampling, compute SHAP on batches

### Model Types
- **Tree-based models** (RF, XGBoost): Use `TreeExplainer` (fastest)
- **Linear models**: Use `LinearExplainer`
- **Neural networks**: Use `DeepExplainer` or `KernelExplainer`

### Memory Usage
- Monitor memory with large SHAP value arrays
- Use `shap.sample()` for background data
- Process in batches for large explanation sets

### Optimization Tips
```python
# Use sampling for background
background = shap.sample(X_train, 100)
explainer = shap.TreeExplainer(model, background)

# Compute SHAP in batches
batch_size = 100
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    shap_values = explainer.shap_values(batch)
    # Process batch
```

## Troubleshooting

### Common Issues

**Issue**: `MemoryError` when computing SHAP values
**Solution**: Reduce sample size, use background sampling, compute in batches

**Issue**: Slow SHAP computation
**Solution**: Use TreeExplainer for tree models, reduce background samples

**Issue**: API timeout on large batches
**Solution**: Reduce batch size, increase timeout, use async processing

**Issue**: Visualizations not rendering
**Solution**: Install plotly/matplotlib, check file permissions

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### 1. Always Validate Data
```python
# Check for missing values, inf, data types
assert not X.isnull().any().any(), "Missing values detected"
assert not np.isinf(X.values).any(), "Infinite values detected"
```

### 2. Use Background Sampling
```python
# For large datasets, sample background
if len(X_train) > 1000:
    background = shap.sample(X_train, 100, random_state=42)
else:
    background = X_train
```

### 3. Cache Explainers
```python
# Cache explainer for reuse
explainer = shap.TreeExplainer(model, background)
# Store explainer with model
joblib.dump({'model': model, 'explainer': explainer}, 'model.pkl')
```

### 4. Handle Errors Gracefully
```python
try:
    shap_values = explainer.shap_values(X)
except Exception as e:
    logger.error(f"SHAP computation failed: {e}")
    # Fallback to model predictions only
```

### 5. Monitor Performance
```python
import time
start = time.time()
shap_values = explainer.shap_values(X)
elapsed = time.time() - start
logger.info(f"SHAP computation took {elapsed:.2f}s")
```

## Resources

### Documentation
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Plotly Documentation](https://plotly.com/python/)

### Papers
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
- Molnar, C. (2020). "Interpretable Machine Learning"

### Additional Examples
- Check the [SHAP GitHub repository](https://github.com/slundberg/shap) for more examples
- Explore [Plotly Dash Gallery](https://dash.gallery/) for dashboard inspiration

## Contributing

To add new examples:
1. Follow the existing structure and documentation style
2. Include comprehensive docstrings and error handling
3. Add logging for production readiness
4. Update this README with your example
5. Test with realistic data volumes

## Support

For questions or issues:
- Check existing [GitHub Issues](https://github.com/yourusername/shap-analytics/issues)
- Read the [SHAP FAQ](https://shap.readthedocs.io/en/latest/faq.html)
- Consult the documentation

## License

These examples are provided as part of the SHAP Analytics project. See the main repository LICENSE file for details.
