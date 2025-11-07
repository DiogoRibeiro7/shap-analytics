# SHAP Analytics Performance Benchmarks

Comprehensive performance benchmarking suite for SHAP Analytics ensuring scalability and detecting performance regressions.

## Overview

This benchmark suite provides:
- **Automated performance testing** with pytest-benchmark
- **Memory profiling** for optimization
- **Regression detection** to prevent performance degradation
- **CI/CD integration** via GitHub Actions
- **Detailed reports** in JSON and HTML formats

## Benchmark Suites

### 1. SHAP Computation (`benchmark_shap_computation.py`)

Tests core SHAP value computation performance across different scenarios.

**What it tests**:
- Model training time (Random Forest, Gradient Boosting)
- Expl

ainer creation overhead
- SHAP value computation for 100, 1K, 10K samples
- Impact of background sample size (10, 50, 100, 200, 500)
- Batch vs sequential processing
- Memory-efficient patterns

**Run**:
```bash
pytest benchmarks/benchmark_shap_computation.py -v --benchmark-autosave
```

**Key metrics**:
- Mean computation time per sample
- Throughput (samples/second)
- Memory usage scaling

### 2. API Throughput (`benchmark_api_throughput.py`)

Tests FastAPI endpoint performance and scalability.

**What it tests**:
- Single request latency
- Batch processing throughput (10, 50, 100 samples)
- Concurrent request handling (10, 100 requests)
- API health check response time
- Performance under load

**Prerequisites**:
```bash
# Start API in separate terminal
python examples/api_deployment.py
```

**Run**:
```bash
pytest benchmarks/benchmark_api_throughput.py -v --benchmark-autosave -m "not slow"
```

**Key metrics**:
- P50, P95, P99 latency
- Requests per second
- Concurrent request handling

### 3. Memory Usage (`benchmark_memory_usage.py`)

Profiles memory consumption and detects memory leaks.

**What it tests**:
- Model memory footprint
- Explainer memory usage
- Computation memory scaling
- Memory leak detection (repeated operations)
- Memory-efficient chunked processing

**Run**:
```bash
pytest benchmarks/benchmark_memory_usage.py -v -s -m memory
```

**Key metrics**:
- Memory usage in MB
- Peak memory
- Memory growth over iterations

### 4. Drift Detection (`benchmark_drift_detection.py`)

Tests drift monitoring system performance.

**What it tests**:
- KS test computation speed
- PSI calculation performance
- SHAP drift detection
- Full drift report generation
- Scaling with features and samples

**Run**:
```bash
pytest benchmarks/benchmark_drift_detection.py -v --benchmark-autosave
```

**Key metrics**:
- Time per feature drift check
- Full report generation time
- Scaling characteristics

## Installation

### Install benchmark dependencies:
```bash
pip install -r benchmarks/requirements.txt
```

### Core dependencies:
- pytest-benchmark
- memory-profiler
- psutil
- httpx (for API tests)

## Running Benchmarks

### Run all benchmarks:
```bash
pytest benchmarks/ -v --benchmark-autosave
```

### Run specific suite:
```bash
pytest benchmarks/benchmark_shap_computation.py -v
```

### Skip slow tests:
```bash
pytest benchmarks/ -v -m "not slow"
```

### Run with memory profiling:
```bash
pytest benchmarks/benchmark_memory_usage.py -v -s
```

### Generate HTML report:
```bash
pytest benchmarks/ --benchmark-histogram
```

## Performance Thresholds

### Regression Detection

Benchmarks include assertions to catch performance regressions:

| Metric | Threshold | Test |
|--------|-----------|------|
| SHAP 100 samples | < 2s | test_shap_computation_threshold_100 |
| SHAP 1K samples | < 10s | test_shap_computation_threshold_1k |
| API single request | < 1s | test_single_request_latency_threshold |
| API batch throughput | > 5 exp/s | test_batch_throughput_threshold |
| Drift detection (20 features) | < 5s | test_feature_drift_threshold |
| Full drift report | < 10s | test_full_drift_report_threshold |

## Comparing with Baseline

### Save current run as baseline:
```bash
pytest benchmarks/ --benchmark-save=baseline
```

### Compare against baseline:
```bash
pytest benchmarks/ --benchmark-compare=baseline
```

### Compare and fail if regression:
```bash
pytest benchmarks/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

## CI/CD Integration

### GitHub Actions Workflow

Benchmarks run automatically on:
- Pull requests to main/develop
- Pushes to main branch
- Manual workflow dispatch

**Workflow location**: `.github/workflows/performance-benchmark.yml`

**What it does**:
1. Runs all benchmark suites
2. Compares against baseline
3. Comments results on PR
4. Stores artifacts (JSON reports)
5. Fails CI if regression detected

### Viewing Results

**In PR comments**:
- Performance comparison table
- Regression warnings
- Link to detailed artifacts

**Artifacts**:
- Navigate to Actions → Workflow Run → Artifacts
- Download benchmark reports (JSON + HTML)

## Interpreting Results

### pytest-benchmark Output

```
Name (time in ms)              Min      Max    Mean  StdDev  Median     IQR
---------------------------------------------------------------------------
test_compute_shap_100      45.2    52.1    47.3    2.1    46.9    2.3
```

- **Mean**: Average execution time (primary metric)
- **Median**: Middle value (robust to outliers)
- **StdDev**: Consistency (lower is better)
- **IQR**: Interquartile range (spread of middle 50%)

### Memory Profiler Output

```
Memory usage for function_name:
  Before: 125.43 MB
  After: 167.21 MB
  Used: 41.78 MB
```

### Performance Trends

Monitor these over time:
1. **Computation time** should scale linearly with data size
2. **Memory usage** should be predictable and not leak
3. **API latency** should stay consistent under normal load
4. **Throughput** should scale with concurrent requests (up to a point)

## Troubleshooting

### Benchmarks failing locally

**Issue**: Tests timeout or fail
**Solution**:
- Check system resources
- Close other applications
- Run with fewer iterations: `--benchmark-min-rounds=1`

### API benchmarks fail

**Issue**: Connection refused
**Solution**:
```bash
# Start API first
python examples/api_deployment.py &
sleep 5  # Wait for startup
pytest benchmarks/benchmark_api_throughput.py
```

### Memory tests inconsistent

**Issue**: Memory measurements vary
**Solution**:
- Run garbage collection before tests
- Use `--benchmark-disable-gc` flag
- Increase warmup rounds

### Slow test execution

**Solution**:
```bash
# Skip slow tests
pytest benchmarks/ -m "not slow"

# Run in parallel (if tests are independent)
pytest benchmarks/ -n auto
```

## Best Practices

### 1. Regular Benchmarking
- Run benchmarks before major releases
- Track performance trends over time
- Set up automated alerts for regressions

### 2. Realistic Workloads
- Test with production-like data sizes
- Use representative feature dimensions
- Include edge cases (very large/small)

### 3. Isolate Tests
- Run on dedicated hardware when possible
- Disable background processes
- Use fixed random seeds

### 4. Document Changes
- Note hardware specifications
- Document Python/library versions
- Track system configuration

## Contributing

### Adding New Benchmarks

1. Create test file in `benchmarks/`
2. Use `@pytest.fixture` for data generation
3. Add `@pytest.mark.slow` for long-running tests
4. Include regression threshold assertions
5. Update this README

### Example benchmark:
```python
def test_my_operation(benchmark):
    """Benchmark my operation."""
    result = benchmark(my_function, arg1, arg2)
    assert benchmark.stats.stats.mean < THRESHOLD
```

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [memory_profiler docs](https://pypi.org/project/memory-profiler/)
- [SHAP documentation](https://shap.readthedocs.io/)
- [Performance testing guide](https://github.com/yourusername/shap-analytics/wiki/Performance-Testing)

## Support

For issues or questions:
- Create an issue with `[benchmark]` tag
- Include system specs and benchmark output
- Attach JSON report if available
