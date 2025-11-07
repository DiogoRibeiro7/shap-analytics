"""
FastAPI API Throughput Benchmark
=================================

Benchmarks FastAPI endpoint performance including:
- Single request latency
- Concurrent request handling
- Batch processing throughput
- Under load performance

Requirements:
    pytest-benchmark
    httpx
    fastapi
    uvicorn

Usage:
    # Start API first in separate terminal:
    python examples/api_deployment.py

    # Run benchmarks:
    pytest benchmarks/benchmark_api_throughput.py -v --benchmark-autosave
"""

import pytest
import asyncio
import time
from typing import List
import numpy as np
import httpx


API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_features():
    """Generate sample feature vectors for testing."""
    # 30 features matching breast cancer dataset
    return list(np.random.rand(30).astype(float))


@pytest.fixture(scope="module")
def api_client():
    """Create HTTP client for API testing."""
    return httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="module")
async def async_api_client():
    """Create async HTTP client."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=TIMEOUT) as client:
        yield client


# ============================================================================
# API Health Checks
# ============================================================================

class TestAPIHealth:
    """Test API availability and health."""

    @pytest.mark.api
    def test_api_health_check(self, benchmark, api_client):
        """Benchmark health endpoint response time."""
        def check_health():
            response = api_client.get("/health")
            assert response.status_code == 200
            return response.json()

        result = benchmark(check_health)
        assert result["status"] == "healthy"


# ============================================================================
# Single Request Benchmarks
# ============================================================================

class TestSingleRequest:
    """Benchmark single request performance."""

    @pytest.mark.api
    def test_single_explanation_latency(self, benchmark, api_client, sample_features):
        """Benchmark single SHAP explanation request latency."""
        payload = {"features": sample_features}

        def make_request():
            response = api_client.post("/explain", json=payload)
            assert response.status_code == 200
            return response.json()

        result = benchmark(make_request)
        assert "shap_values" in result
        assert len(result["shap_values"]) == len(sample_features)

    @pytest.mark.api
    def test_single_request_with_validation(self, benchmark, api_client, sample_features):
        """Benchmark request including response validation."""
        payload = {"features": sample_features}

        def make_validated_request():
            response = api_client.post("/explain", json=payload)
            data = response.json()

            # Validate response structure
            assert "prediction" in data
            assert "prediction_proba" in data
            assert "shap_values" in data
            assert "top_features" in data

            return data

        benchmark(make_validated_request)


# ============================================================================
# Batch Request Benchmarks
# ============================================================================

class TestBatchRequests:
    """Benchmark batch processing performance."""

    @pytest.mark.api
    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_batch_explanation_throughput(self, benchmark, api_client, sample_features, batch_size):
        """Benchmark batch explanation throughput for various sizes."""
        payload = {
            "instances": [sample_features for _ in range(batch_size)]
        }

        def make_batch_request():
            response = api_client.post("/explain/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == batch_size
            return data

        result = benchmark(make_batch_request)

        # Calculate throughput
        processing_time = result["processing_time"]
        throughput = batch_size / processing_time
        print(f"\nThroughput: {throughput:.2f} explanations/second")


# ============================================================================
# Concurrent Request Benchmarks
# ============================================================================

class TestConcurrentRequests:
    """Benchmark concurrent request handling."""

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_concurrent_requests_10(self, benchmark, async_api_client, sample_features):
        """Benchmark 10 concurrent requests."""
        payload = {"features": sample_features}

        async def make_concurrent_requests():
            tasks = [
                async_api_client.post("/explain", json=payload)
                for _ in range(10)
            ]
            responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in responses)
            return responses

        responses = await benchmark.pedantic(
            make_concurrent_requests,
            iterations=5,
            rounds=3
        )

    @pytest.mark.api
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests_100(self, benchmark, async_api_client, sample_features):
        """Benchmark 100 concurrent requests (slow)."""
        payload = {"features": sample_features}

        async def make_concurrent_requests():
            tasks = [
                async_api_client.post("/explain", json=payload)
                for _ in range(100)
            ]
            responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in responses)
            return responses

        await benchmark.pedantic(
            make_concurrent_requests,
            iterations=3,
            rounds=2
        )


# ============================================================================
# Performance Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Tests to catch API performance regressions."""

    @pytest.mark.api
    def test_single_request_latency_threshold(self, benchmark, api_client, sample_features):
        """Ensure single request stays under latency threshold."""
        payload = {"features": sample_features}

        result = benchmark(
            lambda: api_client.post("/explain", json=payload).json()
        )

        # Assert latency threshold: < 1 second for single request
        assert benchmark.stats.stats.mean < 1.0, "API latency regression detected!"

    @pytest.mark.api
    def test_batch_throughput_threshold(self, benchmark, api_client, sample_features):
        """Ensure batch processing maintains throughput."""
        batch_size = 50
        payload = {
            "instances": [sample_features for _ in range(batch_size)]
        }

        result = benchmark(
            lambda: api_client.post("/explain/batch", json=payload).json()
        )

        # Calculate throughput
        throughput = batch_size / result["processing_time"]

        # Assert minimum throughput: > 5 explanations/second
        assert throughput > 5.0, f"Throughput regression: {throughput:.2f} exp/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-autosave", "-m", "not slow"])
