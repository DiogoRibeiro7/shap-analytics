"""
FastAPI Deployment for SHAP Analytics
======================================

This example demonstrates how to deploy SHAP explanations as a production-ready
REST API using FastAPI with Docker support.

Features:
- RESTful API endpoints for SHAP explanations
- Model loading and caching
- Request validation with Pydantic
- Error handling and logging
- Health checks and metrics
- Rate limiting considerations
- Docker-ready configuration

Requirements:
- fastapi
- uvicorn
- pydantic
- shap
- scikit-learn
- numpy
- joblib

Usage:
    # Run locally
    python examples/api_deployment.py

    # Run with uvicorn (production)
    uvicorn examples.api_deployment:app --host 0.0.0.0 --port 8000

    # Run with Docker
    docker build -t shap-analytics-api .
    docker run -p 8000:8000 shap-analytics-api

API Endpoints:
    GET  /                    - Root endpoint
    GET  /health             - Health check
    GET  /metrics            - API metrics and monitoring
    POST /explain            - Generate SHAP explanations
    GET  /model/info         - Get model information
    POST /explain/batch      - Batch explanations
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and explainer (loaded at startup)
model: Optional[RandomForestClassifier] = None
explainer: Optional[shap.TreeExplainer] = None
feature_names: List[str] = []
model_metadata: Dict[str, Any] = {}

# Metrics tracking
metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "explanations_generated": 0,
    "total_processing_time": 0.0,
    "avg_processing_time": 0.0,
}


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================


class PredictionRequest(BaseModel):
    """Request model for single prediction explanation."""

    features: List[float] = Field(
        ..., description="Feature values for prediction", min_items=1, max_items=100
    )

    @validator("features")
    def validate_features(cls, v):
        """Validate feature values are finite."""
        if not all(np.isfinite(x) for x in v):
            raise ValueError("All feature values must be finite numbers")
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": [
                    17.99,
                    10.38,
                    122.8,
                    1001.0,
                    0.1184,
                    0.2776,
                    0.3001,
                    0.1471,
                    0.2419,
                    0.07871,
                    1.095,
                    0.9053,
                    8.589,
                    153.4,
                    0.006399,
                    0.04904,
                    0.05373,
                    0.01587,
                    0.03003,
                    0.006193,
                    25.38,
                    17.33,
                    184.6,
                    2019.0,
                    0.1622,
                    0.6656,
                    0.7119,
                    0.2654,
                    0.4601,
                    0.1189,
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    instances: List[List[float]] = Field(
        ...,
        description="List of feature arrays",
        min_items=1,
        max_items=1000,  # Limit batch size for performance
    )

    @validator("instances")
    def validate_batch(cls, v):
        """Validate batch consistency."""
        if not v:
            raise ValueError("Batch cannot be empty")

        # Check all instances have same length
        lengths = [len(inst) for inst in v]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All instances must have same number of features. Found: {set(lengths)}"
            )

        return v


class ShapExplanation(BaseModel):
    """Response model for SHAP explanation."""

    prediction: int = Field(..., description="Predicted class")
    prediction_proba: List[float] = Field(..., description="Class probabilities")
    shap_values: List[float] = Field(..., description="SHAP values for each feature")
    base_value: float = Field(..., description="Expected value (base)")
    feature_names: List[str] = Field(..., description="Feature names")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")


class BatchShapExplanation(BaseModel):
    """Response model for batch SHAP explanations."""

    explanations: List[ShapExplanation]
    processing_time: float = Field(..., description="Total processing time in seconds")
    count: int = Field(..., description="Number of explanations")


class ModelInfo(BaseModel):
    """Response model for model information."""

    model_type: str
    n_features: int
    feature_names: List[str]
    loaded_at: str
    version: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    uptime_seconds: float


# ============================================================================
# Application Lifecycle Management
# ============================================================================

start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - load model on startup, cleanup on shutdown.

    This is the modern FastAPI pattern for startup/shutdown events.
    """
    # Startup
    logger.info("Starting up SHAP Analytics API...")
    try:
        load_model_and_explainer()
        logger.info("Model and explainer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down SHAP Analytics API...")
    cleanup_resources()


def load_model_and_explainer():
    """
    Load or train model and create SHAP explainer.

    In production, you would load a pre-trained model from disk.
    For this example, we train a simple model on breast cancer data.
    """
    global model, explainer, feature_names, model_metadata

    try:
        model_path = os.environ.get("MODEL_PATH", "models/breast_cancer_rf.joblib")

        # Try to load existing model
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            model = model_data["model"]
            feature_names = model_data["feature_names"]
            logger.info("Model loaded from disk")
        else:
            # Train a new model (for demo purposes)
            logger.info("Training new model (no saved model found)...")
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)

            # Optionally save the model
            os.makedirs("models", exist_ok=True)
            joblib.dump({"model": model, "feature_names": feature_names}, model_path)
            logger.info(f"Model saved to {model_path}")

        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)

        # Store metadata
        model_metadata = {
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
        }

    except Exception as e:
        logger.error(f"Failed to load model and explainer: {e}")
        raise


def cleanup_resources():
    """Clean up resources on shutdown."""
    global model, explainer
    model = None
    explainer = None
    logger.info("Resources cleaned up")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="SHAP Analytics API",
    description="Production-ready API for SHAP model explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware (configure for your security needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware for Request Logging
# ============================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information and track metrics."""
    start_time = time.time()

    # Track request
    metrics["requests_total"] += 1

    try:
        response = await call_next(request)

        process_time = time.time() - start_time

        # Update metrics
        if response.status_code < 400:
            metrics["requests_success"] += 1
        else:
            metrics["requests_failed"] += 1

        metrics["total_processing_time"] += process_time
        metrics["avg_processing_time"] = (
            metrics["total_processing_time"] / metrics["requests_total"]
        )

        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.4f}s with status {response.status_code}"
        )

        response.headers["X-Process-Time"] = str(process_time)
        return response

    except Exception as e:
        metrics["requests_failed"] += 1
        logger.error(f"Request failed: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SHAP Analytics API",
        "version": "1.0.0",
        "description": "REST API for SHAP model explanations",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "explain": "/explain",
            "batch_explain": "/explain/batch",
            "model_info": "/model/info",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Health status including model load state and uptime
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "uptime_seconds": time.time() - start_time,
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics for monitoring and observability.

    Returns:
        Dictionary containing request counts, success/failure rates,
        and performance metrics.

    Note:
        For production, consider using Prometheus for metrics collection.
    """
    success_rate = (
        (metrics["requests_success"] / metrics["requests_total"] * 100)
        if metrics["requests_total"] > 0
        else 0.0
    )

    return {
        "requests": {
            "total": metrics["requests_total"],
            "success": metrics["requests_success"],
            "failed": metrics["requests_failed"],
            "success_rate_percent": round(success_rate, 2),
        },
        "explanations": {"generated": metrics["explanations_generated"]},
        "performance": {
            "total_processing_time_seconds": round(metrics["total_processing_time"], 2),
            "avg_processing_time_seconds": round(metrics["avg_processing_time"], 4),
        },
        "uptime_seconds": round(time.time() - start_time, 2),
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model metadata including type, features, and version
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return model_metadata


@app.post("/explain", response_model=ShapExplanation, tags=["Explanations"])
async def explain_prediction(request: PredictionRequest):
    """
    Generate SHAP explanation for a single prediction.

    Args:
        request: Prediction request with feature values

    Returns:
        SHAP explanation with values, prediction, and top features

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None or explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model or explainer not loaded"
        )

    try:
        # Validate feature count
        if len(request.features) != len(feature_names):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {len(feature_names)} features, got {len(request.features)}",
            )

        # Prepare input
        X = np.array(request.features).reshape(1, -1)

        # Get prediction
        prediction = int(model.predict(X)[0])
        prediction_proba = model.predict_proba(X)[0].tolist()

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

        # Handle multi-class (use predicted class)
        if isinstance(shap_values, list):
            shap_vals = shap_values[prediction][0]
            base_val = explainer.expected_value[prediction]
        else:
            shap_vals = shap_values[0]
            base_val = explainer.expected_value

        # Get top contributing features
        feature_contributions = [
            {
                "feature": feature_names[i],
                "value": float(request.features[i]),
                "shap_value": float(shap_vals[i]),
                "importance": abs(float(shap_vals[i])),
            }
            for i in range(len(feature_names))
        ]
        top_features = sorted(feature_contributions, key=lambda x: x["importance"], reverse=True)[
            :10
        ]

        # Track successful explanation
        metrics["explanations_generated"] += 1

        return {
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "shap_values": [float(v) for v in shap_vals],
            "base_value": float(base_val),
            "feature_names": feature_names,
            "top_features": top_features,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Explanation error: {str(e)}"
        )


@app.post("/explain/batch", response_model=BatchShapExplanation, tags=["Explanations"])
async def explain_batch(request: BatchPredictionRequest):
    """
    Generate SHAP explanations for multiple predictions.

    Args:
        request: Batch prediction request

    Returns:
        List of SHAP explanations with timing information

    Raises:
        HTTPException: If model not loaded or batch processing fails
    """
    if model is None or explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model or explainer not loaded"
        )

    start_time = time.time()

    try:
        explanations = []

        for features in request.instances:
            # Reuse single explanation logic
            single_request = PredictionRequest(features=features)
            explanation = await explain_prediction(single_request)
            explanations.append(explanation)

        processing_time = time.time() - start_time

        return {
            "explanations": explanations,
            "processing_time": processing_time,
            "count": len(explanations),
        }

    except Exception as e:
        logger.error(f"Batch explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch explanation error: {str(e)}",
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Configuration from environment variables
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    workers = int(os.environ.get("API_WORKERS", 1))
    log_level = os.environ.get("LOG_LEVEL", "info")

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "api_deployment:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,  # Set to True for development
    )
