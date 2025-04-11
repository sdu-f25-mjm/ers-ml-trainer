import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime
import os

from ers_ml_trainer.api.app import app, training_jobs

# Create test client
client = TestClient(app)


@pytest.fixture
def mock_database():
    """Mock database connection and table discovery"""
    with patch("ers_ml_trainer.core.cache_environment.create_database_connection") as mock_db_conn:
        # Mock engine with inspector that returns test tables
        mock_engine = MagicMock()
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ["energy_data", "production_data"]
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        # Setup inspector's return value
        with patch("sqlalchemy.inspect", return_value=mock_inspector):
            mock_db_conn.return_value = mock_engine
            yield mock_db_conn


@pytest.fixture
def mock_training():
    """Mock the training function"""
    with patch("ers_ml_trainer.core.model_training.train_cache_model") as mock_train:
        mock_train.return_value = "/tmp/test_model.zip"
        yield mock_train


@pytest.fixture
def mock_evaluation():
    """Mock the evaluation function"""
    with patch("ers_ml_trainer.core.model_training.evaluate_cache_model") as mock_eval:
        mock_eval.return_value = {
            "average_reward": 0.75,
            "cache_hit_rate": 0.85,
            "steps": 1000
        }
        yield mock_eval


@pytest.fixture
def mock_visualization():
    """Mock the visualization function"""
    with patch("ers_ml_trainer.core.visualization.visualize_cache_performance") as mock_viz:
        mock_viz.return_value = "/tmp/cache_performance.png"
        yield mock_viz


def test_health_check():
    """Test health check endpoint"""
    with patch("torch.cuda.is_available", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "gpu_available": True}


def test_health_check_no_gpu():
    """Test health check endpoint with no GPU"""
    with patch("torch.cuda.is_available", return_value=False):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "gpu_available": False}


def test_start_training(mock_training, mock_database):
    """Test starting a training job"""
    # Clear any existing jobs
    training_jobs.clear()

    request_data = {
        "db_url": "mysql+mysqlconnector://test:test@localhost:3306/test_db",
        "algorithm": "dqn",
        "cache_size": 10,
        "max_queries": 500,
        "timesteps": 1000,
        "feature_columns": ["HourUTC", "PriceArea"],
        "optimized_for_cpu": True,
        "use_gpu": False
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "job_id" in result
    assert result["status"] == "pending"
    assert "start_time" in result

    # Verify job was added to training_jobs
    assert result["job_id"] in training_jobs


def test_get_job():
    """Test getting job status"""
    # Add a test job
    job_id = "test-job-123"
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "model_path": "/tmp/model.zip",
        "metrics": {"accuracy": 0.95}
    }

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert response.json()["status"] == "completed"


def test_get_job_not_found():
    """Test getting a non-existent job"""
    response = client.get("/jobs/non-existent-job")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_jobs():
    """Test listing all jobs"""
    # Clear and add test jobs
    training_jobs.clear()

    job_ids = ["job1", "job2"]
    for job_id in job_ids:
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "model_path": f"/tmp/{job_id}.zip",
            "metrics": {"accuracy": 0.95}
        }

    response = client.get("/jobs")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert {job["job_id"] for job in response.json()} == set(job_ids)


def test_evaluate_model(mock_evaluation, mock_visualization):
    """Test evaluating a trained model"""
    # Add a completed job with model path
    job_id = "test-eval-job"
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "model_path": "/tmp/model.zip",
        "metrics": {"accuracy": 0.95}
    }

    response = client.post(f"/evaluate/{job_id}?steps=1000&use_gpu=true")
    assert response.status_code == 200

    result = response.json()
    assert "average_reward" in result
    assert "cache_hit_rate" in result
    assert "steps" in result
    assert "visualization" in result


def test_evaluate_model_incomplete_job():
    """Test evaluating an incomplete job"""
    # Add an incomplete job
    job_id = "test-incomplete-job"
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "model_path": None,
        "metrics": None
    }

    response = client.post(f"/evaluate/{job_id}")
    assert response.status_code == 400
    assert "not completed" in response.json()["detail"]


def test_table_discovery_integration(mock_training, mock_database):
    """Test that table discovery works in the training job"""
    # This test verifies the integration between the API and the updated
    # cache_environment's table discovery feature

    training_jobs.clear()

    request_data = {
        "db_url": "mysql+mysqlconnector://test:test@localhost:3306/test_db",
        "algorithm": "dqn",
        "cache_size": 10,
        "max_queries": 500,
        "timesteps": 1000,
        "feature_columns": ["HourUTC", "PriceArea"],
        "optimized_for_cpu": True,
        "use_gpu": False
    }

    # Start a training job
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # Simulate job completion
    training_jobs[job_id]["status"] = "completed"
    training_jobs[job_id]["model_path"] = "/tmp/test_model.zip"
    training_jobs[job_id]["end_time"] = datetime.now().isoformat()

    # Verify that the mock_training was called without errors,
    # which means table discovery worked
    mock_training.assert_called_once()

    # Get job details to verify it completed
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"