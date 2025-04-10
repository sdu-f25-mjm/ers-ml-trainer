from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import torch
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from uuid import uuid4

from core.model_training import train_cache_model, evaluate_cache_model
from core.visualization import visualize_cache_performance

app = FastAPI(
    title="Cache RL Optimization API",
    description="API for training and deploying RL models for database cache optimization",
    version="1.0.0",
    docs_url="/"
)
API = os.getenv("API_URL", "http://localhost:8000")


# Data models for API requests/responses
class TrainingRequest(BaseModel):
    db_url: str = Field(..., description="Database connection URL")
    algorithm: str = Field("dqn", description="RL algorithm to use (dqn, a2c, ppo)")
    cache_size: int = Field(10, description="Size of the cache")
    max_queries: int = Field(500, description="Maximum number of queries for training")
    timesteps: int = Field(100000, description="Training timesteps")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    optimized_for_cpu: bool = Field(True, description="Optimize for CPU training")
    use_gpu: bool = Field(False, description="Use GPU for training if available")
    gpu_id: Optional[int] = Field(None, description="Specific GPU ID to use (if multiple)")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    learning_rate: Optional[float] = Field(None, description="Learning rate for training")


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    start_time: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# Store training jobs
training_jobs = {}


def get_job_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return training_jobs[job_id]

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}


@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training a new cache optimization model with GPU support"""
    job_id = str(uuid4())
    start_time = datetime.now().isoformat()

    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "start_time": start_time,
        "end_time": None,
        "model_path": None,
        "metrics": None
    }

    background_tasks.add_task(
        run_training_job,
        job_id,
        request.db_url,
        request.algorithm,
        request.cache_size,
        request.max_queries,
        request.timesteps,
        request.feature_columns,
        request.optimized_for_cpu,
        request.use_gpu,
        request.gpu_id,
        request.batch_size,
        request.learning_rate
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "start_time": start_time
    }


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get the status of a training job"""
    return get_job_status(job_id)


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all training jobs"""
    return list(training_jobs.values())


@app.post("/evaluate/{job_id}", response_model=Dict[str, Any],tags= ["evaluation"], description= "Evaluate a trained model from a completed job")
async def evaluate_job_model(job_id: str,
                             steps: int = 1000,
                             use_gpu: bool = True):
    """Evaluate a trained model from a completed job"""
    job = get_job_status(job_id)

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")

    if not job["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for job {job_id}")

    results = evaluate_cache_model(
        model_path=job["model_path"],
        eval_steps=steps,
        db_url=None,  # Will use the default mock DB
        use_gpu=use_gpu
    )

    try:
        visualization_path = visualize_cache_performance(results)
        if visualization_path:
            results["visualization"] = visualization_path
    except Exception as e:
        results["visualization_error"] = str(e)

    return results


async def run_training_job(
        job_id: str,
        db_url: str,
        algorithm: str,
        cache_size: int,
        max_queries: int,
        timesteps: int,
        feature_columns: Optional[List[str]],
        optimized_for_cpu: bool,
        use_gpu: bool = True,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
):
    """Run the training job in the background with GPU support"""
    try:
        training_jobs[job_id]["status"] = "running"

        # Force CPU mode if specifically requested
        if not use_gpu:
            optimized_for_cpu = True

        # Train the model
        model_path = train_cache_model(
            db_url=db_url,
            algoritme=algorithm,
            cache_size=cache_size,
            max_queries=max_queries,
            timesteps=timesteps,
            feature_columns=feature_columns,
            optimeret_for_cpu=optimized_for_cpu,
            gpu_id=gpu_id if use_gpu else None,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Evaluate the model
        eval_results = evaluate_cache_model(
            model_path=model_path,
            eval_steps=1000,
            db_url=db_url,
            use_gpu=use_gpu
        )

        # Visualization
        try:
            visualize_path = visualize_cache_performance(eval_results)
            eval_results["visualization"] = visualize_path
        except Exception as e:
            eval_results["visualization"] = None
            eval_results["visualization_error"] = str(e)

        # Update job status
        training_jobs[job_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "model_path": model_path,
            "metrics": eval_results
        })

    except Exception as e:
        import traceback
        training_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        })