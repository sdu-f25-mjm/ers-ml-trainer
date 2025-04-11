import re

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import torch
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
from uuid import uuid4

from core.model_training import train_cache_model, evaluate_cache_model
from core.visualization import visualize_cache_performance
import threading
from mock.mock_db import generate_mock_database, generate_cache_weights
from mock.simulation import simulate_derived_data_weights

# Global variables to track simulation threads
running_simulations = {}
app = FastAPI(
    title="Cache RL Optimization API",
    description="API for training and deploying RL models for database cache optimization",
    version="1.0.0",
    docs_url="/"
)
API = os.getenv("API_URL", "ers-mariadb")


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


@app.post("/train", response_model=TrainingResponse, tags=["training"], description="Start a new training job")
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


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"], description="Get the status of a training job")
async def get_job(job_id: str):
    """Get the status of a training job"""
    return get_job_status(job_id)


@app.get("/jobs", response_model=List[JobStatus], tags=["jobs"], description="List all training jobs")
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


@app.post("/export/{job_id}", response_model=Dict[str, Any], tags=["deployment"])
async def export_job_model(
        job_id: str,
        output_dir: str = "best_model"
):
    """Export a trained model to TorchScript format for production deployment"""
    job = get_job_status(job_id)

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")

    if not job["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for job {job_id}")

    try:
        from core.model_training import export_model_to_torchscript

        # Export the model to TorchScript format
        output_path = export_model_to_torchscript(
            model_path=job["model_path"],
            output_dir=output_dir
        )

        return {
            "job_id": job_id,
            "original_model": job["model_path"],
            "exported_model": output_path,
            "output_directory": output_dir,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")

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


@app.post("/db/seed", response_model=Dict[str, Any], tags=["database"])
async def seed_database(
        db_url: str = Query("mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db"),
        hours: int = Query(1000, description="Hours of data to generate"),
        price_areas: List[str] = Query([], description="Price areas to include")
):
    """Seed the database with mock energy data"""
    try:
        # Parse connection parameters from URL
        import re
        match = re.match(r'mysql\+mysqlconnector://([^:]+):([^@]+)@([^/]+)/([^?]+)', db_url)
        if not match:
            raise ValueError("Invalid database URL format")

        user, password, host_port, database = match.groups()
        host = host_port.split(':')[0]

        # Generate database
        success = generate_mock_database(
            host=host,
            user=user,
            password=password,
            database=database,
            hours=hours,
            price_areas=price_areas
        )

        if success:
            return {"status": "success", "message": f"Generated {hours} hours of data for {', '.join(price_areas)}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to seed database")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding database: {str(e)}")

@app.post("/db/weights", response_model=Dict[str, Any], tags=["database"])
async def generate_weights(
        db_url: str = Query("mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db")
):
    """Generate mock cache weights for database entries"""
    try:
        # Parse connection parameters from URL
        import re
        match = re.match(r'mysql\+mysqlconnector://([^:]+):([^@]+)@([^/]+)/([^?]+)', db_url)
        if not match:
            raise ValueError("Invalid database URL format")

        user, password, host_port, database = match.groups()
        host = host_port.split(':')[0]

        # Generate weights
        success = generate_cache_weights(
            host=host,
            user=user,
            password=password,
            database=database
        )

        if success:
            return {"status": "success", "message": "Cache weights generated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate cache weights")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating weights: {str(e)}")
@app.post("/simulation/start", response_model=Dict[str, Any], tags=["simulation"])
async def start_simulation(
        db_url: str = Query("mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db"),
        update_interval: int = Query(5, description="Seconds between updates"),
        access_intensity: int = Query(10, description="Number of items to access per interval"),
        simulation_id: Optional[str] = Query(None, description="Custom ID for the simulation")
):
    """Start a cache usage simulation"""
    try:
        # Parse connection parameters from URL
        import re
        match = re.match(r'mysql\+mysqlconnector://([^:]+):([^@]+)@([^/]+)/([^?]+)', db_url)
        if not match:
            raise ValueError("Invalid database URL format")

        user, password, host_port, database = match.groups()
        host = host_port.split(':')[0]

        # Generate a simulation ID if not provided
        sim_id = simulation_id or f"sim_{str(uuid4())[:8]}"

        if sim_id in running_simulations:
            return {"status": "already_running", "simulation_id": sim_id}

        # Start simulation in a separate thread with a stop event
        stop_event = threading.Event()

        def run_simulation():
            simulate_derived_data_weights(
                host=host,
                user=user,
                password=password,
                database=database,
                update_interval=update_interval,
                run_duration=None,
                stop_event=stop_event
            )

        # Start the simulation thread
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.daemon = True
        sim_thread.start()

        # Store simulation info
        running_simulations[sim_id] = {
            "thread": sim_thread,
            "stop_event": stop_event,
            "start_time": datetime.now().isoformat(),
            "config": {
                "db_url": db_url,
                "update_interval": update_interval,
                "access_intensity": access_intensity
            }
        }

        return {
            "status": "started",
            "simulation_id": sim_id,
            "message": "Simulation started successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@app.post("/simulation/{sim_id}/stop", response_model=Dict[str, Any], tags=["simulation"])
async def stop_simulation(sim_id: str):
    """Stop a running cache usage simulation"""
    if sim_id not in running_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")

    try:
        # Signal the thread to stop
        running_simulations[sim_id]["stop_event"].set()

        # Wait for thread to finish (with timeout)
        running_simulations[sim_id]["thread"].join(timeout=5)

        # Clean up
        simulation_info = running_simulations.pop(sim_id)

        return {
            "status": "stopped",
            "simulation_id": sim_id,
            "runtime": str(datetime.now() - datetime.fromisoformat(simulation_info["start_time"]))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")


@app.get("/simulation", response_model=List[Dict[str, Any]], tags=["simulation"])
async def list_simulations():
    """List all running simulations"""
    return [
        {
            "simulation_id": sim_id,
            "start_time": info["start_time"],
            "runtime": str(datetime.now() - datetime.fromisoformat(info["start_time"])),
            "config": info["config"]
        }
        for sim_id, info in running_simulations.items()
    ]

@app.get("/cache/derived/weights", response_model=Dict[str, Any], tags=["cache"])
async def get_derived_data_weights(
        db_url: str = Query("mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db"),
        endpoint: Optional[str] = Query(None, description="Filter by endpoint type")
):
    """Get current cache weights for derived data"""
    try:
        # Parse connection parameters
        match = re.match(r'mysql\+mysqlconnector://([^:]+):([^@]+)@([^/]+)/([^?]+)', db_url)
        if not match:
            raise ValueError("Invalid database URL format")

        user, password, host_port, database = match.groups()
        host = host_port.split(':')[0]

        import mysql.connector

        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cursor = conn.cursor(dictionary=True)

        if endpoint:
            cursor.execute(
                "SELECT * FROM derived_data_cache_weights WHERE endpoint = %s ORDER BY calculated_priority DESC",
                (endpoint,)
            )
        else:
            cursor.execute(
                "SELECT * FROM derived_data_cache_weights ORDER BY calculated_priority DESC"
            )

        results = cursor.fetchall()
        conn.close()

        return {
            "total_items": len(results),
            "endpoint_filter": endpoint,
            "weights": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving derived data weights: {str(e)}")

# Add this endpoint to app.py
@app.get("/logs", response_model=Dict[str, Any], tags=["monitoring"])
async def get_logs(
        lines: int = Query(100, description="Number of log lines to return"),
        level: Optional[str] = Query(None, description="Filter by log level (INFO, ERROR, etc.)"),
        search: Optional[str] = Query(None, description="Search string to filter logs"),
        log_file: Optional[str] = Query(None, description="Specific log file to read")
):
    """Retrieve application logs with optional filtering"""
    try:
        # Default log locations to check
        log_locations = [
            "logs/application.log"
        ]

        # Use specified log file if provided
        if log_file:
            log_locations = [log_file]

        # Find first available log file
        log_path = None
        for loc in log_locations:
            if os.path.exists(loc):
                log_path = loc
                break

        if not log_path:
            return {
                "status": "error",
                "message": "No log file found",
                "available_logs": [f for f in os.listdir("logs") if f.endswith(".log")] if os.path.exists(
                    "logs") else []
            }

        # Read log file with tail behavior
        with open(log_path, "r") as f:
            # Read all lines if file is small, otherwise read from end
            if os.path.getsize(log_path) < 1_000_000:  # 1MB threshold
                log_lines = f.readlines()
            else:
                # For large files, read the last portion
                f.seek(max(0, os.path.getsize(log_path) - 500_000))  # Read last ~500KB
                # Discard first line which might be partial
                f.readline()
                log_lines = f.readlines()

        # Apply filters
        if level:
            level_upper = level.upper()
            log_lines = [line for line in log_lines if f" - {level_upper} - " in line]

        if search:
            log_lines = [line for line in log_lines if search in line]

        # Return only the last 'lines' number of entries
        result_lines = log_lines[-lines:]

        # Parse log lines into structured data
        structured_logs = []
        for line in result_lines:
            # Parse log format: timestamp - module - level - message
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^ ]+) - ([A-Z]+) - (.*)', line)
            if match:
                structured_logs.append({
                    "timestamp": match.group(1),
                    "module": match.group(2),
                    "level": match.group(3),
                    "message": match.group(4).strip()
                })
            else:
                structured_logs.append({"raw": line.strip()})

        return {
            "total_lines": len(log_lines),
            "returned_lines": len(result_lines),
            "filters_applied": {
                "level": level,
                "search": search
            },
            "log_file": log_path,
            "logs": structured_logs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")