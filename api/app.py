import asyncio
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from core.visualization import visualize_cache_performance
from mock.mock_db import generate_mock_database
from mock.simulation import simulate_derived_data_weights
from core.utils import is_cuda_available, build_db_url, \
    build_custom_db_url  # lightweight utility (does not import torch)

# Global dictionaries to hold simulation & training job state
running_simulations = {}
training_jobs = {}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Create FastAPI app
app = FastAPI(
    title="Cache RL Optimization API",
    description="API for training and deploying RL models for database cache optimization",
    version="1.0.0",
    docs_url="/"
)
API = os.getenv("API_URL", "localhost:8000")

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


def get_job_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return training_jobs[job_id]


def get_database_connection(db_url, max_retries=5, initial_backoff=1, max_backoff=30):
    """Create a database engine with retry logic."""
    logger = logging.getLogger(__name__)
    retries = 0
    backoff = initial_backoff

    while True:
        try:
            logger.info(f"Attempting to connect to database (attempt {retries + 1}/{max_retries})...")
            engine = create_engine(
                db_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                connect_args={'connect_timeout': 10}
            )

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            masked_url = db_url.replace("://", "://***:***@", 1).split("@")[-1]
            logger.info(f"Successfully connected to database at {masked_url}")
            return engine

        except (SQLAlchemyError, OperationalError) as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                return None
            jitter = random.uniform(0, 0.1 * backoff)
            sleep_time = backoff + jitter
            logger.warning(f"Database connection failed. Retrying in {sleep_time:.2f}s. Error: {e}")
            time.sleep(sleep_time)
            backoff = min(backoff * 2, max_backoff)
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            return None


def start_training_in_process(job_id, db_url, algorithm, cache_size, max_queries,
                              timesteps, feature_columns, optimized_for_cpu,
                              use_gpu, gpu_id, batch_size, learning_rate):
    # Run training job asynchronously in a process
    asyncio.run(run_training_job(
        job_id, db_url, algorithm, cache_size, max_queries, timesteps,
        feature_columns, optimized_for_cpu, use_gpu, gpu_id, batch_size, learning_rate
    ))

# Utility function to get column names from the derived_data_cache_weights table
def get_derived_cache_columns(db_url: str) -> List[str]:
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = inspector.get_columns("derived_data_cache_weights")
        return [col["name"] for col in columns]
    except Exception as e:
        raise Exception(f"Could not retrieve columns: {e}")



@app.on_event("startup")
async def startup_db_client():
    logger = logging.getLogger(__name__)
    logger.info("Testing database connection on startup...")
    db_url = build_db_url()
    engine = get_database_connection(db_url)
    if engine is None:
        logger.warning("Could not connect to database, but service will continue to run")
    else:
        logger.info("Database connection test successful")


@app.get("/health")
def health_check():
    return {"status": "ok"}

# Example endpoint that returns available columns
@app.get("/available-columns", response_model=Dict[str, List[str]], tags=["database"])
def available_columns(
        db_type: str = Query("mysql+mysqlconnector", description="Database type: mysql, postgres, or sqlite"),
        host: str = Query("ers-mariadb", description="Database hostname"),
        port: int = Query(3306, description="Database port"),
        user: str = Query("cacheuser", description="Database username"),
        password: str = Query("cachepass", description="Database password"),
        database: str = Query("cache_db", description="Database name")):
    db_url = build_custom_db_url(db_type, host, port, database, user, password)
    columns = get_derived_cache_columns(db_url)
    return {"available_columns": columns}

@app.post("/train", response_model=TrainingResponse, tags=["training"], description="Start a new training job")
async def start_training(
    background_tasks: BackgroundTasks,
    db_type: str = Query("mysql+mysqlconnector", description="Database type: mysql, postgres, or sqlite"),
    host: str = Query("ers-mariadb", description="Database hostname"),
    port: int = Query(3306, description="Database port"),
    user: str = Query("cacheuser", description="Database username"),
    password: str = Query("cachepass", description="Database password"),
    database: str = Query("cache_db", description="Database name"),
    algorithm: str = Query("dqn", description="RL algorithm to use (dqn, a2c, ppo)"),
    cache_size: int = Query(10, description="Size of the cache"),
    max_queries: int = Query(500, description="Maximum number of queries for training"),
    timesteps: int = Query(100000, description="Training timesteps"),
    feature_columns: Optional[List[str]] = Query([""], description="Feature columns to use"),
    optimized_for_cpu: bool = Query(is_cuda_available(), description="Optimize for CPU training"),
    use_gpu: bool = Query(False, description="Use GPU for training if available"),
    gpu_id: Optional[int] = Query(None, description="Specific GPU ID to use (if multiple)"),
    batch_size: Optional[int] = Query(None, description="Batch size for training"),
    learning_rate: Optional[float] = Query(None, description="Learning rate for training")
):
    job_id = str(uuid4())
    start_time = datetime.now().isoformat()
    db_url = build_custom_db_url(db_type, host, port, database, user, password)

    logger.info(f"Starting training job: {job_id}")
    logger.info(f"Database URL: {db_url}")

    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "start_time": start_time,
        "end_time": None,
        "model_path": None,
        "metrics": None
    }

    background_tasks.add_task(
        start_training_in_process,
        job_id,
        db_url,
        algorithm,
        cache_size,
        max_queries,
        timesteps,
        feature_columns,
        optimized_for_cpu,
        use_gpu,
        gpu_id,
        batch_size,
        learning_rate
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "start_time": start_time
    }


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"], description="Get the status of a training job")
async def get_job(job_id: str):
    return get_job_status(job_id)


@app.get("/jobs", response_model=List[JobStatus], tags=["jobs"], description="List all training jobs")
async def list_jobs():
    return list(training_jobs.values())


@app.post("/evaluate/{job_id}", response_model=Dict[str, Any], tags=["evaluation"],
          description="Evaluate a trained model from a completed job")
async def evaluate_job_model(job_id: str, steps: int = 1000, use_gpu: bool = True):
    job = get_job_status(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
    if not job["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for job {job_id}")

    # Dynamically import the correct evaluation function based on use_gpu and availability.
    if use_gpu and is_cuda_available():
        from core.model_training_gpu import evaluate_cache_model
    else:
        from core.model_training_cpu import evaluate_cache_model

    results = evaluate_cache_model(
        model_path=job["model_path"],
        eval_steps=steps,
        db_url=None,  # default mock DB will be used inside the function
        use_gpu=use_gpu
    )

    try:
        vis_path = visualize_cache_performance(results)
        if vis_path:
            results["visualization"] = vis_path
    except Exception as e:
        results["visualization_error"] = str(e)

    return results


@app.post("/export/{job_id}", response_model=Dict[str, Any], tags=["deployment"])
async def export_job_model(job_id: str, output_dir: str = "best_model"):
    job = get_job_status(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
    if not job["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for job {job_id}")
    try:
        # For export, we will use the GPU version if available
        if is_cuda_available():
            from core.model_training_gpu import export_model_to_torchscript
        else:
            from core.model_training_cpu import export_model_to_torchscript

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
    try:
        training_jobs[job_id]["status"] = "running"

        # If use_gpu is not requested or not available, force CPU training.
        if not use_gpu or not is_cuda_available():
            optimized_for_cpu = True
            from core.model_training_cpu import train_cache_model
        else:
            from core.model_training_gpu import train_cache_model

        model_path = train_cache_model(
            db_url=db_url,
            algoritme=algorithm,
            cache_size=cache_size,
            max_queries=max_queries,
            timesteps=timesteps,
            feature_columns=feature_columns,
            gpu_id=gpu_id if use_gpu else None,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Evaluate the trained model using the same module that was used for training.
        if use_gpu and is_cuda_available():
            from core.model_training_gpu import evaluate_cache_model
        else:
            from core.model_training_cpu import evaluate_cache_model

        eval_results = evaluate_cache_model(
            model_path=model_path,
            eval_steps=1000,
            db_url=db_url,
            use_gpu=use_gpu
        )

        try:
            vis_path = visualize_cache_performance(eval_results)
            eval_results["visualization"] = vis_path
        except Exception as e:
            eval_results["visualization_error"] = str(e)

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
        host: str = Query("ers-mariadb", description="Database hostname"),
        port: int = Query(3306, description="Database port"),
        user: str = Query("cacheuser", description="Database username"),
        password: str = Query("cachepass", description="Database password"),
        database: str = Query("cache_db", description="Database name"),
        hours: int = Query(1000, description="Hours of data to generate"),
        db_type: str = Query("mysql", description="Database type: mysql, postgres, or sqlite")
):
    """Seed the database with mock energy data"""
    try:
        # Generate database with the specified database type
        success = generate_mock_database(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            hours=hours,
            db_type=db_type
        )

        if success:
            return {"status": "success", "message": f"Database seeded with {hours} hours of data"}
        else:
            raise HTTPException(status_code=500, detail="Failed to seed database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding database: {str(e)}")


@app.post("/simulation/start", response_model=Dict[str, Any], tags=["simulation"])
async def start_simulation(
        host: str = Query("ers-mariadb", description="Database hostname"),
        port: int = Query(3306, description="Database port"),
        user: str = Query("cacheuser", description="Database username"),
        password: str = Query("cachepass", description="Database password"),
        database: str = Query("cache_db", description="Database name"),
        update_interval: int = Query(5, description="Simulation update interval in seconds"),
        db_type: str = Query("mysql", description="Database type: mysql, postgres, or sqlite"),
        simulation_id: str = Query(None, description="Optional custom simulation ID")
):
    """Start a simulation of derived data usage"""
    try:
        # Generate a unique ID for this simulation if not provided
        sim_id = simulation_id or f"sim_{uuid4()}"

        # Check if a simulation with this ID is already running
        if sim_id in running_simulations:
            return {"status": "already_running", "simulation_id": sim_id}

        # Create database connection
        db_url = build_custom_db_url(db_type, host, port, database, user, password)
        success = get_database_connection(db_url)

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to connect to {db_type} database")

        # Create database engine
        engine = create_engine(db_url)
        db = engine.connect()

        # Create stop event for this simulation
        stop_event = threading.Event()

        # Start simulation in a separate thread
        sim_thread = threading.Thread(
            target=simulate_derived_data_weights,
            args=(db, update_interval, None, stop_event),
            daemon=True  # Thread will be terminated when main thread exits
        )
        sim_thread.start()

        # Store thread and stop event in global registry
        running_simulations[sim_id] = {
            "thread": sim_thread,
            "stop_event": stop_event,
            "start_time": datetime.now().isoformat(),
            "db_type": db_type,
            "database": database,
            "update_interval": update_interval
        }

        return {
            "status": "started",
            "simulation_id": sim_id,
            "start_time": running_simulations[sim_id]["start_time"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@app.post("/simulation/stop/{simulation_id}", response_model=Dict[str, Any], tags=["simulation"])
async def stop_simulation(simulation_id: str):
    """Stop a running simulation"""
    if simulation_id not in running_simulations:
        raise HTTPException(status_code=404, detail=f"No simulation found with ID {simulation_id}")

    try:
        # Signal the thread to stop
        running_simulations[simulation_id]["stop_event"].set()

        # Wait for thread to finish (with timeout)
        running_simulations[simulation_id]["thread"].join(timeout=5.0)

        # Store end time
        end_time = datetime.now().isoformat()

        # Clean up entry
        sim_data = running_simulations.pop(simulation_id)

        return {
            "status": "stopped",
            "simulation_id": simulation_id,
            "start_time": sim_data["start_time"],
            "end_time": end_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")


@app.get("/simulation/status", response_model=Dict[str, Any], tags=["simulation"])
async def simulation_status():
    """Get status of all running simulations"""
    result = {}
    for sim_id, data in running_simulations.items():
        result[sim_id] = {
            "db_type": data["db_type"],
            "database": data["database"],
            "update_interval": data["update_interval"],
            "start_time": data["start_time"],
            "running": data["thread"].is_alive()
        }
    return {"simulations": result, "count": len(running_simulations)}


@app.get("/simulation/status/{simulation_id}", response_model=Dict[str, Any], tags=["simulation"])
async def get_simulation_status(simulation_id: str):
    """Get status of a specific simulation"""
    if simulation_id not in running_simulations:
        raise HTTPException(status_code=404, detail=f"No simulation found with ID {simulation_id}")

    data = running_simulations[simulation_id]
    return {
        "simulation_id": simulation_id,
        "db_type": data["db_type"],
        "database": data["database"],
        "update_interval": data["update_interval"],
        "start_time": data["start_time"],
        "running": data["thread"].is_alive()
    }

@app.get("/cache/derived/weights", response_model=Dict[str, Any], tags=["cache"])
async def get_derived_data_weights(
        host: str = Query("ers-mariadb", description="Database hostname"),
        port: int = Query(3306, description="Database port"),
        user: str = Query("cacheuser", description="Database username"),
        password: str = Query("cachepass", description="Database password"),
        database: str = Query("cache_db", description="Database name"),
        endpoint: Optional[str] = Query(None, description="Filter by endpoint type")
):
    """Get current cache weights for derived data"""
    try:
        import mysql.connector

        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
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
