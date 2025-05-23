# api/app.py
import base64
import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query

from api.app_utils import get_derived_cache_columns, get_dynamic_feature_columns_enum, TrainingResponse, AlgorithmEnum, \
    CacheTableEnum, \
    start_training_in_process, modelStatus, get_status, training_models,DatabaseTypeEnum
from config import DB_DRIVER, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_URL, API_URL
from core.model_training import evaluate_cache_model, export_model_to_torchscript
from core.utils import is_cuda_available, build_db_url, \
    build_custom_db_url
from core.visualization import visualize_cache_performance
from database.database_connection import get_database_connection, save_best_model_base64
from database.tables_enum import TableEnum
from mock.mock_db import generate_mock_database
from mock.simulate_live import simulate_visits

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

# Global dictionaries to hold simulation & training model state
app = FastAPI(
    title="Cache RL Optimization API",
    description="API for training and deploying RL models for database cache optimization.",
    version="1.0.0",
    docs_url="/"
)
running_simulations = {}


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


@app.get("/health", tags=["Monitoring"], description="Health check for API and GPU availability")
def health_check():
    return {"status": "ok", "gpu_available": is_cuda_available()}


@app.get("/available-columns", response_model=Dict[str, List[str]], tags=["Database"],
         description="Get available columns from the cache_metrics table")
def available_columns(
        db_type: str = Query(DB_DRIVER, description="Database type: mysql, postgres, or sqlite"),
        host: str = Query(DB_HOST, description="Database hostname"),
        port: int = Query(DB_PORT, description="Database port"),
        user: str = Query(DB_USER, description="Database username"),
        password: str = Query(DB_PASSWORD, description="Database password"),
        database: str = Query(DB_NAME, description="Database name")
):
    db_url = build_custom_db_url(db_type, host, port, database, user, password)
    columns = get_derived_cache_columns(db_url)
    return {"available_columns": columns}


@app.get("/feature-columns-enum", response_model=Dict[str, List[str]], tags=["Database"],
         description="Get available feature columns as enum values for selection")
def feature_columns_enum(
        db_type: str = Query(DB_DRIVER, description="Database type: mysql, postgres, or sqlite"),
        host: str = Query(DB_HOST, description="Database hostname"),
        port: int = Query(DB_PORT, description="Database port"),
        user: str = Query(DB_USER, description="Database username"),
        password: str = Query(DB_PASSWORD, description="Database password"),
        database: str = Query(DB_NAME, description="Database name")
):
    """
    Returns the available feature columns as enum values for selection.
    """
    db_url = build_custom_db_url(db_type, host, port, database, user, password)
    FeatureColumnsEnum = get_dynamic_feature_columns_enum(db_url)
    return {"feature_columns_enum": [e.value for e in FeatureColumnsEnum]}


@app.post("/train", response_model=TrainingResponse, tags=["Training"],
          description="Start a new RL model training job for cache optimization")
async def start_training(
        background_tasks: BackgroundTasks,
        db_type: str = Query(
            DB_DRIVER,
            description="Database driver/dialect for connection (e.g., mysql, postgresql, sqlite)"
        ),
        host: str = Query(
            DB_HOST,
            description="Hostname or IP address of the database server"
        ),
        port: int = Query(
            DB_PORT,
            description="Port number where the database server is listening"
        ),
        user: str = Query(
            DB_USER,
            description="Username with privileges to connect to the database"
        ),
        password: str = Query(
            DB_PASSWORD,
            description="Password for the database user"
        ),
        database: str = Query(
            DB_NAME,
            description="Name of the database/schema to connect to"
        ),
        algorithm: AlgorithmEnum = Query(
            AlgorithmEnum.ppo,
            description="Reinforcement learning algorithm for cache model (options: "
                        + ", ".join([e.value for e in AlgorithmEnum]) + ")"
        ),
        cache_size: int = Query(
            30,
            description="Maximum number of items the simulated cache can hold"
        ),
        max_queries: int = Query(
            500,
            description="Total number of simulated queries to run per episode"
        ),
        timesteps: int = Query(
            200000,
            description="Number of training timesteps (env steps) to execute"
        ),

        table_name: str = Query(
            "cache_metrics",
            description="Name of the table containing cache metrics (options: "
                        + ", ".join([e.value for e in TableEnum]) + ")"
        ),
        feature_columns: Optional[List[str]] = Query(
            None,
            description="List of metric columns to use as features; None = auto‐detect all"
        ),
        cache_weights: Optional[List[CacheTableEnum]] = Query(
            None,
            description="Optional metric weights; None = equal weighting across all"
        ),

        # Compute & optimizer
        use_gpu: bool = Query(
            True,
            description="Enable GPU if CUDA is available"
        ),
        batch_size: Optional[int] = Query(
            128,
            description="Batch size for each training update"
        ),
        learning_rate: Optional[float] = Query(
            3e-4,
            description="Learning rate for the RL optimizer"
        ),
):
    model_id = f"{algorithm.value}-{str(uuid4())}"
    logger.info(f"Starting training model {model_id}")
    start_time = datetime.now().isoformat()
    db_url = build_custom_db_url(db_type, host, port, database, user, password)
    logger.info(f"Database URL: {db_url}")

    training_models[model_id] = {
        "model_id": model_id,
        "status": "pending",
        "start_time": start_time,
        "end_time": None,
        "model_path": None,
        "metrics": None
    }
    logger.info("Training model added to queue")

    cache_keys = [f.value for f in cache_weights] if cache_weights else None

    # fetch actual columns once
    available_cols = get_derived_cache_columns(db_url)

    if feature_columns:
        invalid = [col for col in feature_columns if col not in available_cols]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature columns: {invalid}"
            )
        selected_features = feature_columns
    else:
        selected_features = available_cols

    background_tasks.add_task(
        start_training_in_process,
        model_id,
        db_url,
        algorithm,
        cache_size,  # <-- cache_size is passed here
        max_queries,
        timesteps,
        table_name,
        cache_keys,  # <-- pass cache_weights as cache_keys
        use_gpu,
        batch_size,
        learning_rate,
        selected_features
    )

    return {
        "model_id": model_id,
        "status": "pending",
        "start_time": start_time
    }


@app.get("/models/{model_id}", response_model=modelStatus, tags=["Training"],
         description="Get the status and details of a specific training model")
async def get_model(model_id: str):
    return get_status(model_id)


@app.get("/models", response_model=List[modelStatus], tags=["Training"],
         description="List all training models and their statuses")
async def list_models():
    return list(training_models.values())


@app.post("/evaluate/{model_id}", response_model=Dict[str, Any], tags=["Training"],
          description="Evaluate a trained RL model and return performance metrics")
async def evaluate_model(model_id: str, steps: int = 1000, use_gpu: bool = False):
    model = get_status(model_id)
    if model["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not completed")
    if not model["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for model {model_id}")

    results = evaluate_cache_model(
        model_path=model["model_path"],
        eval_steps=steps,
        db_url=DB_URL,
        use_gpu=use_gpu
    )

    try:
        vis_paths = visualize_cache_performance(results)
        if vis_paths:
            results["visualizations"] = vis_paths
    except Exception as e:
        results["visualization_error"] = str(e)

    return results


@app.post("/export/{model_id}", response_model=Dict[str, Any], tags=["Training"],
          description="Export a trained model as TorchScript and save to database")
async def export_model(model_id: str, output_dir: str = "best_model"):
    model = get_status(model_id)
    if model["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not completed")
    if not model["model_path"]:
        raise HTTPException(status_code=400, detail=f"No model path found for model {model_id}")
    try:
        db_url = build_db_url()
        engine = get_database_connection(db_url)

        model_path = model["model_path"]
        logger.info(f"Exporting model: {model_path}")
        output_path = export_model_to_torchscript(
            model_path=model["model_path"],
            output_dir=output_dir
        )
        logger.info(f"Exported model to: {output_path}")

        # Save metadata for Java/consumer compatibility
        meta_path = os.path.join(output_dir, "metadata.json")
        
        # Load comprehensive metadata from training if available
        training_meta_path = model_path + ".meta.json"
        comprehensive_metadata = {}
        if os.path.exists(training_meta_path):
            with open(training_meta_path, "r", encoding="utf-8") as f:
                comprehensive_metadata = json.load(f)
        elif os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                comprehensive_metadata = json.load(f)

        with open(output_path, "rb") as f:
            model_bytes = f.read()
        model_base64 = base64.b64encode(model_bytes).decode("utf-8")

        # Determine input_dimension from comprehensive_metadata
        input_dimension = None
        if "obs_shape" in comprehensive_metadata and comprehensive_metadata["obs_shape"]:
            input_dimension = comprehensive_metadata["obs_shape"][0]
        elif "feature_columns" in comprehensive_metadata and "cache_size" in comprehensive_metadata:
            feature_count = len(comprehensive_metadata.get("feature_columns", []))
            cache_s = comprehensive_metadata.get("cache_size", 0)
            input_dimension = feature_count + cache_s
            logger.info(f"Calculated input_dimension={input_dimension} from features and cache_size in training metadata.")
        
        if not input_dimension or input_dimension <= 0:
            input_dimension = 43
            logger.warning(f"Could not determine input dimension from metadata, using default: {input_dimension}")

        save_best_model_base64(
            engine=engine,
            metadata=comprehensive_metadata,
            model_base64=model_base64,
            input_dimension=input_dimension
        )
        return {
            "model_id": model_id,
            "original_model": model["model_path"],
            "exported_model": output_path,
            "output_directory": output_dir,
            "status": "success",
            "saved_to_db": True,
            "metadata_path": meta_path,
            "input_dimension": input_dimension
        }
    except Exception as e:
        logger.error(f"Failed to export model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")


@app.post("/db/seed", response_model=Dict[str, Any], tags=["Database"],
          description="Seed the database with mock or simulated cache metrics and energy data")
async def seed_database(
        db_type: str = Query(DatabaseTypeEnum.mysql, description="Database type: mysql, postgres, or sqlite"),
        host: str = Query(DB_HOST, description="Database hostname"),
        port: int = Query(DB_PORT, description="Database port"),
        user: str = Query(DB_USER, description="Database username"),
        password: str = Query(DB_PASSWORD, description="Database password"),
        database: str = Query(DB_NAME, description="Database name"),
        api_url: str = Query(API_URL, description="API URL for data generation"),
        hours: int = Query(1000, description="Hours of data to generate"),
        data_types: Optional[List[TableEnum]] = Query(None, description="Data types: " + ", ".join(
            [e.name for e in TableEnum])),
        use_simulate_live: bool = Query(False,
            description="If true, use the new simulate_live.simulate_visits for cache_metrics data"
        ),
        amount_of_urls : int = Query(10,
            description="Number of endpoints to use for simulation; if None, all endpoints are used"
        ),
        n: int = Query(10000,
                       description="Number of simulated visits for simulate_live (only used if use_simulate_live=True)")
):
    """Seed the database with mock energy data"""
    try:
        # Generate database with the specified database type
        if use_simulate_live:
            # Use simulate_visits directly for cache_metrics
            from mock.mock_db import get_db_handler
            db_handler = get_db_handler(db_type)
            if db_type == 'sqlite':
                if not db_handler.connect('', 0, '', '', database):
                    raise HTTPException(status_code=500, detail="Failed to connect to SQLite database")
            else:
                if not db_handler.connect(host, port, user, password, database):
                    raise HTTPException(status_code=500, detail=f"Failed to connect to {db_type} database")
            from database.create_tables import create_tables
            create_tables(db_handler)
            simulate_visits(
                n=n,
                update_interval=0,
                api_url=api_url,
                run_duration=hours,
                stop_event=None,
                endpoints_to_use=amount_of_urls
            )
            db_handler.commit()
            db_handler.close()
            return {"status": "success", "message": f"Database seeded with {n} simulated visits using simulate_live"}
        else:
            success = generate_mock_database(
                host=host,
                user=user,
                password=password,
                database=database,
                port=port,
                hours=hours,
                db_type=db_type,
                data_types=data_types,
            )

            if success:
                return {"status": "success", "message": f"Database seeded with {hours} hours of data"}
            else:
                raise HTTPException(status_code=500, detail="Failed to seed database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding database: {str(e)}")


@app.post("/simulation/start", response_model=Dict[str, Any], tags=["Simulation"],
          description="Start a background simulation of cache metrics")
async def start_simulation(
        update_interval: int = Query(5),
        api_url: str = Query(API_URL, description="API URL for data generation"),
        run_duration: Optional[int] = Query(None, description="Duration of the simulation in seconds"),
        simulation_id: str = Query(None),
        amount_of_urls : int = Query(10,description="Number of endpoints to use for simulation; if None, all endpoints are used"
        ),
        n: int = Query(10000, description="Number of simulated visits to generate"),
):
    try:
        sim_id = simulation_id or f"sim_{uuid4()}"
        if sim_id in running_simulations:
            return {"status": "already_running", "simulation_id": sim_id}

        from mock.mock_db import get_db_handler

        stop_event = threading.Event()

        sim_thread = threading.Thread(
            target=simulate_visits,
            args=(n, update_interval, api_url, run_duration, stop_event,amount_of_urls),  # adapt as needed
            daemon=True
        )

        sim_thread.start()

        running_simulations[sim_id] = {
            "thread": sim_thread,
            "stop_event": stop_event,
            "start_time": datetime.now().isoformat(),
            "run_duration": run_duration
        }

        return {
            "status": "started",
            "simulation_id": sim_id,
            "start_time": running_simulations[sim_id]["start_time"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@app.post("/simulation/stop/{simulation_id}", response_model=Dict[str, Any], tags=["Simulation"],
          description="Stop a running simulation by ID")
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


@app.get("/simulation/status", response_model=Dict[str, Any], tags=["Simulation"],
         description="Get status of all running cache metric simulations")
async def simulation_status():
    """Get status of all running simulations"""
    result = {}
    for sim_id, data in running_simulations.items():
        result[sim_id] = {
            "run_duration": data["run_duration"],
            "start_time": data["start_time"],
            "running": data["thread"].is_alive()
        }
    return {"simulations": result, "count": len(running_simulations)}


@app.get("/simulation/status/{simulation_id}", response_model=Dict[str, Any], tags=["Simulation"],
         description="Get status of a specific simulation by ID")
async def get_simulation_status(simulation_id: str):
    """Get status of a specific simulation"""
    if simulation_id not in running_simulations:
        raise HTTPException(status_code=404, detail=f"No simulation found with ID {simulation_id}")

    data = running_simulations[simulation_id]
    return {
        "simulation_id": simulation_id,
        "db_type": data["db_type"],
        "database": data["database"],
        "run_duration": data["run_duration"],
        "start_time": data["start_time"],
        "running": data["thread"].is_alive()
    }


@app.get("/cache/derived/weights", response_model=Dict[str, Any], tags=["Training"],
         description="Get current cache weights for derived data, optionally filtered by endpoint")
async def get_derived_data_weights(
        host: str = Query(DB_HOST, description="Database hostname"),
        port: int = Query(DB_PORT, description="Database port"),
        user: str = Query(DB_USER, description="Database username"),
        password: str = Query(DB_PASSWORD, description="Database password"),
        database: str = Query(DB_NAME, description="Database name"),
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
                "SELECT * FROM cache_metrics WHERE endpoint = %s ORDER BY calculated_priority DESC",
                (endpoint,)
            )
        else:
            cursor.execute(
                "SELECT * FROM cache_metrics ORDER BY calculated_priority DESC"
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


@app.get("/logs", response_model=Dict[str, Any], tags=["Monitoring"],
         description="Retrieve application logs with optional filtering")
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
