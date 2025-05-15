import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import os
import json

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect

from core.utils import list_available_models, is_cuda_available
from core.visualization import visualize_cache_performance

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

running_simulations = {}
training_models = {}


def load_trained_models(models_dir="models"):
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return []

    models = []
    for folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, folder)
        if not os.path.isdir(model_path):
            continue

        # Load metadata if present
        meta_path = os.path.join(model_path + ".meta.json")
        metadata = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        # Determine cache_size: prefer cache_size_mb, else legacy cache_size
        cache_size = metadata.get("cache_size_mb", metadata.get("cache_size", None))

        # Determine timestamp
        timestamp = metadata.get("trained_at", folder.split("_")[-1])

        # Build model_id string
        model_id = f"{metadata.get('algorithm', folder)}_{cache_size}_{timestamp}"

        models.append({
            "model_id":      model_id,
            "algorithm":     metadata.get("algorithm"),
            "device":        metadata.get("device"),
            # Expose cache_size in MB for clients
            "cache_size_mb": cache_size,
            "trained_at":    timestamp,
            "path":          model_path
        })

    return models


# Initialize the training_jobs dictionary with existing models
models = load_trained_models()
for model in models:
    training_models[model["model_id"]] = {
        "model_id": model["model_id"],
        "status": "completed",
        "start_time": model["trained_at"],
        "end_time": None,
        "model_path": model["path"],
        "metrics": model
    }
logger.info(f"Loaded {len(training_models)} trained models")


# database types
class DatabaseTypeEnum(str, Enum):
    mysql = "mysql"
    postgresql = "postgresql"
    sqlite = "sqlite"
    oracle = "oracle"
    mssql = "mssql"


# Define allowed algorithms
class AlgorithmEnum(str, Enum):
    dqn = "dqn"
    a2c = "a2c"
    ppo = "ppo"


# Add new enum for training weights
class CacheTableEnum(str, Enum):
    LOAD_TIME = "load_time_ms"
    SIZE = "size_bytes"
    INTENSITY = "traffic_intensity"


class TrainingResponse(BaseModel):
    model_id: str
    status: str
    start_time: str


class modelStatus(BaseModel):
    model_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


from core.model_training import train_cache_model, evaluate_cache_model


def get_status(model_id: str):
    if model_id not in training_models:
        raise HTTPException(status_code=404, detail=f"Job {model_id} not found")
    return training_models[model_id]


async def run_training_job(
        model_id: str,
        db_url: str,
        algorithm: AlgorithmEnum,
        cache_size: int,
        max_queries: int,
        timesteps: int,
        table_name: str,
        feature_columns: Optional[List],
        use_gpu: bool = True,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        cache_weights: Optional[List[str]] = None  # <-- add this param
):
    try:
        logger.info(f"Running training job {model_id} with algorithm {algorithm}")
        training_models[model_id]["status"] = "running"

        # If use_gpu is not requested or not available, force CPU training.
        logger.info(f"is_cuda_available: {is_cuda_available()}, use_gpu: {use_gpu}")
        if is_cuda_available() and use_gpu:
            logger.info(f"Training job {model_id} with algorithm {algorithm} using GPU")
        else:
            logger.info(f"Training job {model_id} with algorithm {algorithm} using CPU")
        model_path = train_cache_model(
            db_url=db_url,
            algorithm=algorithm,
            cache_size=cache_size,
            max_queries=max_queries,
            table_name=table_name,
            feature_columns=feature_columns,
            timesteps=timesteps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            cache_weights=cache_weights  # <-- pass through
        )
        logger.info(f"Evaluating model at {model_path}")
        eval_results = evaluate_cache_model(
            model_path=model_path,
            eval_steps=1000,
            db_url=db_url,
            use_gpu=use_gpu
        )

        logger.info(f"Training job {model_id} completed successfully. Model saved at {model_path}")
        # Evaluate the trained model using the same module that was used for training.

        logger.info(f"Evaluation results: {eval_results}")

        try:
            vis_path = visualize_cache_performance(eval_results)
            eval_results["visualization"] = vis_path
        except Exception as e:
            eval_results["visualization_error"] = str(e)

        training_models[model_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "model_path": model_path,
            "metrics": eval_results
        })

    except Exception as e:
        import traceback
        training_models[model_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def start_training_in_process(
        model_id: str,
        db_url: str,
        algorithm: AlgorithmEnum,
        cache_size: int,
        max_queries: int,
        timesteps: int,
        table_name: str,
        cache_keys: Optional[List[str]],
        use_gpu: bool,
        batch_size: Optional[int],
        learning_rate: Optional[float],
        feature_columns: List[str]  # added parameter
):
    """
    Background task wrapper for running training.
    """
    logger.info(f"Starting training for job {model_id}")
    # invoke training with feature_columns
    asyncio.run(run_training_job(
        model_id=model_id,
        db_url=db_url,
        algorithm=algorithm,
        cache_size=cache_size,
        max_queries=max_queries,
        timesteps=timesteps,
        table_name=table_name,
        feature_columns=feature_columns,  # pass through
        use_gpu=use_gpu,
        batch_size=batch_size,
        learning_rate=learning_rate,
        cache_weights=cache_keys  # <-- pass through
    ))


# Utility function to get column names from the cache_metrics table
def get_derived_cache_columns(db_url: str) -> List[str]:
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = inspector.get_columns("cache_metrics")
        return [col["name"] for col in columns]
    except Exception as e:
        raise Exception(f"Could not retrieve columns: {e}")


def get_dynamic_feature_columns_enum(db_url: str):
    """
    Dynamically create a FeatureColumnsEnum based on the columns in cache_metrics.
    """
    columns = get_derived_cache_columns(db_url)
    # Optionally filter out non-feature columns (like id, timestamp, etc.)
    exclude = {"id", "cache_name", "timestamp"}
    feature_columns = [col for col in columns if col not in exclude]
    # Dynamically create the Enum
    return Enum('FeatureColumnsEnum', {col: col for col in feature_columns})

