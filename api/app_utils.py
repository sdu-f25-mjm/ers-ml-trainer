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

# -----------------------------------------------------------------------------
# Configure module‐level logger with UTF-8 encoding to avoid UnicodeEncodeError
# -----------------------------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "application.log")

file_handler = logging.FileHandler(log_path, encoding="utf-8")

# --- Patch: Ensure StreamHandler uses UTF-8 encoding on Windows ---
import sys
if sys.platform.startswith("win"):
    import io
    stream_handler = logging.StreamHandler(stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True))
else:
    stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)

running_simulations: Dict[str, Any] = {}
training_models: Dict[str, Any] = {}


def load_trained_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return []

    models: List[Dict[str, Any]] = []
    for folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, folder)
        if not os.path.isdir(model_path):
            continue

        # Load metadata if present
        meta_path = model_path + ".meta.json"
        metadata: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read metadata for {folder}: {e}")

        # Extract cache_size and timestamp
        cache_size = metadata.get("cache_size")
        trained_at = metadata.get("trained_at", folder.split("_")[-1])

        # Build a stable model_id
        algo = metadata.get("algorithm", folder)
        model_id = f"{algo}_{cache_size}_{trained_at}"

        models.append({
            "model_id":   model_id,
            "algorithm":  metadata.get("algorithm"),
            "device":     metadata.get("device"),
            "cache_size": cache_size,
            "trained_at": trained_at,
            "path":       model_path,
        })

    return models


# Initialize the training_models dict with existing models
_existing_models = load_trained_models()
for m in _existing_models:
    training_models[m["model_id"]] = {
        "model_id":   m["model_id"],
        "status":     "completed",
        "start_time": m["trained_at"],
        "end_time":   None,
        "model_path": m["path"],
        "metrics":    m,
    }

logger.info(f"Loaded {len(training_models)} trained models")


# Supported database types
class DatabaseTypeEnum(str, Enum):
    mysql = "mysql"
    postgresql = "postgresql"
    sqlite = "sqlite"
    oracle = "oracle"
    mssql = "mssql"


# Supported RL algorithms
class AlgorithmEnum(str, Enum):
    dqn = "dqn"
    a2c = "a2c"
    ppo = "ppo"


# Cache‐table enums for weighting
class CacheTableEnum(str, Enum):
    LOAD_TIME = "load_time_ms"
    SIZE      = "size_bytes"
    INTENSITY = "traffic_intensity"


class TrainingResponse(BaseModel):
    model_id:    str
    status:      str
    start_time:  str


class modelStatus(BaseModel):
    model_id:    str
    status:      str
    start_time:  str
    end_time:    Optional[str] = None
    model_path:  Optional[str] = None
    metrics:     Optional[Dict[str, Any]] = None


from core.model_training import train_cache_model, evaluate_cache_model  # noqa: E402


def get_status(model_id: str) -> Dict[str, Any]:
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
    feature_columns: Optional[List[str]],
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    cache_weights: Optional[List[str]] = None,
) -> None:
    try:
        logger.info(f"Running training job {model_id} with algorithm {algorithm}")
        training_models[model_id]["status"] = "running"

        # Determine compute device
        cuda_ok = is_cuda_available() and use_gpu
        logger.info(f"is_cuda_available: {is_cuda_available()}, use_gpu: {use_gpu}")
        logger.info(f"Using {'GPU' if cuda_ok else 'CPU'} for training")

        # Train
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
            cache_weights=cache_weights,
        )

        # Evaluate
        logger.info(f"Evaluating model at {model_path}")
        eval_results = evaluate_cache_model(
            model_path=model_path,
            eval_steps=1000,
            db_url=db_url,
            use_gpu=use_gpu,
        )

        logger.info(f"Training job {model_id} completed. Model saved at {model_path}")
        logger.info(f"Evaluation results: {eval_results}")

        # Visualize (optional)
        try:
            vis = visualize_cache_performance(eval_results)
            eval_results["visualization"] = vis
        except Exception as e:
            eval_results["visualization_error"] = str(e)

        # Update status
        training_models[model_id].update({
            "status":     "completed",
            "end_time":   datetime.utcnow().isoformat() + "Z",
            "model_path": model_path,
            "metrics":    eval_results,
        })

    except Exception as e:
        import traceback
        training_models[model_id].update({
            "status":     "failed",
            "end_time":   datetime.utcnow().isoformat() + "Z",
            "error":      str(e),
            "traceback":  traceback.format_exc(),
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
    feature_columns: List[str],
) -> None:
    """
    Wrapper to launch the async training job in a fresh asyncio loop,
    so FastAPI BackgroundTasks doesn’t block the main loop.
    """
    logger.info(f"Starting background task for model {model_id}")
    asyncio.run(run_training_job(
        model_id=model_id,
        db_url=db_url,
        algorithm=algorithm,
        cache_size=cache_size,
        max_queries=max_queries,
        timesteps=timesteps,
        table_name=table_name,
        feature_columns=feature_columns,
        use_gpu=use_gpu,
        batch_size=batch_size,
        learning_rate=learning_rate,
        cache_weights=cache_keys,
    ))


def get_derived_cache_columns(db_url: str) -> List[str]:
    """Inspect the cache_metrics table and return its column names."""
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        cols = inspector.get_columns("cache_metrics")
        return [c["name"] for c in cols]
    except Exception as e:
        raise RuntimeError(f"Could not retrieve cache_metrics columns: {e}")


def get_dynamic_feature_columns_enum(db_url: str) -> Enum:
    """
    Dynamically build an Enum of feature column names (excludes id, timestamp, etc.).
    """
    cols = get_derived_cache_columns(db_url)
    exclude = {"id", "cache_name", "timestamp"}
    choices = {c: c for c in cols if c not in exclude}
    return Enum("FeatureColumnsEnum", choices)
