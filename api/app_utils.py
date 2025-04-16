import asyncio
import logging

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

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
training_jobs = {}

def load_trained_models():
    """
    Load all trained models at runtime and populate the training_jobs dictionary
    """
    global training_jobs
    logger.info("Loading trained models into memory...")
    models = list_available_models()
    for model in models:
        model_id = f"{model['algorithm']}_{model['cache_size']}_{model['timestamp']}"
        training_jobs[model_id] = {
            "job_id": model_id,
            "status": "completed",
            "start_time": model['created_at'],
            "end_time": None,
            "model_path": model['path'],
            "metrics": model['metadata']
        }
    logger.info(f"Loaded {len(training_jobs)} trained models")
    return training_jobs


# Initialize the training_jobs dictionary with existing models
load_trained_models()


# Define allowed algorithms
class AlgorithmEnum(str, Enum):
    dqn = "dqn"
    a2c = "a2c"
    ppo = "ppo"

# Add new enum for training weights
class CacheTableEnum(str, Enum):
    CACHE_WEIGHTS = "derived_data_cache_weights"


class FeatureColumnsEnum(str, Enum):
    recency = "recency"
    access_frequency = "usage_frequency"
    time_relevance = "time_relevance"
    production_importance = "production_importance"
    volatility = "volatility"
    complexity = "complexity"
    calculated_priority = "calculated_priority"
    last_accessed = "last_accessed"
    access_count = "access_count"


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

from core.model_training import train_cache_model, evaluate_cache_model

def get_job_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return training_jobs[job_id]


async def run_training_job(
        job_id: str,
        db_url: str,
        algorithm: Optional[List],
        cache_size: int,
        max_queries: int,
        timesteps: int,
        table_name: str,
        feature_columns: Optional[List],
        use_gpu: bool = True,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
):
    try:
        logger.info(f"Running training job {job_id} with algorithm {algorithm}")
        training_jobs[job_id]["status"] = "running"

        # If use_gpu is not requested or not available, force CPU training.
        logger.info(f"is_cuda_available: {is_cuda_available()}, use_gpu: {use_gpu}")
        if is_cuda_available() and use_gpu:
            logger.info(f"Training job {job_id} with algorithm {algorithm} using GPU")
        else:
            logger.info(f"Training job {job_id} with algorithm {algorithm} using CPU")
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
        )
        logger.info(f"Evaluating model at {model_path}")
        eval_results = evaluate_cache_model(
            model_path=model_path,
            eval_steps=1000,
            db_url=db_url,
            use_gpu=use_gpu
        )

        logger.info(f"Training job {job_id} completed successfully. Model saved at {model_path}")
        # Evaluate the trained model using the same module that was used for training.

        logger.info(f"Evaluation results: {eval_results}")

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

def start_training_in_process(job_id, db_url, algorithm, cache_size, max_queries,
                              timesteps,table_name, feature_columns,
                              use_gpu, batch_size, learning_rate
                              ):
    """Start training in a separate process."""
    logger.info(f"Starting training for job {job_id}")
    # Run training job asynchronously in a process
    asyncio.run(run_training_job(
        job_id, db_url, algorithm, cache_size, max_queries, timesteps,table_name,
        feature_columns, use_gpu, batch_size, learning_rate
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