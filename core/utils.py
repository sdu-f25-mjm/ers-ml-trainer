# core/utils.py
import logging
import os
import subprocess

import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def configure_gpu_environment():
    """
    Configure and return the device to be used for computations.
    If CUDA is available, returns 'cuda', otherwise returns 'cpu'.
    """
    if is_cuda_available():
        device = 'cuda'
        logger.info("CUDA is available. Configuring GPU environment.")
    else:
        device = 'cpu'
        logger.info("CUDA is not available. Using CPU.")

    # Optionally, set environment variables or torch settings here.
    os.environ["DEVICE"] = device
    return device


def is_cuda_available():
    """Check if CUDA is available without importing torch if possible"""
    try:
        # Try to detect NVIDIA GPUs using subprocess first
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2
        )
        count = int(result.stdout.strip())
        return count > 0
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, TimeoutError):
        # If nvidia-smi fails, try importing torch
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def get_gpu_info():
    """Get detailed information about available GPUs"""

    # First check if CUDA is available without importing torch
    if not is_cuda_available():
        logger.info("No CUDA-capable GPUs detected")
        return None

    gpu_info = []
    # Try using nvidia-smi first (doesn't require torch)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )

        for line in result.stdout.strip().split("\n"):
            if line:
                idx, name, total_mem, used_mem, temp = line.split(", ")
                gpu_info.append({
                    "index": int(idx),
                    "name": name,
                    "total_memory_mb": float(total_mem),
                    "used_memory_mb": float(used_mem),
                    "temperature_c": float(temp)
                })

        return gpu_info  # Return early if nvidia-smi worked
    except (subprocess.SubprocessError, FileNotFoundError):
        pass  # Continue to torch fallback

    # Only import torch if nvidia-smi failed
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Fallback to basic torch info
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024)
            })

        return gpu_info
    except ImportError:
        logger.info("PyTorch not installed, cannot get GPU information")
        return None


def print_system_info():
    """Print system information including CPU, RAM and GPU details"""

    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    logger.info(f"CPU: {cpu_count} cores, {cpu_threads} threads")
    if cpu_freq:
        logger.info(f"CPU Frequency: {cpu_freq.current:.2f}MHz")

    # RAM info
    ram = psutil.virtual_memory()
    logger.info(f"RAM: {ram.total / (1024 ** 3):.2f}GB total, {ram.available / (1024 ** 3):.2f}GB available")

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        for gpu in gpu_info:
            logger.info(f"GPU {gpu['index']}: {gpu['name']}")
            if "total_memory_mb" in gpu:
                logger.info(f"  Memory: {gpu['used_memory_mb']:.0f}MB used / {gpu['total_memory_mb']:.0f}MB total")
            if "temperature_c" in gpu:
                logger.info(f"  Temperature: {gpu['temperature_c']}°C")


def build_db_url():
    """
    Build the database URL from environment variables.
    Defaults are provided for local development.
    """
    driver = os.getenv("DB_DRIVER", "mysql+mysqlconnector")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    dbname = os.getenv("DB_NAME", "cache_db")
    user = os.getenv("DB_USER", "cacheuser")
    password = os.getenv("DB_PASSWORD", "cachepass")
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"


def build_custom_db_url(driver, host, port, dbname, user, password):
    """
    Build the database URL from environment variables.
    Defaults are provided for local development.
    """
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"


def list_available_models(models_dir="models"):
    """
    Find and list all available trained models in the models directory.

    Args:
        models_dir: Directory where models are saved

    Returns:
        List of dictionaries containing model information
    """
    import os
    import re
    import json
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory '{models_dir}' does not exist")
        return []

    models = []
    model_pattern = re.compile(r"database_cache_model_(\w+)_(\w+)_(\d+)_(\d{8}_\d{6})$")

    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)

        # Skip directories and non-zip files (SB3 models are zip files)
        if os.path.isdir(filepath) or not filepath.endswith(".zip"):
            continue

        match = model_pattern.match(filename.replace(".zip", ""))
        if not match:
            continue

        algorithm, device, cache_size, timestamp = match.groups()

        # Try to load metadata if available
        metadata = {}
        metadata_path = f"{filepath.replace('.zip', '')}.meta.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                pass

        models.append({
            "path": filepath,
            "algorithm": algorithm,
            "device": device,
            "cache_size": int(cache_size),
            "timestamp": timestamp,
            "created_at": metadata.get("trained_at", timestamp),
            "metadata": metadata
        })

    # Sort by timestamp, newest first
    return sorted(models, key=lambda x: x["timestamp"], reverse=True)
