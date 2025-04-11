# core/gpu_utils.py
import logging
import torch
import psutil
import subprocess


def get_gpu_info():
    """Get detailed information about available GPUs"""
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        logger.info("No CUDA-capable GPUs detected")
        return None

    gpu_info = []
    try:
        # Try using nvidia-smi through subprocess
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

    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback to basic torch info
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024)
            })

    return gpu_info


def print_system_info():
    """Print system information including CPU, RAM and GPU details"""
    logger = logging.getLogger(__name__)

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