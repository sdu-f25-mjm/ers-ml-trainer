# core/model_training.py
import json
import logging
import os
from datetime import datetime

import numpy as np
# Remove torch and tensorflow imports from here

from core.gpu_utils import is_cuda_available, print_system_info, configure_gpu_environment


# Configure GPU environment if available and not forcing CPU
if is_cuda_available():
    configure_gpu_environment()  # Optionally pass a gpu_id if needed
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")


class CacheFeatureExtractor:
    """
    Custom feature extractor for cache observations with GPU optimization
    Lazily imported only when torch is available
    """

    def __new__(cls, *args, **kwargs):
        try:
            import torch.nn as nn
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            class TorchFeatureExtractor(BaseFeaturesExtractor):
                def __init__(self, observation_space, features_dim=128):
                    super().__init__(observation_space, features_dim)
                    n_input = int(np.prod(observation_space.shape))

                    self.network = nn.Sequential(
                        nn.Linear(n_input, 256),
                        nn.ReLU(),
                        nn.LayerNorm(256),
                        nn.Linear(256, features_dim),
                        nn.ReLU(),
                    )

                def forward(self, observations):
                    return self.network(observations)

            return TorchFeatureExtractor(*args, **kwargs)
        except ImportError:
            logging.getLogger(__name__).warning("PyTorch not available, using default feature extractor")
            return None


def export_model_to_torchscript(model_path, output_dir="best_model"):
    """
    Export trained stable-baselines3 RL model to TorchScript format for production deployment
    """
    logger = logging.getLogger(__name__)

    try:
        import torch
        from stable_baselines3 import DQN, A2C, PPO
    except ImportError:
        logger.error("PyTorch or stable-baselines3 not installed, cannot export model")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine model type from filename
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    else:
        model = DQN.load(model_path)

    # Set to evaluation mode
    model.policy.set_training_mode(False)

    # Extract the policy network
    if hasattr(model.policy, 'q_net'):
        policy_net = model.policy.q_net  # For DQN
    elif hasattr(model.policy, 'actor'):
        policy_net = model.policy.actor  # For A2C/PPO
    else:
        policy_net = model.policy  # Fallback

    # Create example input tensor matching observation space shape
    example_input = torch.zeros((1, model.observation_space.shape[0]),
                                dtype=torch.float32)

    try:
        # Export to TorchScript via tracing
        traced_model = torch.jit.trace(policy_net, example_input)

        # Save the model
        output_path = os.path.join(output_dir, "policy.pt")
        traced_model.save(output_path)

        # Save metadata for model deployment
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "original_model": model_path,
                "observation_space_shape": list(model.observation_space.shape),
                "action_space_size": model.action_space.n,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        logger.info(f"Model successfully exported to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def configure_gpu_environment(gpu_id=None):
    """Configure GPU environment for optimal training performance"""
    logger = logging.getLogger(__name__)

    # Print system information
    print_system_info()

    # Check CUDA availability without importing torch if possible
    if not is_cuda_available():
        logger.warning("CUDA not available. Using CPU for training.")
        return False

    try:
        # Only import torch when we know CUDA is available
        import torch
        import tensorflow as tf
    except ImportError:
        logger.warning("PyTorch or TensorFlow not installed. Using CPU for training.")
        return False

    # Log GPU information
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} CUDA device(s)")

    for i in range(gpu_count):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set GPU device if specified
    if gpu_id is not None and gpu_id < gpu_count:
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # Configure TensorFlow GPUs
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow GPU memory growth enabled")

                if gpu_id is not None and gpu_id < len(gpus):
                    tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            except RuntimeError as e:
                logger.error(f"TensorFlow GPU configuration error: {e}")
    except Exception as e:
        logger.warning(f"Failed to configure TensorFlow: {e}")

    # Optimize PyTorch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return True


def train_cache_model(db_url, algoritme="dqn", cache_size=10, max_queries=500,
                      feature_columns=None, timesteps=100000, optimeret_for_cpu=False,
                      gpu_id=None, batch_size=None, learning_rate=None):
    """Train a cache optimization RL model with GPU/CUDA support"""
    logger = logging.getLogger(__name__)

    # Import only when function is called
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not installed")

    try:
        from stable_baselines3 import DQN, A2C, PPO
        from stable_baselines3.common.callbacks import EvalCallback
        import torch
    except ImportError:
        logger.error("Required ML libraries not installed: stable-baselines3 or torch")
        return None

    # Configure directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("model_checkpoints", exist_ok=True)

    # Configure GPU if not optimizing for CPU
    has_gpu = False
    if not optimeret_for_cpu:
        has_gpu = configure_gpu_environment(gpu_id)

    device = "cuda" if has_gpu else "cpu"
    logger.info(f"Starting training with {algoritme.upper()} algorithm on {device.upper()}")

    # Import environment only when needed
    from core.cache_environment import create_mariadb_cache_env

    # Validate algorithm
    algoritme = algoritme.lower()
    if algoritme not in ["dqn", "a2c", "ppo"]:
        logger.warning(f"Unknown algorithm '{algoritme}', falling back to DQN")
        algoritme = "dqn"

    # Set optimal hyperparameters based on device
    if batch_size is None:
        batch_size = {
            "dqn": 128 if has_gpu else 64,
            "a2c": 64 if has_gpu else 32,
            "ppo": 256 if has_gpu else 64
        }[algoritme]

    if learning_rate is None:
        learning_rate = {
            "dqn": 0.0005 if has_gpu else 0.0003,
            "a2c": 0.001 if has_gpu else 0.0007,
            "ppo": 0.0003 if has_gpu else 0.0002
        }[algoritme]

    # Configure environments
    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries
    )

    eval_env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries
    )

    # Configure callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Configure network architecture
    if has_gpu:
        policy_kwargs = {
            "net_arch": [256, 256],
            "features_extractor_class": CacheFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128}
        }
    else:
        policy_kwargs = {"net_arch": [128, 128]}

    # Base model parameters
    model_params = {
        "verbose": 1,
        "device": device,
        "policy_kwargs": policy_kwargs
    }

    # Record starting time and resources
    start_time = datetime.now()
    if has_gpu:
        initial_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
        logger.info(f"Initial GPU memory: {initial_gpu_mem:.2f} MB")

    # Create model based on algorithm
    if algoritme == "a2c":
        model = A2C(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=16 if has_gpu else 8,
            ent_coef=0.01,
            **model_params
        )
    elif algoritme == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=512 if has_gpu else 256,
            batch_size=batch_size,
            n_epochs=10 if has_gpu else 4,
            **model_params
        )
    else:  # Default to DQN
        model = DQN(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            buffer_size=50000 if has_gpu else 10000,
            learning_starts=2000 if has_gpu else 1000,
            batch_size=batch_size,
            target_update_interval=1000 if has_gpu else 500,
            **model_params
        )

    # Train model
    logger.info(f"Starting training with {algoritme.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    # Calculate training time
    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    # Log final GPU memory if applicable
    if has_gpu:
        final_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
        logger.info(f"Final GPU memory: {final_gpu_mem:.2f} MB")
        # Free GPU memory
        torch.cuda.empty_cache()

    # Save model with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"models/database_cache_model_{algoritme}_{device}_{cache_size}_{timestamp}"
    model.save(model_name)
    logger.info(f"Model saved to: {model_name}")

    # Save model metadata
    metadata = {
        "algorithm": algoritme,
        "device": device,
        "cache_size": cache_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": training_time.total_seconds(),
        "timesteps": timesteps,
        "feature_columns": feature_columns or ["usage_frequency", "recency", "complexity"],
        "trained_at": timestamp
    }

    with open(f"{model_name}.meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up
    env.close()
    eval_env.close()

    return model_name


def evaluate_cache_model(model_path, eval_steps=1000, db_url=None, use_gpu=True):
    """Evaluate the trained model with GPU support"""
    logger = logging.getLogger(__name__)

    # Import libraries only when needed
    try:
        from stable_baselines3 import DQN, A2C, PPO
    except ImportError:
        logger.error("stable-baselines3 not installed, cannot evaluate model")
        return None

    # Import torch only if needed
    try:
        import torch
        has_gpu = torch.cuda.is_available() and use_gpu
    except ImportError:
        logger.warning("PyTorch not installed, using CPU for evaluation")
        has_gpu = False

    # Configure GPU if requested and available
    if has_gpu:
        configure_gpu_environment()
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Evaluating model on {device.upper()}")

    # Load model
    logger.info(f"Loading model from: {model_path}")
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, device=device)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, device=device)
    else:
        model = DQN.load(model_path, device=device)

    logger.info(f"Model loaded successfully on {device}!")

    # Use default database if not specified
    if db_url is None:
        db_url = "sqlite:///mock/mock_database.db"

    # Extract cache size from model path if available
    import re
    cache_size_match = re.search(r'cache_(\d+)', model_path)
    cache_size = int(cache_size_match.group(1)) if cache_size_match else 10

    # Import environment only when needed
    from core.cache_environment import create_mariadb_cache_env

    # Create evaluation environment
    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size
    )

    obs, _ = env.reset()

    # Evaluation variables
    total_reward = 0
    done = False
    step_count = 0
    saved_hit_rates = []
    hit_history = []
    rewards = []
    inference_times = []

    # Start evaluation
    eval_start = datetime.now()

    while not done and step_count < eval_steps:
        # Measure inference time
        pred_start = datetime.now()
        action, _ = model.predict(obs, deterministic=True)
        inference_time = (datetime.now() - pred_start).total_seconds() * 1000  # ms
        inference_times.append(inference_time)

        # Track cache hit/miss
        prev_hits = env.cache_hits
        obs, reward, terminated, truncated, info = env.step(action)
        is_hit = 1 if env.cache_hits > prev_hits else 0
        hit_history.append(is_hit)

        # Track rewards
        total_reward += reward
        rewards.append(reward)

        # Check if done
        done = terminated or truncated
        step_count += 1

        # Log progress
        if step_count % 100 == 0:
            current_hit_rate = info['cache_hit_rate']
            saved_hit_rates.append(current_hit_rate)
            avg_inference = sum(inference_times[-100:]) / min(100, len(inference_times[-100:]))
            logger.info(f"Step {step_count}, hit rate: {current_hit_rate:.4f}, "
                        f"avg inference: {avg_inference:.2f}ms")

    # Calculate evaluation time
    eval_time = (datetime.now() - eval_start).total_seconds()
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    # Clean up
    env.close()
    if has_gpu:
        torch.cuda.empty_cache()

    # Calculate moving average for hit rates
    window_size = 25
    moving_hit_rates = [
        np.mean(hit_history[max(0, i - window_size):i + 1])
        for i in range(len(hit_history))
    ]

    # Return results
    return {
        'hit_rates': saved_hit_rates,
        'hit_history': hit_history,
        'rewards': rewards,
        'moving_hit_rates': moving_hit_rates,
        'final_hit_rate': info['cache_hit_rate'],
        'total_reward': total_reward,
        'avg_inference_time_ms': sum(inference_times) / len(inference_times),
        'evaluation_time_seconds': eval_time,
        'device_used': device
    }