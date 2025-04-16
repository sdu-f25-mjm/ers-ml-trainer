# core/model_training_gpu.py
import json
import logging
import os
from datetime import datetime

import numpy as np

from core.utils import is_cuda_available, print_system_info, configure_gpu_environment

# Ensure a GPU is available – exit if not.
if not is_cuda_available():
    raise EnvironmentError("CUDA is not available. This version is optimized for GPU usage only.")

# Import torch now since we know CUDA is available
import torch
configure_gpu_environment()  # Optionally pass a gpu_id if needed
device = "cuda"

print(f"Using {device} device")


class CacheFeatureExtractor:
    """
    Custom feature extractor for cache observations with GPU optimization.
    This version assumes that torch is available.
    """

    def __new__(cls, *args, **kwargs):
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


def export_model_to_torchscript(model_path, output_dir="best_model"):
    """
    Export trained stable-baselines3 RL model to TorchScript format for production deployment.
    This function is optimized for GPU usage.
    """
    logger = logging.getLogger(__name__)

    try:
        from stable_baselines3 import DQN, A2C, PPO
    except ImportError:
        logger.error("stable-baselines3 not installed, cannot export model")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine model type from filename and load model
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
    example_input = torch.zeros((1, int(model.observation_space.shape[0])), dtype=torch.float32)

    try:
        # Export to TorchScript via tracing
        traced_model = torch.jit.trace(policy_net, example_input)

        # Save the model
        output_path = os.path.join(output_dir, "policy.pt")
        traced_model.save(output_path)

        # Save metadata for model deployment with conversion to native int types
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "original_model": model_path,
                "observation_space_shape": [int(x) for x in model.observation_space.shape],
                "action_space_size": int(model.action_space.n),
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        logger.info(f"Model successfully exported to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def configure_gpu_environment(gpu_id=None):
    logger = logging.getLogger(__name__)
    print_system_info()

    # Since this version is for GPU-only, exit if CUDA isn't available.
    if not is_cuda_available():
        logger.error("CUDA is not available. This version requires GPU.")
        raise EnvironmentError("CUDA is not available.")

    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not installed. Continuing with GPU configuration for PyTorch only.")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} CUDA device(s)")
    for i in range(gpu_count):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_id is not None and gpu_id < gpu_count:
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("TensorFlow GPU memory growth enabled")
            if gpu_id is not None and gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
    except Exception as e:
        logger.warning(f"Failed to configure TensorFlow: {e}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return True


def train_cache_model_gpu(db_url, algoritme="dqn", cache_size=10, max_queries=500, table_name="derived_data_cache_weights",
                          feature_columns=None, timesteps=100000, gpu_id=None,
                          batch_size=None, learning_rate=None):
    """
    Train the cache model using GPU-optimized settings.
    This function assumes that a GPU is available.
    """
    logger = logging.getLogger(__name__)

    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not installed, continuing with GPU training using PyTorch.")

    # Since GPU is required, verify availability and configure environment.
    if not is_cuda_available():
        logger.error("CUDA is not available. Training requires a GPU.")
        raise EnvironmentError("CUDA is not available.")

    # Import GPU-specific libraries
    import torch
    initial_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
    logger.info(f"Initial GPU memory: {initial_gpu_mem:.2f} MB")

    device = "cuda"
    logger.info(f"Starting training with {algoritme.upper()} algorithm on {device.upper()}")

    from core.cache_environment import create_mariadb_cache_env
    algoritme = algoritme.lower()
    if algoritme not in ["dqn", "a2c", "ppo"]:
        logger.warning(f"Unknown algorithm '{algoritme}', falling back to DQN")
        algoritme = "dqn"

    # Use GPU-optimized hyperparameters
    if batch_size is None:
        batch_size = {
            "dqn": 128,
            "a2c": 64,
            "ppo": 256
        }[algoritme]

    if learning_rate is None:
        learning_rate = {
            "dqn": 0.0005,
            "a2c": 0.001,
            "ppo": 0.0003
        }[algoritme]

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

    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    policy_kwargs = {
        "net_arch": [256, 256],
        "features_extractor_class": CacheFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128}
    }

    model_params = {
        "verbose": 1,
        "device": device,
        "policy_kwargs": policy_kwargs
    }

    from datetime import datetime
    start_time = datetime.now()
    initial_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
    logger.info(f"Initial GPU memory: {initial_gpu_mem:.2f} MB")

    if algoritme == "a2c":
        from stable_baselines3 import A2C
        model = A2C(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=16,
            ent_coef=0.01,
            **model_params
        )
    elif algoritme == "ppo":
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=512,
            batch_size=batch_size,
            n_epochs=10,
            **model_params
        )
    else:
        from stable_baselines3 import DQN
        model = DQN(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            buffer_size=50000,
            learning_starts=2000,
            batch_size=batch_size,
            target_update_interval=1000,
            **model_params
        )

    logger.info(f"Starting training with {algoritme.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    final_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
    logger.info(f"Final GPU memory: {final_gpu_mem:.2f} MB")
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"models/database_cache_model_{algoritme}_{device}_{cache_size}_{timestamp}"
    model.save(model_name)
    logger.info(f"Model saved to: {model_name}")

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

    env.close()
    eval_env.close()

    return model_name


def evaluate_cache_model_gpu(model_path, eval_steps=1000, db_url=None, use_gpu=True):
    """
    Evaluate the trained cache model using GPU.
    This function assumes GPU usage is available.
    """
    logger = logging.getLogger(__name__)

    import torch
    if not torch.cuda.is_available() or not use_gpu:
        raise EnvironmentError("CUDA is not available or GPU evaluation is disabled.")

    configured = configure_gpu_environment()
    device = "cuda" if configured else "cpu"
    logger.info(f"Evaluating model on {device.upper()}")

    from stable_baselines3 import PPO, A2C, DQN
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, device=device)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, device=device)
    else:
        model = DQN.load(model_path, device=device)

    logger.info(f"Model loaded successfully on {device}!")

    if db_url is None:
        db_url = "sqlite:///mock/mock_database.db"

    import re
    match = re.search(r'cache_(\d+)', model_path)
    cache_size = int(match.group(1)) if match else 10

    from core.cache_environment import create_mariadb_cache_env
    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size
    )

    obs, _ = env.reset()

    total_reward = 0
    done = False
    step_count = 0
    saved_hit_rates = []
    hit_history = []
    rewards = []
    inference_times = []

    from datetime import datetime
    eval_start = datetime.now()

    while not done and step_count < eval_steps:
        pred_start = datetime.now()
        action, _ = model.predict(obs, deterministic=True)
        inference_times.append((datetime.now() - pred_start).total_seconds() * 1000)

        prev_hits = env.cache_hits
        obs, reward, terminated, truncated, info = env.step(action)
        hit_history.append(1 if env.cache_hits > prev_hits else 0)

        total_reward += reward
        rewards.append(reward)

        done = terminated or truncated
        step_count += 1

        if step_count % 100 == 0:
            current_hit_rate = info['cache_hit_rate']
            saved_hit_rates.append(current_hit_rate)
            avg_inference = sum(inference_times[-100:]) / len(inference_times[-100:])
            logger.info(f"Step {step_count}, hit rate: {current_hit_rate:.4f}, avg inference: {avg_inference:.2f}ms")

    eval_time = (datetime.now() - eval_start).total_seconds()
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    env.close()
    torch.cuda.empty_cache()

    import numpy as np
    window_size = 25
    moving_hit_rates = [
        np.mean(hit_history[max(0, i - window_size):i + 1])
        for i in range(len(hit_history))
    ]

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
