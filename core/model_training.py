# core/model_training_utils.py


import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from api.app_utils import AlgorithmEnum
from core.cache_environment import create_mariadb_cache_env
from core.utils import is_cuda_available

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
    """Configure GPU environment if available"""

    if not is_cuda_available():
        logger.warning("CUDA is not available. Using CPU only.")
        return False

    try:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s)")
        for i in gpu_count:
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Configure additional GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        return True
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
        return False


def get_device(use_gpu=False):
    """Determine which device to use (CPU or GPU)"""
    if use_gpu:
        if is_cuda_available():
            configure_gpu_environment()
            return "cuda"
        else:
            logger.warning("CUDA not available. Falling back to CPU.")
            return "cpu"
    return "cpu"


def create_feature_extractor(device="cpu"):
    """Create appropriate feature extractor based on device"""

    class CacheFeatureExtractor:
        """Custom feature extractor for cache observations."""

        def __new__(cls, *args, **kwargs):
            import torch.nn as nn
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

            if device == "cuda":
                # GPU-optimized extractor
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
            else:
                # CPU-optimized extractor (simpler architecture)
                class TorchFeatureExtractor(BaseFeaturesExtractor):
                    def __init__(self, observation_space, features_dim=64):
                        super().__init__(observation_space, features_dim)
                        n_input = int(np.prod(observation_space.shape))
                        self.network = nn.Sequential(
                            nn.Linear(n_input, 128),
                            nn.ReLU(),
                            nn.Linear(128, features_dim),
                            nn.ReLU(),
                        )

                    def forward(self, observations):
                        return self.network(observations)

            return TorchFeatureExtractor(*args, **kwargs)

    return CacheFeatureExtractor


def get_hyperparameters(algorithm, device):
    """Get optimal hyperparameters based on algorithm and device"""
    base_params = {
        "dqn": {
            "cpu": {
                "learning_rate": 0.0003,
                "batch_size": 64,
                "buffer_size": 10000,
                "learning_starts": 1000,
                "target_update_interval": 500,
                "net_arch": [128, 128]
            },
            "cuda": {
                "learning_rate": 0.0005,
                "batch_size": 128,
                "buffer_size": 50000,
                "learning_starts": 2000,
                "target_update_interval": 1000,
                "net_arch": [256, 256]
            }
        },
        "a2c": {
            "cpu": {
                "learning_rate": 0.0007,
                "batch_size": 32,
                "n_steps": 8,
                "ent_coef": 0.01,
                "net_arch": [128, 128]
            },
            "cuda": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "n_steps": 16,
                "ent_coef": 0.01,
                "net_arch": [256, 256]
            }
        },
        "ppo": {
            "cpu": {
                "learning_rate": 0.0002,
                "batch_size": 64,
                "n_steps": 256,
                "n_epochs": 4,
                "net_arch": [128, 128]
            },
            "cuda": {
                "learning_rate": 0.0003,
                "batch_size": 256,
                "n_steps": 512,
                "n_epochs": 10,
                "net_arch": [256, 256]
            }
        }
    }

    return base_params[algorithm][device]


def export_model_to_torchscript(model_path, output_dir="best_model"):
    """
    Export trained stable-baselines3 RL model to TorchScript format for production deployment.
    Works for both CPU and GPU models.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Try first on CPU to avoid device mismatches
        logger.info(f"Loading model {model_path} to CPU for export")
        if "ppo" in model_path.lower():
            model = PPO.load(model_path, device="cpu")
            model_type = "ppo"
        elif "a2c" in model_path.lower():
            model = A2C.load(model_path, device="cpu")
            model_type = "a2c"
        else:
            model = DQN.load(model_path, device="cpu")
            model_type = "dqn"

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
        example_input = torch.zeros((1, int(model.observation_space.shape[0])),
                                    dtype=torch.float32, device="cpu")

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
                "observation_space_shape": [int(x) for x in model.observation_space.shape],
                "action_space_size": int(model.action_space.n),
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": "cpu",
                "model_type": model_type
            }, f, indent=2)

        logger.info(f"Model successfully exported to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def train_cache_model(
        db_url,
        algorithm=AlgorithmEnum,
        cache_size=10,
        max_queries=500,
        table_name=None,
        feature_columns=None,
        timesteps=100000,
        batch_size=None,
        learning_rate=None,
        use_gpu=False,
        cache_weights=None
):
    """
    Train a cache model using either CPU or GPU based on availability and preferences.
    """
    # Determine device
    device = get_device(use_gpu)

    # Print system information
    if device == "cuda":
        configure_gpu_environment()

    # Normalize algorithm name
    algorithm = algorithm.lower()
    if algorithm not in ["dqn", "a2c", "ppo"]:
        logger.warning(f"Unknown algorithm '{algorithm}', falling back to DQN")
        algorithm = "dqn"

    # Get optimal hyperparameters for this device
    params = get_hyperparameters(algorithm, device)

    # Override with user-specified values if provided
    if batch_size is not None:
        params["batch_size"] = batch_size
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    logger.info(f"Using batch size: {params['batch_size']}")

    # Create training environment
    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights  # <-- pass through
    )

    # Create evaluation environment
    eval_env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights  # <-- pass through
    )
    eval_env = Monitor(eval_env)

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Configure policy kwargs with appropriate feature extractor
    policy_kwargs = {"net_arch": params["net_arch"]}
    if device == "cuda":
        feature_extractor = create_feature_extractor(device)
        if feature_extractor:
            policy_kwargs.update({
                "features_extractor_class": feature_extractor,
                "features_extractor_kwargs": {"features_dim": 128}
            })

    # Set common model parameters
    model_params = {
        "verbose": 1,
        "device": device,
        "policy_kwargs": policy_kwargs
    }

    # Track initial resource usage
    start_time = datetime.now()
    if device == "cuda":
        initial_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
        logger.info(f"Initial GPU memory: {initial_gpu_mem:.2f} MB")

    # Create the appropriate model based on algorithm
    if algorithm == "a2c":
        model = A2C(
            "MlpPolicy", env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            ent_coef=params["ent_coef"],
            **model_params
        )
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            **model_params
        )
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            target_update_interval=params["target_update_interval"],
            **model_params
        )

    # Train model
    logger.info(f"Starting training with {algorithm.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    logger.info("Training loop finished.")  # <--- Add this line

    # Log training completion
    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    # Clean up GPU memory if applicable
    if device == "cuda":
        final_gpu_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
        logger.info(f"Final GPU memory: {final_gpu_mem:.2f} MB")
        torch.cuda.empty_cache()

    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"models/database_cache_model_{algorithm}_{device}_{cache_size}_{timestamp}"
    model.save(model_name)
    logger.info(f"Model saved to: {model_name}")

    # Save metadata
    metadata = {
        "algorithm": algorithm,
        "device": device,
        "cache_size": cache_size,
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "training_time_seconds": training_time.total_seconds(),
        "timesteps": timesteps,
        "feature_columns": feature_columns,
        "trained_at": timestamp,
        "model_type": algorithm  # <-- Add model_type to metadata
    }

    with open(f"{model_name}.meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save to database with model_type
    # If you have a DB connection object (conn), pass model_type to save_best_model
    # Example:
    # save_best_model(model, model_name, conn, description="Best model", model_type=algorithm)

    # Close environments
    env.close()
    eval_env.close()

    return model_name


def safe_cuda_initialization():
    """Test if CUDA is fully functional"""
    import torch
    import logging

    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return False

    try:
        test_tensor = torch.zeros(10, 10).cuda()
        test_result = test_tensor + 1
        torch.cuda.synchronize()
        logger.info("CUDA initialization successful")
        return True
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        return False


def evaluate_cache_model(model_path, eval_steps=1000, db_url=None, use_gpu=False,
                         table_name="cache_metrics"):
    """Evaluate a trained cache model with robust GPU handling using Gymnasium API."""
    import torch
    import json
    import os

    # Check if GPU should and can be used
    use_cuda = use_gpu and safe_cuda_initialization()
    device = "cuda" if use_cuda else "cpu"

    logger.info(f"Evaluating model on {device.upper()}")

    try:
        # Load feature columns from model metadata if not provided
        metadata_path = f"{model_path.replace('.zip', '')}.meta.json"
        feature_columns = None
        cache_size_mb = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                feature_columns = metadata.get('feature_columns')
                # prefer cache_size_mb over legacy cache_size
                cache_size_mb = metadata.get('cache_size_mb', metadata.get('cache_size'))
                logger.info(f"Loaded feature columns from metadata: {feature_columns}")
                logger.info(f"Loaded cache_size_mb from metadata: {cache_size_mb}")
            except Exception as e:
                logger.warning(f"Failed to load feature columns from metadata: {e}")

        # Default feature columns if still None
        if not feature_columns:
            from api.app_utils import CacheTableEnum
            feature_columns = [e.value for e in CacheTableEnum]
            logger.info(f"Using default feature columns: {feature_columns}")

        # Default MB size if not found
        if cache_size_mb is None:
            cache_size_mb = 10
            logger.info(f"Using default cache_size_mb: {cache_size_mb} MB")

        # Load model with proper device
        if "ppo" in model_path.lower():
            model = PPO.load(model_path, device=device)
        elif "a2c" in model_path.lower():
            model = A2C.load(model_path, device=device)
        else:
            model = DQN.load(model_path, device=device)

        logger.info(f"Model loaded successfully on {device}!")

        # Create environment matching training (use cache_size_mb)
        env = create_mariadb_cache_env(
            db_url=db_url,
            table_name=table_name,
            feature_columns=feature_columns,
            cache_size_mb=cache_size_mb,
            max_queries=eval_steps
        )

        # --- SHAPE CHECK: Ensure observation space matches model ---
        env_obs_shape = env.observation_space.shape
        model_obs_shape = model.observation_space.shape
        logger.info(f"Model expects observation shape: {model_obs_shape}")
        logger.info(f"Environment provides observation shape: {env_obs_shape}")
        logger.info(f"Feature columns used: {feature_columns}")
        logger.info(f"Cache size used: {cache_size_mb} MB")

        if env_obs_shape != model_obs_shape:
            error_msg = (
                f"Error: Observation shape mismatch. "
                f"Model expects {model_obs_shape}, but environment provides {env_obs_shape}. "
                f"Check that feature_columns and cache_size match between training and evaluation. "
                f"Model feature_columns: {feature_columns}, cache_size_mb: {cache_size_mb}"
            )
            logger.error(f"Model evaluation failed: {error_msg}")
            return {"error": error_msg, "success": False}

        # --- GYMNASIUM EVALUATION LOOP ---
        obs, _ = env.reset()  # Gymnasium reset returns (obs, info)
        total_reward = 0
        done = False
        step_count = 0
        saved_hit_rates = []
        hit_history = []
        rewards = []
        inference_times = []
        step_reasoning = []  # New: reasoning per step
        in_cache = []  # New: is item in cache after action

        # Add: Track which URLs (cache_names) are hit during evaluation
        urls_hit = []

        eval_start = datetime.now()

        while not done and step_count < eval_steps:
            # Model predicts action given observation
            pred_start = datetime.now()
            action, _ = model.predict(obs, deterministic=True)
            inference_times.append((datetime.now() - pred_start).total_seconds() * 1000)

            # Step through the environment (Gymnasium API)
            prev_hits = env.cache_hits
            prev_cache = list(env.cache)  # Copy for reasoning
            obs, reward, terminated, truncated, info = env.step(action)
            hit = env.cache_hits > prev_hits
            hit_history.append(1 if hit else 0)
            total_reward += reward
            rewards.append(reward)
            done = terminated or truncated

            # Reasoning: why this score/result
            if hit:
                reason = f"Cache HIT: item was already in cache before action."
            else:
                if action == 1:
                    reason = f"Cache MISS: item not in cache, added to cache."
                else:
                    reason = f"Cache MISS: item not in cache, not added (action=0)."
            step_reasoning.append(reason)

            # Is the current item in cache after action?
            # The current query index points to the next query, so check previous
            current_idx = (env.current_query_idx - 1) % len(env.data)
            current_item = env.data.iloc[current_idx]
            is_in_cache = any(
                all(current_item[col] == cached_item[col] for col in env.feature_columns)
                for cached_item in env.cache
            )
            in_cache.append(is_in_cache)

            # Track cache_name (URL) for each step
            cache_name = current_item.get("cache_name") if hasattr(current_item, "get") else current_item["cache_name"]
            urls_hit.append(cache_name)

            step_count += 1

            # Optionally log progress
            if step_count % 100 == 0:
                current_hit_rate = info['cache_hit_rate']
                saved_hit_rates.append(current_hit_rate)
                avg_inference = sum(inference_times[-100:]) / len(inference_times[-100:])
                logger.info(f"Step {step_count}, hit rate: {current_hit_rate:.4f}, "
                            f"avg inference: {avg_inference:.2f}ms")

        # --- END GYMNASIUM EVALUATION LOOP ---

        # Calculate evaluation time
        eval_time = (datetime.now() - eval_start).total_seconds()
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

        # Clean up
        env.close()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Calculate moving hit rate
        window_size = 25
        moving_hit_rates = [
            np.mean(hit_history[max(0, i - window_size):i + 1])
            for i in range(len(hit_history))
        ]

        # --- Add diagnostics for visualization ---
        # These are placeholders; replace with real values if you log them during training/evaluation
        episode_rewards = [sum(rewards)] if rewards else []
        losses = []
        q_values_history = []
        entropies = []

        # Return evaluation metrics, now with reasoning and in_cache and diagnostics
        return {
            'hit_rates': saved_hit_rates,
            'hit_history': hit_history,
            'rewards': rewards,
            'moving_hit_rates': moving_hit_rates,
            'final_hit_rate': info['cache_hit_rate'],
            'total_reward': total_reward,
            'avg_inference_time_ms': sum(inference_times) / len(inference_times),
            'evaluation_time_seconds': eval_time,
            'device_used': device,
            'step_reasoning': step_reasoning,
            'in_cache': in_cache,
            'urls_hit': urls_hit,
            # Diagnostics for visualization
            'episode_rewards': episode_rewards,
            'losses': losses,
            'q_values_history': q_values_history,
            'entropies': entropies
        }
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")

        # If failed on GPU, retry on CPU
        if device == "cuda":
            logger.info("Retrying evaluation on CPU")
            return evaluate_cache_model(model_path, eval_steps, db_url, use_gpu=False, table_name=table_name)

        return {"error": str(e), "success": False}



