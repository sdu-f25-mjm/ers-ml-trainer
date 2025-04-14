# core/model_training_cpu.py

import json
import logging
import os
import torch

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback

from datetime import datetime

from core.cache_environment import create_mariadb_cache_env
from core.utils import print_system_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# In this CPU-only version, we force the device to be CPU.
# Optionally, print system info to verify hardware.
print_system_info()
device = "cpu"
print(f"Using {device} device")


def export_model_to_torchscript(model_path, output_dir="best_model"):
    """
    Export trained stable-baselines3 RL model to TorchScript format for production deployment.
    This CPU-only version exports the model to run on CPU.
    """
    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    # Determine model type from filename and load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, device="cpu")
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, device="cpu")
    else:
        model = DQN.load(model_path, device="cpu")

    model.policy.set_training_mode(False)

    if hasattr(model.policy, 'q_net'):
        policy_net = model.policy.q_net  # For DQN
    elif hasattr(model.policy, 'actor'):
        policy_net = model.policy.actor  # For A2C/PPO
    else:
        policy_net = model.policy  # Fallback

    example_input = torch.zeros((1, model.observation_space.shape[0]), dtype=torch.float32)

    try:
        traced_model = torch.jit.trace(policy_net, example_input)
        output_path = os.path.join(output_dir, "policy.pt")
        traced_model.save(output_path)

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



def train_cache_model_cpu(db_url, algoritme="dqn", cache_size=10, max_queries=500,
                          feature_columns=None, timesteps=100000
                          , batch_size=None, learning_rate=None):
    """
    Train the cache model using CPU-only settings.
    """

    # Force CPU-only training; ignore GPU-related configuration.
    logger.info("Starting training on CPU.")

    algoritme = algoritme.lower()
    if algoritme not in ["dqn", "a2c", "ppo"]:
        logger.warning(f"Unknown algorithm '{algoritme}', falling back to DQN")
        algoritme = "dqn"

    if batch_size is None:
        batch_size = {
            "dqn": 64,
            "a2c": 32,
            "ppo": 64
        }[algoritme]

    if learning_rate is None:
        learning_rate = {
            "dqn": 0.0003,
            "a2c": 0.0007,
            "ppo": 0.0002
        }[algoritme]

    print("Using batch size:", batch_size)
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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Use CPU-optimized model architecture.
    policy_kwargs = {"net_arch": [128, 128]}

    model_params = {
        "verbose": 1,
        "device": "cpu",
        "policy_kwargs": policy_kwargs
    }

    from datetime import datetime
    start_time = datetime.now()

    if algoritme == "a2c":
        model = A2C(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=8,
            ent_coef=0.01,
            **model_params
        )
    elif algoritme == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=256,
            batch_size=batch_size,
            n_epochs=4,
            **model_params
        )
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=batch_size,
            target_update_interval=500,
            **model_params
        )

    logger.info(f"Starting training with {algoritme.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"models/database_cache_model_{algoritme}_{device}_{cache_size}_{timestamp}"
    model.save(model_name)
    logger.info(f"Model saved to: {model_name}")

    metadata = {
        "algorithm": algoritme,
        "device": "cpu",
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


def evaluate_cache_model_cpu(model_path, eval_steps=1000, db_url=None):
    """
    Evaluate the trained cache model using CPU.
    """
    logger = logging.getLogger(__name__)
    device = "cpu"
    logger.info(f"Evaluating model on {device.upper()}")

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
            logger.info(f"Step {step_count}, hit rate: {current_hit_rate:.4f}, "
                        f"avg inference: {avg_inference:.2f}ms")

    eval_time = (datetime.now() - eval_start).total_seconds()
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    env.close()

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