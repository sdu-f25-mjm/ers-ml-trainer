import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from api.app_utils import AlgorithmEnum
from core.cache_environment import create_mariadb_cache_env
from core.utils import is_cuda_available

# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/application.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def configure_gpu_environment() -> bool:
    """
    Configure GPU environment if available.
    Returns True if GPU was configured successfully, False otherwise.
    """
    if not is_cuda_available():
        logger.warning("CUDA is not available. Using CPU only.")
        return False

    try:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA device(s)")
        for idx in range(gpu_count):
            name = torch.cuda.get_device_name(idx)
            logger.info(f"  GPU {idx}: {name}")

        # Enable TF32 where available
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        return True
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
        return False


def get_device(use_gpu: bool = False) -> str:
    """
    Determine which device to use (CPU or GPU).
    """
    if use_gpu and is_cuda_available():
        configure_gpu_environment()
        return "cuda"
    if use_gpu:
        logger.warning("Requested GPU but CUDA is not available; falling back to CPU.")
    return "cpu"


def create_feature_extractor(device: str = "cpu"):
    """
    Factory for a custom SB3 feature extractor, optimized for CPU vs GPU.
    """
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch.nn as nn

    class CacheFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = None):
            if features_dim is None:
                features_dim = 128 if device == "cuda" else 64
            super().__init__(observation_space, features_dim)
            input_dim = int(np.prod(observation_space.shape))
            if device == "cuda":
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.LayerNorm(256),
                    nn.Linear(256, features_dim),
                    nn.ReLU(),
                )
            else:
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, features_dim),
                    nn.ReLU(),
                )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.network(observations)

    return CacheFeatureExtractor


def get_hyperparameters(
    algorithm: str, device: str
) -> Dict[str, Union[int, float, List[int]]]:
    """
    Fetch base hyperparameters for a given algorithm+device.
    """
    base_params = {
        "dqn": {
            "cpu": {
                "learning_rate": 3e-4,
                "batch_size": 64,
                "buffer_size": 10_000,
                "learning_starts": 1_000,
                "target_update_interval": 500,
                "net_arch": [128, 128],
                "ent_coef": 0.01,        # encourage exploration
                "vf_coef": 0.1,         # lower value loss coefficient
            },
            "cuda": {
                "learning_rate": 5e-4,
                "batch_size": 128,
                "buffer_size": 50_000,
                "learning_starts": 2_000,
                "target_update_interval": 1_000,
                "net_arch": [256, 256],
                "ent_coef": 0.01,
                "vf_coef": 0.1,
            },
        },
        "a2c": {
            "cpu": {
                "learning_rate": 7e-4,
                "batch_size": 32,
                "n_steps": 8,
                "ent_coef": 0.01,
                "vf_coef": 0.1,
                "net_arch": [128, 128],
            },
            "cuda": {
                "learning_rate": 1e-3,
                "batch_size": 64,
                "n_steps": 16,
                "ent_coef": 0.01,
                "vf_coef": 0.1,
                "net_arch": [256, 256],
            },
        },
        "ppo": {
            "cpu": {
                "learning_rate": 1e-4,  # lower lr for stability
                "batch_size": 64,
                "n_steps": 256,
                "n_epochs": 4,
                "ent_coef": 0.02,       # boost entropy
                "vf_coef": 0.1,         # lower value loss coef
                "clip_range": 0.2,
                "net_arch": [128, 128],
            },
            "cuda": {
                "learning_rate": 2e-4,
                "batch_size": 256,
                "n_steps": 512,
                "n_epochs": 10,
                "ent_coef": 0.02,
                "vf_coef": 0.1,
                "clip_range": 0.2,
                "net_arch": [256, 256],
            },
        },
    }
    return base_params[algorithm][device]


def export_model_to_torchscript(model_path: str, output_dir: str = "best_model") -> str:
    """
    Export a trained SB3 model policy to TorchScript.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        if "ppo" in model_path.lower():
            model = PPO.load(model_path, device="cpu")
        elif "a2c" in model_path.lower():
            model = A2C.load(model_path, device="cpu")
        else:
            model = DQN.load(model_path, device="cpu")

        model.policy.set_training_mode(False)
        if hasattr(model.policy, "q_net"):
            policy_net = model.policy.q_net
        elif hasattr(model.policy, "actor"):
            policy_net = model.policy.actor
        else:
            policy_net = model.policy

        example = torch.zeros((1, int(model.observation_space.shape[0])), dtype=torch.float32)
        traced = torch.jit.trace(policy_net, example)

        output_path = os.path.join(output_dir, "policy.pt")
        traced.save(output_path)

        meta = {
            "original_model": model_path,
            "obs_shape": list(model.observation_space.shape),
            "action_size": int(model.action_space.n),
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "device": "cpu",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"TorchScript export successful: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def train_cache_model(
    db_url: str,
    algorithm: Union[AlgorithmEnum, str] = AlgorithmEnum.dqn,
    cache_size: int = 10,
    max_queries: int = 500,
    table_name: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    timesteps: int = 100_000,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_gpu: bool = False,
    cache_weights: Optional[List[str]] = None,
) -> str:
    """
    Train a database-cache RL model with SB3.
    """
    assert cache_size > 0, "cache_size must be positive"
    logger.info(f"Training: algo={algorithm}, cache_size={cache_size}, timesteps={timesteps}")

    device = get_device(use_gpu)
    algo_str = (
        algorithm.value if isinstance(algorithm, AlgorithmEnum) else str(algorithm)
    ).lower()
    if algo_str not in ("dqn", "a2c", "ppo"):
        logger.warning(f"Unknown algorithm '{algo_str}', defaulting to 'dqn'")
        algo_str = "dqn"

    params = get_hyperparameters(algo_str, device)
    if batch_size:
        params["batch_size"] = batch_size
    if learning_rate:
        params["learning_rate"] = learning_rate
    logger.info(f"Hyperparameters: {params}")

    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights,
    )
    eval_env = Monitor(
        create_mariadb_cache_env(
            db_url=db_url,
            cache_size=cache_size,
            feature_columns=feature_columns,
            max_queries=max_queries,
            table_name=table_name,
            cache_weights=cache_weights,
        )
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=max_queries,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    policy_kwargs = {"net_arch": params["net_arch"]}
    if device == "cuda":
        policy_kwargs.update({
            "features_extractor_class": create_feature_extractor(device),
            "features_extractor_kwargs": {"features_dim": 128},
        })

    # common args
    common_kwargs = {
        "verbose": 1,
        "device": device,
        "policy_kwargs": policy_kwargs,
    }

    # instantiate model with entropy & value coeffs, advantage normalization
    if algo_str == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            **common_kwargs
        )
    elif algo_str == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            clip_range=params.get("clip_range", 0.2),
            normalize_advantage=True,
            **common_kwargs
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            target_update_interval=params["target_update_interval"],
            **common_kwargs
        )

    # Train
    start = datetime.utcnow()
    logger.info(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Training completed in {elapsed:.1f}s")

    if device == "cuda":
        torch.cuda.empty_cache()

    # Save model & metadata
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/cache_model_{algo_str}_{device}_{cache_size}_{ts}"
    model.save(save_path)
    meta = {
        "algorithm": algo_str,
        "device": device,
        "cache_size": cache_size,
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "timesteps": timesteps,
        "feature_columns": feature_columns,
        "trained_at": ts,
    }
    with open(save_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    env.close()
    eval_env.close()
    return save_path


def evaluate_cache_model(
    model_path: str,
    eval_steps: int = 1_000,
    db_url: Optional[str] = None,
    use_gpu: bool = False,
    table_name: str = "cache_metrics",
) -> Dict[str, Any]:
    """
    Run a full evaluation of a trained cache model, collecting:
    - hit_history, rewards, inference_times,
    - actions, cache_occupancy, step_reasoning, in_cache, urls, moving rates.
    """
    from api.app_utils import CacheTableEnum

    use_cuda = use_gpu and safe_cuda_initialization()
    device = "cuda" if use_cuda else "cpu"
    logger.info(f"Evaluating on device: {device}")

    # load metadata
    meta_path = model_path + ".meta.json"
    feature_columns = None
    cache_size = None
    if os.path.exists(meta_path):
        try:
            info = json.load(open(meta_path, encoding="utf-8"))
            feature_columns = info.get("feature_columns")
            cache_size = info.get("cache_size")
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")

    if not feature_columns:
        feature_columns = [e.value for e in CacheTableEnum]
    if cache_size is None:
        cache_size = 10

    # load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, device=device)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, device=device)
    else:
        model = DQN.load(model_path, device=device)
    logger.info("Model loaded for evaluation")

    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=eval_steps,
        table_name=table_name,
    )

    if env.observation_space.shape != model.observation_space.shape:
        msg = (f"Shape mismatch: env {env.observation_space.shape} vs model "
               f"{model.observation_space.shape}")
        logger.error(msg)
        return {"error": msg, "success": False}

    obs, _ = env.reset()
    total_reward = 0.0

    hits: List[int] = []
    rewards: List[float] = []
    inference_times: List[float] = []
    actions: List[int] = []
    occupancy: List[int] = []
    reasoning: List[str] = []
    in_cache: List[bool] = []
    urls: List[Any] = []

    start = datetime.utcnow()
    for step in range(eval_steps):
        t0 = datetime.utcnow()
        action, _ = model.predict(obs, deterministic=True)
        duration_ms = (datetime.utcnow() - t0).total_seconds() * 1000
        inference_times.append(duration_ms)
        actions.append(int(action))

        prev_hits = env.cache_hits
        obs, reward, done, _, info = env.step(action)
        occupancy.append(len(env.cache))
        hit = env.cache_hits > prev_hits

        hits.append(int(hit))
        rewards.append(reward)
        total_reward += reward

        # reasoning
        if hit:
            reasoning.append("HIT: in cache.")
        elif action == 1:
            reasoning.append("MISS→cached")
        else:
            reasoning.append("MISS: no cache")

        # in_cache flag
        idx = (env.current_query_idx - 1) % len(env.data)
        current = env.data.iloc[idx]
        still = any(all(current[c] == itm[c] for c in env.feature_columns) for itm in env.cache)
        in_cache.append(still)

        urls.append(current.get("cache_name", None))
        if done:
            break

    eval_time = (datetime.utcnow() - start).total_seconds()
    logger.info(f"Evaluation done in {eval_time:.2f}s")

    env.close()
    if device == "cuda":
        torch.cuda.empty_cache()

    moving = [np.mean(hits[max(0, i - 24): i + 1]) for i in range(len(hits))]

    return {
        "hit_history": hits,
        "rewards": rewards,
        "moving_hit_rates": moving,
        "final_hit_rate": info.get("cache_hit_rate"),
        "total_reward": total_reward,
        "avg_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "evaluation_time_seconds": eval_time,
        "actions": actions,
        "cache_occupancy": occupancy,
        "inference_times": inference_times,
        "step_reasoning": reasoning,
        "in_cache": in_cache,
        "urls": urls,
        "success": True,
    }