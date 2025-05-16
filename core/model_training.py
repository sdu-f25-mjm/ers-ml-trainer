import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
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
    """Configure GPU if available."""
    if not is_cuda_available():
        logger.warning("CUDA not available, using CPU.")
        return False
    try:
        for idx in range(torch.cuda.device_count()):
            logger.info(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception as e:
        logger.error(f"GPU configuration error: {e}")
        return False


def get_device(use_gpu: bool = False) -> str:
    """Return 'cuda' or 'cpu'."""
    if use_gpu and is_cuda_available():
        configure_gpu_environment()
        return "cuda"
    if use_gpu:
        logger.warning("Requested GPU but unavailable; falling back to CPU.")
    return "cpu"


def create_feature_extractor(device: str = "cpu"):
    """Custom SB3 feature extractor."""
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch.nn as nn

    class CacheFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, obs_space, features_dim: int = None):
            features_dim = features_dim or (128 if device == "cuda" else 64)
            super().__init__(obs_space, features_dim)
            in_dim = int(np.prod(obs_space.shape))
            if device == "cuda":
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 256), nn.ReLU(),
                    nn.LayerNorm(256),
                    nn.Linear(256, features_dim), nn.ReLU(),
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128), nn.ReLU(),
                    nn.Linear(128, features_dim), nn.ReLU(),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return CacheFeatureExtractor

def get_hyperparameters(alg: str, device: str) -> Dict[str, Any]:
    """Base hyperparameters by algorithm & device."""
    params = {
        "dqn": {
            "cpu":   {"learning_rate":3e-4, "batch_size":64,  "buffer_size":10_000,
                      "learning_starts":1_000, "target_update_interval":500,
                      "net_arch":[128,128], "ent_coef":0.05, "vf_coef":0.1},
            "cuda":  {"learning_rate":5e-4, "batch_size":128, "buffer_size":50_000,
                      "learning_starts":2_000, "target_update_interval":1_000,
                      "net_arch":[256,256], "ent_coef":0.05, "vf_coef":0.1},
        },
        "a2c": {
            "cpu":   {"learning_rate":7e-4, "batch_size":32, "n_steps":8,
                      "ent_coef":0.01, "vf_coef":0.1, "net_arch":[128,128]},
            "cuda":  {"learning_rate":1e-3, "batch_size":64, "n_steps":16,
                      "ent_coef":0.01, "vf_coef":0.1, "net_arch":[256,256]},
        },
        "ppo": {
            "cpu":   {"learning_rate":1e-4, "batch_size":64,  "n_steps":256,
                      "n_epochs":4, "ent_coef":0.05, "vf_coef":0.1,
                      "clip_range":0.2, "max_grad_norm":10.0,
                      "net_arch":[128,128]},
            "cuda":  {"learning_rate":2e-4, "batch_size":256, "n_steps":512,
                      "n_epochs":10, "ent_coef":0.05, "vf_coef":0.1,
                      "clip_range":0.2, "max_grad_norm":10.0,
                      "net_arch":[256,256]},
        },
    }
    return params[alg][device]


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

        logger.debug(f"TorchScript export successful: {output_path}")
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
    logger.info(f"ENTER train_cache_model: algo={algorithm}, cache_size={cache_size}, timesteps={timesteps}")

    # 1) device & algorithm
    device = get_device(use_gpu)
    algo_str = (algorithm.value if isinstance(algorithm, AlgorithmEnum) else str(algorithm)).lower()
    if algo_str not in ("dqn", "a2c", "ppo"):
        logger.warning(f"Invalid algo '{algo_str}', defaulting to dqn")
        algo_str = "dqn"

    # 2) hyperparameters
    params = get_hyperparameters(algo_str, device)
    if batch_size:
        params["batch_size"] = batch_size
    if learning_rate:
        params["learning_rate"] = learning_rate
    
    # Ensure PPO's rollout length does not exceed episode length (max_queries)
    if algo_str == "ppo":
        original_n = params["n_steps"]
        params["n_steps"] = min(params["n_steps"], max_queries)
        # Adjust PPO parameters for better exploration in cache environments
        params["ent_coef"] = 0.1  # Increased from 0.05 for better exploration
        params["clip_range"] = 0.3  # Wider clip range for more policy change
        
        if params["n_steps"] != original_n:
            logger.debug(f"Adjusted PPO n_steps from {original_n} → {params['n_steps']} to match max_queries={max_queries}")
            logger.debug(f"Increased entropy coefficient to {params['ent_coef']} and clip range to {params['clip_range']}")
    
    # Add early stopping to prevent catastrophic forgetting
    if algo_str in ["a2c", "ppo"]:
        logger.debug("Adding early stopping callback to prevent performance degradation")
    
    logger.info(f"Hyperparameters: {params}")

    # 3) build & wrap train_env
    train_env = DummyVecEnv([lambda: Monitor(create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights,
    ))])
    logger.debug("Train DummyVecEnv + Monitor created")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    logger.debug("Train VecNormalize applied")

    # 4) build eval_env and wrap with VecNormalize (share stats with train_env)
    eval_base = DummyVecEnv([lambda: Monitor(create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=max_queries,
        table_name=table_name,
        cache_weights=cache_weights,
    ))])
    logger.debug("Eval DummyVecEnv + Monitor created")
    # Wrap eval_env to mirror train normalization
    eval_env = VecNormalize(eval_base, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Disable updating statistics during evaluation
    eval_env.training = False
    eval_env.norm_reward = False
    # Share statistics from train_env
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    logger.info("Eval VecNormalize wrapper applied with shared stats")

    # 5) EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="model_checkpoints/",
        log_path="model_checkpoints/",
        eval_freq=max_queries * 10,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    logger.info("EvalCallback configured")

    # 6) policy kwargs & LR schedule
    logger.info("Setting up policy_kwargs...")
    policy_kwargs = {"net_arch": params["net_arch"]}
    if device == "cuda":
        logger.info("Configuring GPU feature extractor...")
        try:
            policy_kwargs.update({
                "features_extractor_class": create_feature_extractor(device),
                "features_extractor_kwargs": {"features_dim": 128},
            })
            logger.info("GPU feature extractor configured successfully")
        except Exception as e:
            logger.error(f"Error configuring feature extractor: {e}")
            raise

    logger.info("Setting up learning rate schedule...")
    try:
        # Simplify LR handling - avoid potential hang in get_linear_fn
        if algo_str == "ppo":
            logger.info(f"Using simple learning rate decay for PPO instead of function")
            lr_schedule = params["learning_rate"]  # Use static LR for now
        else:
            lr_schedule = params["learning_rate"]

        logger.info(f"Learning rate set to: {lr_schedule}")

        # Create common kwargs dict in a safe way
        common = {}
        common["verbose"] = 1
        common["device"] = device
        common["policy_kwargs"] = policy_kwargs
        logger.info(f"Common kwargs created with device={device}, verbose=1")
    except Exception as e:
        logger.error(f"Error setting up learning rate or common kwargs: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    logger.info(f"Common kwargs: device={device}, net_arch={params['net_arch']}")

    # 7) instantiate model
    logger.info(f"Creating {algo_str.upper()} model...")
    try:
        if algo_str == "a2c":
            model = A2C("MlpPolicy", train_env,
                        learning_rate=params["learning_rate"],
                        n_steps=params["n_steps"],
                        ent_coef=params["ent_coef"],
                        vf_coef=params["vf_coef"],
                        **common)
            logger.info("A2C model created successfully")
        elif algo_str == "ppo":
            model = PPO("MlpPolicy", train_env,
                        learning_rate=lr_schedule,
                        n_steps=params["n_steps"],
                        batch_size=params["batch_size"],
                        n_epochs=params["n_epochs"],
                        ent_coef=params["ent_coef"],
                        vf_coef=params["vf_coef"],
                        clip_range=params["clip_range"],
                        max_grad_norm=params["max_grad_norm"],
                        normalize_advantage=True,
                        **common)
            logger.info("PPO model created successfully")
        else:
            model = DQN("MlpPolicy", train_env,
                        learning_rate=params["learning_rate"],
                        buffer_size=params["buffer_size"],
                        learning_starts=params["learning_starts"],
                        batch_size=params["batch_size"],
                        target_update_interval=params["target_update_interval"],
                        **common)
            logger.info("DQN model created successfully")
    except Exception as e:
        logger.error(f"Error instantiating model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    logger.info(f"Model instantiated: {algo_str.upper()} on {device}")

    # 8) train with more detailed progress monitoring
    start = datetime.utcnow()
    logger.info(f"Starting model.learn() for {timesteps} steps...")

    # Track performance over time
    reward_history = []
    hit_rate_history = []
    last_log_time = datetime.utcnow()
    log_interval = 60  # seconds

    # Properly implement ProgressCallback inheriting from BaseCallback
    class ProgressCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.last_mean_reward = -float('inf')
            self.no_improvement_count = 0
            self.last_log_time = datetime.utcnow()
            
        def _init_callback(self) -> None:
            # Initialize callback variables here
            pass
            
        def _on_step(self) -> bool:
            nonlocal last_log_time, reward_history, hit_rate_history
            
            # Check if it's time to log progress
            now = datetime.utcnow()
            elapsed = (now - last_log_time).total_seconds()
            
            if elapsed >= log_interval:
                # Get episode info from the model's episode_buffer
                if len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    mean_reward = np.mean(rewards) if rewards else -float('inf')
                    reward_history.append(mean_reward)
                    
                    # Extract hit rate from infos if available
                    if hasattr(self.locals, 'infos') and self.locals.get('infos'):
                        hit_rates = [info.get('cache_hit_rate', 0) for info in self.locals['infos'] 
                                    if isinstance(info, dict) and 'cache_hit_rate' in info]
                        if hit_rates:
                            mean_hit_rate = np.mean(hit_rates)
                            hit_rate_history.append(mean_hit_rate)
                            logger.info(f"Progress: ts={self.num_timesteps}/{timesteps}, "
                                       f"reward={mean_reward:.2f}, hit_rate={mean_hit_rate:.2f}")
                        else:
                            logger.info(f"Progress: ts={self.num_timesteps}/{timesteps}, "
                                       f"reward={mean_reward:.2f}")
                    else:
                        logger.info(f"Progress: ts={self.num_timesteps}/{timesteps}, "
                                   f"reward={mean_reward:.2f}")
                                   
                    # Early stopping check - if rewards are not improving
                    if mean_reward > self.last_mean_reward:
                        self.last_mean_reward = mean_reward
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                        
                    if self.no_improvement_count >= 5:
                        logger.warning(f"No improvement for 5 consecutive checks. Consider early stopping.")
                        
                last_log_time = now
                
            return True  # Return True to continue training

    try:
        # Create the callbacks and pass to learn
        progress_callback = ProgressCallback()
        callbacks = [eval_callback, progress_callback]
        
        model.learn(total_timesteps=timesteps, callback=callbacks)
        logger.info(f"model.learn() finished in {(datetime.utcnow() - start).total_seconds():.1f}s")
        
        # Log final performance metrics
        if reward_history:
            logger.info(f"Training reward history: {reward_history}")
        if hit_rate_history: 
            logger.info(f"Training hit rate history: {hit_rate_history}")
            
    except Exception as e:
        logger.error(f"Error during model.learn(): {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    # 9) save VecNormalize stats
    norm_path = "model_checkpoints/vecnormalize.pkl"
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    train_env.save(norm_path)
    logger.info(f"Saved VecNormalize stats at {norm_path}")

    # 10) save model & metadata
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
    logger.info(f"Model & metadata saved to {save_path}")

    train_env.close()
    eval_env.close()
    if device == "cuda":
        torch.cuda.empty_cache()
    return save_path


def evaluate_cache_model(
    model_path: str,
    eval_steps: int = 1_000,
    db_url: Optional[str] = None,
    use_gpu: bool = False,
    table_name: str = "cache_metrics",
) -> Dict[str, Any]:
    """Run a full evaluation of a trained cache model."""
    from api.app_utils import CacheTableEnum

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    logger.info(f"Evaluating on device: {device}")

    # Load metadata
    meta_path = model_path + ".meta.json"
    try:
        info = json.load(open(meta_path, encoding="utf-8"))
        feature_columns = info.get("feature_columns")
        cache_size = info.get("cache_size", 10)
    except Exception:
        feature_columns = None
        cache_size = 10

    if not feature_columns:
        feature_columns = [e.value for e in CacheTableEnum]

    # Load the correct model type
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, device=device)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path, device=device)
    else:
        model = DQN.load(model_path, device=device)
    logger.info("Model loaded for evaluation")

    # Create eval env
    env = create_mariadb_cache_env(
        db_url=db_url,
        cache_size=cache_size,
        feature_columns=feature_columns,
        max_queries=eval_steps,
        table_name=table_name,
    )
    obs, _ = env.reset()

    # Run episodes
    results = {
        "hit_history": [], "rewards": [], "moving_hit_rates": [],
        "total_reward": 0.0, "avg_inference_ms": 0.0,
        "actions": [], "cache_occupancy": [], "inference_times": [],
        "step_reasoning": [], "in_cache": [], "urls": [],
        "evaluation_time_seconds": 0.0, "success": True
    }
    start = datetime.utcnow()
    for i in range(eval_steps):
        t0 = datetime.utcnow()
        action, _ = model.predict(obs, deterministic=True)
        inf_ms = (datetime.utcnow() - t0).total_seconds() * 1000
        results["inference_times"].append(inf_ms)
        results["actions"].append(int(action))

        prev_hits = env.cache_hits
        obs, reward, done, _, info = env.step(action)
        hit = env.cache_hits > prev_hits

        results["hit_history"].append(int(hit))
        results["rewards"].append(reward)
        results["total_reward"] += reward
        results["cache_occupancy"].append(len(env.cache))
        results["step_reasoning"].append("HIT" if hit else ("MISS→cached" if action==1 else "MISS"))
        idx = (env.current_query_idx - 1) % len(env.data)
        current = env.data.iloc[idx]
        still = any(all(current[c]==itm[c] for c in env.feature_columns) for itm in env.cache)
        results["in_cache"].append(still)
        results["urls"].append(current.get("cache_name"))

        if done:
            break

    results["evaluation_time_seconds"] = (datetime.utcnow() - start).total_seconds()
    results["avg_inference_ms"] = float(np.mean(results["inference_times"])) if results["inference_times"] else 0.0
    results["moving_hit_rates"] = [np.mean(results["hit_history"][max(0, i-24):i+1]) for i in range(len(results["hit_history"]))]

    env.close()
    torch.cuda.empty_cache()
    return results

