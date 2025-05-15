# ERS-ML-Trainer: Reinforcement Learning for Smart Caching

## Overview

ERS-ML-Trainer is a Python-based system for training, evaluating, and deploying reinforcement learning (RL) models to optimize caching strategies for API/database queries. It supports dynamic feature selection, model export, and integration with real or simulated data sources.

## Features

- **Dynamic Feature Selection:** Automatically discovers and uses available columns from the `cache_metrics` table.
- **RL Algorithms:** Supports DQN, PPO, and A2C for cache policy learning.
- **API-Driven:** FastAPI endpoints for training, evaluation, model export, and job management.
- **Model Export:** Exports trained models as TorchScript (`policy.pt`) for easy integration with Java or other consumers.
- **Mock & Simulation:** Tools for generating realistic cache metrics and simulating API traffic.
- **Database Support:** Works with MySQL, MariaDB, PostgreSQL, and SQLite.
- **Extensible:** Easily add new features, endpoints, or reward strategies.

## Architecture

```
+-------------------+      +---------------------+      +----------------------+
|   API (FastAPI)   | <--> |   RL Trainer Core   | <--> |   Database (SQL)     |
+-------------------+      +---------------------+      +----------------------+
        |                        |                               |
        v                        v                               v
+-------------------+      +---------------------+      +----------------------+
|   Mock/Simulator  |      |   Model Export      |      |   Model Registry     |
+-------------------+      +---------------------+      +----------------------+
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- MySQL/MariaDB/PostgreSQL/SQLite (for persistent storage)
- (Optional) CUDA GPU for accelerated training

### Installation

```bash
git clone https://github.com/your-org/ers-ml-trainer.git
cd ers-ml-trainer
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Database Setup

1. Edit `database/01_create_tables.sql` and `database/02_seed_tables.sql` as needed.
2. Run the SQL scripts to create and seed your database, or let the app create tables on startup.

### Configuration

Edit environment variables or `.env` file for database connection settings:

```
DB_DRIVER=mysql+mysqlconnector
DB_HOST=localhost
DB_PORT=3306
DB_USER=cacheuser
DB_PASSWORD=cachepass
DB_NAME=cache_db
```

### Running the API

```bash
uvicorn api.app:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

### Training a Model

Send a POST request to `/train` with your desired parameters. Example:

```bash
curl -X POST "http://localhost:8000/train?algorithm=dqn&cache_size=10&timesteps=100000"
```

Monitor job status via `/jobs/{job_id}`.

### Exporting a Model

After training completes, export the model for deployment:

```bash
curl -X POST "http://localhost:8000/export/{job_id}?output_dir=best_model"
```

The exported model (`policy.pt`) and metadata will be in the specified directory.

### Simulating Traffic

Use the simulator to generate realistic API traffic:

```bash
python mock/simulate_live.py --api_url http://localhost:8000 --n 10000
```

### Mock Data Generation

Generate mock cache metrics for testing:

```bash
python mock/mock_db.py --db_type mysql --host localhost --user cacheuser --password cachepass --database cache_db
```

## Docker Build & Run

You can build and run ERS-ML-Trainer using Docker for both CPU and GPU environments.

### CPU Version

```bash
docker build -t ers-ml-trainer:cpu -f docker/Dockerfile.cpu .
```
```bash
docker run --rm -p 8000:8000 --env-file .env ers-ml-trainer:cpu
```

### GPU Version (NVIDIA CUDA)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -t ers-ml-trainer:gpu -f Dockerfile.gpu .
```
```bash
docker run --rm --gpus all -p 8000:8000 --env-file .env ers-ml-trainer:gpu
```

> **Note:** Adjust `--env-file` or environment variables as needed for your database and configuration.

## Key Components

- **api/app.py:** FastAPI endpoints for training, evaluation, export, and job management.
- **core/model_training.py:** RL training and evaluation logic.
- **core/cache_environment.py:** Custom OpenAI Gym environment for cache simulation.
- **database/database_connection.py:** Database utilities and table creation.
- **mock/simulate_live.py:** Simulates API traffic for realistic cache metrics.
- **mock/mock_db.py:** Generates mock data for testing and development.

## Customization

- **Feature Columns:** The system can dynamically select features from `cache_metrics`. You can specify custom columns via the API.
- **Cache Weights:** Apply custom weights to features to prioritize certain metrics during training.
- **Endpoints:** Update `ers-api.yaml` to reflect your API structure; the simulator will use this to generate traffic.

## Best Practices

- Use normalized endpoint names (without timestamps) as cache keys for better generalization.
- Align your feature columns in both training and production for consistent model performance.
- Monitor cache hit ratios, load times, and other metrics to evaluate RL policy effectiveness.

## Troubleshooting

- **Shape Mismatch:** Ensure the number and order of features in your input match the model's `feature_columns` (see metadata).
- **Database Errors:** Check that all required tables exist and columns match the schema in `database/01_create_tables.sql`.
- **SSL Errors in Simulation:** Use `http://` if your API server does not support HTTPS.

## License

MIT License

## Contributors

- [Your Name]
- [Other Contributors]

## Acknowledgements

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Gym](https://www.gymlibrary.dev/)

# System Overview: How the RL Caching Pipeline Works

## 1. Data & Simulation

- **Database Tables:** The system expects a database (e.g., MySQL/MariaDB) with tables like `cache_metrics` and others for energy/production data.
- **Mock Data:** You can generate mock data using `mock/mock_db.py` or simulate realistic API traffic with `mock/simulate_live.py`. The latter sends real HTTP requests to your API endpoints, and the backend records cache metrics as a result.
- **Feature Columns:** The RL agent uses columns from `cache_metrics` (like `cache_name`, `hit_ratio`, `load_time_ms`, etc.) as features for learning.

## 2. API Layer

- **FastAPI App:** The API (in `api/app.py`) exposes endpoints for:
  - Training RL models (`/train`)
  - Evaluating models (`/evaluate/{model_id}`)
  - Exporting models (`/export/{model_id}`)
  - Seeding/simulating data (`/db/seed`, `/simulation/start`)
  - Monitoring and logs
- **Dynamic Feature Discovery:** Endpoints like `/available-columns` and `/feature-columns-enum` let you inspect which columns are available for training.

## 3. RL Training

- **Environment:** `core/cache_environment.py` defines a Gymnasium environment that simulates a cache. The agent observes features of each query and the cache state, and decides whether to cache the result.
- **Training:** The `/train` endpoint launches a background job that:
  - Loads data from the database.
  - Builds the RL environment with selected features and cache size.
  - Trains a model (DQN, PPO, or A2C) for a specified number of timesteps.
  - Saves the model and metadata (algorithm, features, cache size, etc.).
- **Custom Weights:** You can specify `cache_weights` to prioritize certain metrics in the reward calculation.

## 4. Model Evaluation

- **Evaluation:** The `/evaluate/{model_id}` endpoint runs the trained model in the environment, tracking hit rates, rewards, and step-by-step reasoning (including which URLs are hit).
- **Visualization:** Results are visualized (see `core/visualization.py`) with plots of hit/miss patterns, rewards, and even real-time training progress using `RealTimeTrainingPlotter`.

## 5. Model Export & Registry

- **Export:** The `/export/{model_id}` endpoint converts the trained model to TorchScript (`policy.pt`) and saves it (and its metadata) to disk and the `rl_models` table in the database, including all relevant training parameters.
- **Metadata:** All important info (algorithm, device, cache size, feature columns, etc.) is stored with the model for reproducibility and downstream use.

## 6. Integration & Inference

- **Deployment:** The exported `policy.pt` can be loaded in production systems (e.g., Java with DJL) for real-time cache decision-making.
- **Feature Alignment:** The input features provided to the model at inference time **must match** the `feature_columns` used during training (see metadata).

## 7. Monitoring & Simulation

- **Simulation:** You can run live simulations to generate realistic cache metrics, which are then used for RL training.
- **Monitoring:** The API provides endpoints for checking job status, simulation status, and retrieving logs.

## 8. Extensibility

- **Add Features:** You can add new metrics to `cache_metrics` and use them as features.
- **Custom Rewards:** Adjust the reward logic in `core/cache_environment.py` to reflect your caching goals.
- **New Algorithms:** Add new RL algorithms by extending the training logic.

---

## Typical Workflow

1. **Seed or simulate data** in your database.
2. **Inspect available features** via the API.
3. **Start a training job** with your chosen algorithm, cache size, and features.
4. **Monitor training progress** and visualize results.
5. **Evaluate the trained model** to check performance.
6. **Export the model** for deployment.
7. **Integrate the exported model** into your production cache pipeline, ensuring feature alignment.
8. **Repeat** as you gather more data or want to tune your cache policy.

---
`

For most cache‐optimization tasks you’ll want a batch size that balances stable learning with hardware limits:

• DQN: 32–64
• A2C: 16–32
• PPO: 64–256

Smaller batches (e.g. 32) give noisier but more frequent updates; larger ones (128–256) smooth your gradient estimates but increase memory/compute per update. Start at 64, watch GPU/CPU memory and training stability, then tweak up or down.