
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

