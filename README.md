I'll add a section about seeding the database to the README:

# Energy Data Cache Optimization using Reinforcement Learning

This project uses reinforcement learning to optimize database caching strategies for energy data queries. By training
models to predict which data should be kept in cache, it improves query performance and reduces database load.

## Features

- **Reinforcement Learning Cache Optimization**
    - Support for multiple algorithms (DQN, A2C, PPO)
    - Smart data importance evaluation based on renewable energy ratio
    - Adaptive caching based on data volatility and complexity metrics

- **GPU-Accelerated Training**
    - CUDA/GPU support for faster model training
    - Automatic GPU detection and optimization
    - Configurable for multi-GPU environments

- **REST API Service**
    - Model training and evaluation endpoints
    - Async job management for long-running training tasks
    - Performance visualization

- **Database Integration**
    - Support for MariaDB/MySQL
    - Mock energy database generation for testing
    - Intelligent query caching

## Requirements

- Python 3.11+
- CUDA-compatible GPU (optional, for accelerated training)
- Docker and Docker Compose (for containerized deployment)

## Installation

### Using Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/sdu-f25-mjm/ers-ml-trainer.git
cd ers-ml-trainer

# Start with GPU support (if available)
GPU_COUNT=1 docker-compose -f docker/docker-compose.yml up -d

# Or without GPU
docker-compose -f docker/docker-compose.yml up -d
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/sdu-f25-mjm/ers-ml-trainer.git
cd ers-ml-trainer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### API Endpoints

Start the API server:

```bash
python -m main --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with Swagger UI documentation.

### Training a Model via API

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "db_url": "mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db",
    "algorithm": "dqn",
    "cache_size": 10,
    "max_queries": 500,
    "timesteps": 10000,
    "use_gpu": true
  }'
```

### Evaluating a Model

```bash
curl -X POST "http://localhost:8000/evaluate/{job_id}?steps=1000"
```

## Configuration

You can modify the following parameters:

- **Algorithm**: `dqn` (default), `a2c`, or `ppo`
- **Cache Size**: Number of items to keep in cache
- **Timesteps**: Training duration
- **Feature Columns**: Data fields to use for cache decisions
- **GPU Usage**: Enable/disable GPU acceleration

# Build and Run Commands for ERS ML Trainer

## Docker Deployment

### CPU Build & Run

```bash
# Build the CPU image
docker build -f docker/Dockerfile.cpu -t ers-ml-trainer:cpu .
```

```bash
# Run using docker-compose (CPU)
docker-compose -f docker/docker-compose.yml up -d
```

### GPU Build & Run

```bash
# Build the GPU image
docker build -f docker/Dockerfile.gpu -t ers-ml-trainer:gpu .
```

```bash
# Run with GPU support (specify number of GPUs)
docker compose -f docker/docker-compose.yml up -d
```

### Database Setup

```bash
# Generate mock database data
docker-compose -f docker/docker-compose.yml exec ers-ml-trainer python -m mock.mock_db
```

## Manual Deployment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Start API server
python -m main --host 0.0.0.0 --port 8000

# With hot reload (for development)
python -m main --reload
```

## API Usage

```bash
# Train a model
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "db_url": "mysql+mysqlconnector://cacheuser:cachepass@ers-mariadb:3306/cache_db",
    "algorithm": "dqn",
    "cache_size": 10,
    "timesteps": 10000
  }'
  
# Evaluate a model
curl -X POST "http://localhost:8000/evaluate/{job_id}?steps=1000"

# Export model to TorchScript
curl -X POST "http://localhost:8000/export/{job_id}"

# View logs
curl "http://localhost:8000/logs?lines=100&level=ERROR"
```

## Database Seeding

The project includes a database seeding tool to generate realistic energy data for testing and development purposes.

### Using the Mock Database Generator

```bash
# Using Docker (recommended)
docker-compose -f docker/docker-compose.yml exec ers-ml-trainer python -m mock.mock_db

# From local environment
python -m mock.mock_db
```

The mock generator creates the following tables:

- `energy_data`: Primary energy production and consumption metrics
- `production_data`: Detailed breakdown of energy production sources
- `consumption_data`: Detailed consumption patterns by sector
- `exchange_data`: Cross-border energy exchange metrics
- `carbon_intensity`: Carbon intensity and energy mix percentages

Each table contains hourly time series data structured according to energy market standards.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start API server with auto-reload
python -m main --reload
```

## License

[MIT License](LICENSE)