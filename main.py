import uvicorn
import os
import argparse
import logging
from api.app import app


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cache_rl_api.log")
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Cache RL Optimization API Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')

    args = parser.parse_args()

    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("cache_eval_results", exist_ok=True)

    # Configure logging
    configure_logging()

    # Start the API service
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()