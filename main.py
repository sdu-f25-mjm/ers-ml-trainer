# main.py
import argparse
import logging
import os
import sys
from pathlib import Path
from api.app import app

import uvicorn
from dotenv import load_dotenv


def setup_logging(log_level: str):
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/application.log")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")


def create_app():
    """Configure and return the app for use by uvicorn."""
    load_dotenv()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    from core.utils import print_system_info
    print_system_info()

    return app


def main():
    """Main entry point for the application when run as a script."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Cache RL Optimization API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    from core.utils import print_system_info
    print_system_info()
    logger.info(f"Starting server on {args.host}:{args.port}")

    try:
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


app = create_app()

if __name__ == "__main__":
    main()