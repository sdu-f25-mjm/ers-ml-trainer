#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import uvicorn
from dotenv import load_dotenv
from pathlib import Path


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


def setup_mock_database(force_recreate: bool):
    """Set up the mock database for testing if needed."""
    logger = logging.getLogger(__name__)
    try:
        from mock.mock_db import generate_mock_database

        # Use environment variables or defaults
        host = os.environ.get('DB_HOST', 'localhost')
        user = os.environ.get('DB_USER', 'cacheuser')
        password = os.environ.get('DB_PASSWORD', 'cachepass')
        database = os.environ.get('DB_NAME', 'cache_db')

        # Check if we're using Docker
        if os.environ.get('DOCKER_ENV') == 'true':
            host = 'ers-mariadb'
            logger.info(f"Docker environment detected, using database host: {host}")

        logger.info(f"Setting up mock database on {host}")

        success = generate_mock_database(
            host=host,
            user=user,
            password=password,
            database=database,
            hours=24,  # Smaller dataset for faster initialization
            price_areas=["DK1", "DK2"]
        )

        if success:
            logger.info("Mock database setup completed successfully")
        else:
            logger.warning("Mock database setup failed")

    except Exception as e:
        logger.error(f"Error setting up mock database: {e}")
        logger.info("Continuing without mock database")


def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cache RL Optimization API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--mock-db", action="store_true", help="Set up a mock database for testing")
    parser.add_argument("--force-recreate-db", action="store_true", help="Force recreation of the mock database")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Show system info
    from ers_ml_trainer.core.gpu_utils import print_system_info
    print_system_info()

    # Set up mock database if requested
    if args.mock_db:
        setup_mock_database(args.force_recreate_db)

    logger.info(f"Starting server on {args.host}:{args.port}")

    # Import the API app
    try:
        # Start the FastAPI application using uvicorn
        uvicorn.run(
            "ers_ml_trainer.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()