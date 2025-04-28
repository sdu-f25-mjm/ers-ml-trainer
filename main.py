# main.py
import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from api.app import app
from config import DB_URL


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

    # Ensure all tables are created at the very beginning of the startup process
    from database.create_tables import create_tables
    from mock.mock_db import get_db_handler  # Use your handler factory

    logger = logging.getLogger(__name__)
    if DB_URL:
        # Parse DB_DRIVER to get the base type (e.g., "mysql", "postgres", "sqlite")
        db_type_full = DB_URL.split(":", 1)[0]
        db_type = db_type_full.split("+")[0]  # Only use the base type
        db_handler = get_db_handler(db_type)
        if db_handler.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", 3306)),
                user=os.getenv("DB_USER", "cacheuser"),
                password=os.getenv("DB_PASSWORD", "cachepass"),
                database=os.getenv("DB_NAME", "cache_db")
        ):
            try:
                create_tables(db_handler)
                logger.info("Database tables ensured at startup.")

                # --- Verify table existence after startup ---
                try:
                    # Use a generic SQL query to get table names
                    table_query = "SHOW TABLES"
                    result = db_handler.execute_query(table_query)
                    existing_tables = set()
                    if result:
                        rows = result.fetchall()
                        for row in rows:
                            # MySQL returns a tuple per row
                            if isinstance(row, (list, tuple)) and len(row) > 0:
                                existing_tables.add(row[0])
                            elif isinstance(row, dict):
                                existing_tables.update(row.values())
                    required_tables = {"best_models", "cache_metrics"}
                    for table in required_tables:
                        if table in existing_tables:
                            logger.info(f"Table '{table}' exists in the database.")
                        else:
                            logger.error(f"Table '{table}' is MISSING in the database!")
                except Exception as e:
                    logger.error(f"Error verifying table existence: {e}")

            except Exception as e:
                logger.error(f"Failed to create tables at startup: {e}")
        else:
            logger.warning("Could not connect to database for table creation.")
    else:
        logger.warning("DB_URL not set; skipping table creation.")

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
