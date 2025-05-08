import logging
import os

logger = logging.getLogger(__name__)


def safe_execute(db_handler, sql, description):
    try:
        db_handler.execute_query(sql)
        logger.info(f"Table/index for '{description}' created or already exists.")
    except Exception as e:
        logger.error(f"Failed to create table/index for '{description}': {e}")


def load_sql_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def create_tables(db_handler):
    """
    Create all necessary database tables using the canonical SQL schema.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "01_create_tables.sql")
    schema_sql = load_sql_file(schema_path)
    # Split on semicolon to execute each statement separately (handles MySQL/MariaDB)
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            safe_execute(db_handler, stmt, stmt.split("(")[0].strip().split()[-1])
    logger.info("All tables creation attempted (see errors above if any failed)")
    return True


def seed_tables(db_handler):
    """
    Seed static values using the canonical seed SQL.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seed_path = os.path.join(base_dir, "02_seed_tables.sql")
    seed_sql = load_sql_file(seed_path)
    for stmt in seed_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            safe_execute(db_handler, stmt, "seed")
    logger.info("All tables seeded (see errors above if any failed)")
    return True
