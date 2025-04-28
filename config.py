import os

DB_DRIVER = os.getenv("DB_DRIVER", "mysql+mysqlconnector")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME", "cache_db")
DB_USER = os.getenv("DB_USER", "cacheuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "cachepass")
DB_URL = os.getenv(
    "DB_URL",
    f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
