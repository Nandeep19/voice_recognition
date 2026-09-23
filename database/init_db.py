from src.database import init_db


if __name__ == "__main__":
    init_db()
    print("PostgreSQL tables, pgvector extension and person ID sequence initialized.")
