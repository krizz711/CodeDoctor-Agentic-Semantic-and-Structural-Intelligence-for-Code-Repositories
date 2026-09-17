import sqlite3
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "codedoctor.db")
MIGRATIONS_DIR = Path(__file__).parent.parent / "db" / "migrations"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def apply_migrations():
    """
    Applies SQL migrations sequentially.
    Enforces Rule 2 (Defensive Coding) by ensuring the DB is in a known state.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create migrations tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    if not MIGRATIONS_DIR.exists():
        logger.warning(f"Migrations directory not found: {MIGRATIONS_DIR}")
        return
        
    migrations = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.sql')])
    
    for migration in migrations:
        cursor.execute('SELECT version FROM schema_migrations WHERE version = ?', (migration,))
        if cursor.fetchone() is None:
            logger.info(f"Applying migration {migration}")
            with open(MIGRATIONS_DIR / migration, 'r') as f:
                cursor.executescript(f.read())
            cursor.execute('INSERT INTO schema_migrations (version) VALUES (?)', (migration,))
            conn.commit()
    
    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_migrations()
    print("Database migrations applied successfully.")
