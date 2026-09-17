import logging
from app.db import get_connection
from app.chunker import parse_javascript, ASTParseError

logger = logging.getLogger(__name__)

def index_file(repo_name: str, file_path: str, commit_hash: str, content: str):
    """
    Indexes a single file by parsing it into chunks and storing it in SQLite.
    Rule 7: Catches ASTParseError to handle malformed files robustly without crashing the batch.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if file already exists
        cursor.execute("SELECT id FROM files WHERE repo_name=? AND file_path=? AND commit_hash=?", 
                       (repo_name, file_path, commit_hash))
        if cursor.fetchone():
            logger.info(f"File {file_path} already indexed. Skipping.")
            return True
            
        cursor.execute(
            "INSERT INTO files (repo_name, file_path, commit_hash, content) VALUES (?, ?, ?, ?)",
            (repo_name, file_path, commit_hash, content)
        )
        file_id = cursor.lastrowid
        
        try:
            chunks = parse_javascript(content)
            for chunk in chunks:
                cursor.execute(
                    """INSERT INTO chunks 
                       (file_id, start_byte, end_byte, semantic_hash, ast_type, content) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (file_id, chunk["start_byte"], chunk["end_byte"], 
                     chunk["semantic_hash"], chunk["ast_type"], chunk["content"])
                )
            conn.commit()
            return True
            
        except ASTParseError as e:
            logger.error(f"Failed to parse {file_path}: {e}. File entry rolled back.")
            conn.rollback()
            return False
            
    except Exception as e:
        logger.error(f"Database error while indexing {file_path}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
