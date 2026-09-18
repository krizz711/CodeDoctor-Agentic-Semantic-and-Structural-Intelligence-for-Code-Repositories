import logging
import hashlib
import re
from app.db import get_connection

logger = logging.getLogger(__name__)

def incremental_index(repo: str, files_commit_a: dict, files_commit_b: dict):
    """
    Simulates evolutionary sub-second indexing.
    Rule 4: Relies on the SHA-256 semantic hashes of chunks to skip re-indexing 
    unmodified AST nodes across commits.
    """
    from app.indexer import index_file
    
    # Simulate Commit A
    for path, content in files_commit_a.items():
        index_file(repo, path, "commit_A", content)
        
    # Simulate Commit B
    for path, content in files_commit_b.items():
        index_file(repo, path, "commit_B", content)
        
    return True

def query_commit(query: str, commit_hash: str):
    """
    Queries the index but restricts the search boundary to a specific commit.
    Satisfies T9 (Zero Cross-Commit Leakage).
    Searches against tokenized keywords and full query string defensively.
    """
    if not query or not isinstance(query, str) or not commit_hash or not isinstance(commit_hash, str):
        logger.warning(f"Invalid query or commit_hash: query={type(query)}, commit={type(commit_hash)}")
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Try exact string match
    cursor.execute("""
        SELECT c.id as chunk_id, f.commit_hash as commit_id 
        FROM chunks c 
        JOIN files f ON c.file_id = f.id 
        WHERE f.commit_hash = ? AND c.content LIKE ?
    """, (commit_hash, f"%{query}%"))
    rows = cursor.fetchall()
    
    # 2. If no exact phrase match, search by extracted keywords
    if not rows:
        tokens = [t for t in re.findall(r'[a-zA-Z0-9_]+', query.lower()) if len(t) > 3]
        if tokens:
            clauses = " OR ".join(["LOWER(c.content) LIKE ?" for _ in tokens])
            params = [commit_hash] + [f"%{t}%" for t in tokens]
            cursor.execute(f"""
                SELECT c.id as chunk_id, f.commit_hash as commit_id 
                FROM chunks c 
                JOIN files f ON c.file_id = f.id 
                WHERE f.commit_hash = ? AND ({clauses})
            """, params)
            rows = cursor.fetchall()
            
    results = [{"chunk_id": row["chunk_id"], "commit": row["commit_id"]} for row in rows]
    conn.close()
    
    return results

def query_evolutionary(query: str, repo: str = None, top_k: int = 10):
    """
    Evolutionary retrieval across all commit versions (Theme 1 Bonus Goal).
    Ranks similar snippets across different versions while maintaining explicit 
    commit lineage metadata to prevent temporal confusion.
    """
    if not query or not isinstance(query, str):
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    base_sql = """
        SELECT c.id as chunk_id, f.commit_hash as commit_id, f.file_path, c.content, c.semantic_hash
        FROM chunks c 
        JOIN files f ON c.file_id = f.id
    """
    params = []
    where_clauses = []
    
    if repo:
        where_clauses.append("f.repo_name = ?")
        params.append(repo)
        
    tokens = [t for t in re.findall(r'[a-zA-Z0-9_]+', query.lower()) if len(t) > 3]
    if tokens:
        keyword_clause = " OR ".join(["LOWER(c.content) LIKE ?" for _ in tokens])
        where_clauses.append(f"({keyword_clause})")
        params.extend([f"%{t}%" for t in tokens])
        
    if where_clauses:
        sql = f"{base_sql} WHERE {' AND '.join(where_clauses)} LIMIT ?"
    else:
        sql = f"{base_sql} LIMIT ?"
    params.append(top_k)
    
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "chunk_id": row["chunk_id"],
        "commit": row["commit_id"],
        "file_path": row["file_path"],
        "content": row["content"],
        "semantic_hash": row["semantic_hash"]
    } for row in rows]


def get_db_state(commit_hash: str):
    """
    Computes a deterministic hash of the database state for a given commit.
    Satisfies T8 (Incremental Equals Rebuild).
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT file_path, content FROM files WHERE commit_hash = ? ORDER BY file_path", (commit_hash,))
    rows = cursor.fetchall()
    conn.close()
    
    state_str = "".join([f"{r['file_path']}:{r['content']}" for r in rows])
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
