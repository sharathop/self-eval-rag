import sqlite3
import json
from datetime import datetime

DB_PATH = "rag_pipeline_logs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer TEXT,
            final_verdict TEXT,
            display_verdict TEXT,
            cosine_score REAL,
            bert_score REAL,
            nli_label TEXT,
            context_relevance_score REAL,
            context_relevance_verdict TEXT,
            context_recall_score REAL,
            context_recall_verdict TEXT,
            retries_used INTEGER,
            full_generation_eval TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(query, answer, eval_result, relevance, recall, retries, display_verdict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_logs (
            timestamp, query, answer, final_verdict, display_verdict,
            cosine_score, bert_score, nli_label,
            context_relevance_score, context_relevance_verdict,
            context_recall_score, context_recall_verdict,
            retries_used, full_generation_eval
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        query,
        answer,
        eval_result.get("final_verdict"),
        display_verdict,
        eval_result.get("cosine", {}).get("score"),
        eval_result.get("bert_score", {}).get("score"),
        eval_result.get("nli", {}).get("label"),
        relevance.get("avg_relevance_score"),
        relevance.get("verdict"),
        recall.get("score"),
        recall.get("verdict"),
        retries,
        json.dumps(eval_result)
    ))
    conn.commit()
    conn.close()


def get_all_logs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_cached_result(query, max_age_hours=24):
    """
    Look up the most recent result for this exact query (case-insensitive),
    within the last `max_age_hours`. Returns the cached row as a dict if found,
    else None. Caches only FINAL, already-evaluated/retried results — the
    self-correction pipeline always runs at least once for any new query.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM query_logs
        WHERE LOWER(query) = LOWER(?)
        AND datetime(timestamp) >= datetime('now', ?)
        ORDER BY id DESC
        LIMIT 1
    """, (query, f'-{max_age_hours} hours'))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None