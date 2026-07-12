from fastapi import FastAPI
from pydantic import BaseModel
import json

from genrate import run_pipeline
from db import init_db, log_query, get_all_logs, get_cached_result

app = FastAPI(title="Self-Correcting RAG Pipeline", version="1.0.0")

# Create the database/table on startup if it doesn't exist yet
init_db()


class QueryRequest(BaseModel):
    query: str
    max_retries: int = 1
    use_cache: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    final_verdict: str
    display_verdict: str
    full_generation_eval: dict
    context_relevance: dict
    context_recall: dict
    retries_used: int
    retry_log: list
    from_cache: bool = False


@app.get("/")
def home():
    return {"message": "Self-Correcting RAG Pipeline is running"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    # --- Check cache first: same question asked recently already went
    # through the full retrieve -> generate -> evaluate -> retry pipeline,
    # so we reuse that verified result instead of repeating all the work. ---
    if request.use_cache:
        cached = get_cached_result(request.query, max_age_hours=24)
        if cached:
            print(f"\n[Cache Hit] Reusing verified result for: '{request.query}'")
            return QueryResponse(
                query=cached["query"],
                answer=cached["answer"],
                final_verdict=cached["final_verdict"],
                display_verdict=cached["display_verdict"],
                full_generation_eval=json.loads(cached["full_generation_eval"]),
                context_relevance={
                    "avg_relevance_score": cached["context_relevance_score"],
                    "verdict": cached["context_relevance_verdict"]
                },
                context_recall={
                    "score": cached["context_recall_score"],
                    "verdict": cached["context_recall_verdict"]
                },
                retries_used=cached["retries_used"],
                retry_log=[],  # not stored historically, only for live runs
                from_cache=True
            )

    # --- Cache miss: run the full self-correcting pipeline ---
    answer, eval_result, relevance, recall, retries, display_verdict, retry_log = run_pipeline(
        request.query, max_retries=request.max_retries
    )

    # Log this query + result to SQLite (including the smarter display_verdict)
    log_query(request.query, answer, eval_result, relevance, recall, retries, display_verdict)

    return QueryResponse(
        query=request.query,
        answer=answer,
        final_verdict=eval_result.get("final_verdict"),
        display_verdict=display_verdict,
        full_generation_eval=eval_result,
        context_relevance=relevance,
        context_recall=recall,
        retries_used=retries,
        retry_log=retry_log,
        from_cache=False
    )


@app.get("/history")
def history(limit: int = 50):
    logs = get_all_logs(limit=limit)
    return {"count": len(logs), "logs": logs}