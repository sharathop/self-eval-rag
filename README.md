# Self-Correcting RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that evaluates its own answers and automatically retries when retrieval or generation quality is weak — instead of silently returning a possibly-wrong response.

## Architecture

```
User Question
     │
     ▼
Hybrid Retrieval (FAISS vector search + BM25 keyword search)
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Retrieval Self-Eval (Context Relevance + Context Recall)
     │  └─ if weak → widen search and retry
     ▼
Generation (Groq / Llama 3.3 70B)
     │
     ▼
Generation Self-Eval (external eval framework: NLI, cosine, BERTScore, fluency)
     │  └─ if "Hallucinated" → double-checked by an LLM judge before retrying
     │  └─ if genuinely weak → regenerate with a stricter grounding prompt
     ▼
Final Verdict + SQLite Logging + Cache
```

## Stack

- **Retrieval:** FAISS (vector) + BM25 (keyword), combined and reranked with a cross-encoder
- **Chunking:** Structure-aware — splits on chapter/section headers rather than fixed character counts
- **Generation:** Groq API (Llama 3.3 70B)
- **Self-evaluation:** A separate FastAPI service (NLI, cosine similarity, BERTScore, fluency) + custom retrieval-quality checks (Context Relevance, Context Recall)
- **Backend:** FastAPI
- **Storage:** SQLite (query history, caching)
- **UI:** Streamlit

## Running locally

You need three services running at once:

1. **Eval framework** (separate project, port 8000):
   ```bash
   uvicorn main:app --reload
   ```

2. **This RAG pipeline** (port 8001):
   ```bash
   uvicorn main:app --reload --port 8001
   ```

3. **Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

Set your Groq API key in a `.env` file (not committed):
```
GROQ_API_KEY=your_key_here
```

## Known limitations (found through testing, not assumed)

- **NLI struggles with numeric facts and lists.** The eval framework's NLI component frequently labels correct numeric answers (e.g., "accuracy of 99.6%") or list-format answers as "neutral" rather than "entailment," dragging the overall verdict down even when cosine similarity and BERTScore both agree the answer is faithful. Fixed with a `display_verdict` override: if cosine and BERTScore are both >=0.75, the NLI dissent is overridden.
- **NLI also misfires on long, multi-claim answers**, sometimes labeling them "Hallucinated" even when every individual claim is grounded in the context. Fixed with a secondary LLM-judge check that verifies claim-by-claim before a retry is triggered.
- **Reranking can favor surface wording over authoritative answers.** A query like "what dataset was used for training" initially retrieved a chunk about the 70/15/15 train/test split (matching the word "training") instead of the actual dataset-source section, because the correct chunk didn't use that keyword as prominently. Fixed by widening `final_k` (top-N kept after reranking) from 3 to 10.
- **Context Relevance and Context Recall can disagree** — a case where Relevance was scored low (0.47) while Recall was scored high (1.0) for the same retrieval, since they measure different things (topical closeness vs. sufficiency of information).

## Project story

This project extends an existing standalone LLM hallucination-detection framework into a live, closed-loop RAG system: retrieval and generation are now self-checked and self-corrected in real time, using that same eval framework as one component rather than a standalone tool.