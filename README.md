# Self-Correcting RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that evaluates its own answers and automatically retries when retrieval or generation quality is weak — instead of silently returning a possibly-wrong response. Built as an extension of an existing standalone LLM hallucination-detection framework, now wired in as a live component of a closed-loop retrieval + generation + self-evaluation system.

**Live demo:** https://self-eval-rag-w4ebandtxswtbapvmhguya.streamlit.app
**Eval framework (separate project, used as a component here):** https://sha6th-llm-eval-ap.hf.space/docs

## Architecture

```
User Question
     |
     v
Hybrid Retrieval (FAISS vector search + BM25 keyword search)
     |
     v
Cross-Encoder Reranking (wide candidate pool -> top N)
     |
     v
Retrieval Self-Eval (Context Relevance + Context Recall)
     |  -> if weak -> widen search and retry
     v
Generation (Groq / Llama 3.3 70B)
     |
     v
Generation Self-Eval (external eval framework: NLI, cosine, BERTScore, fluency)
     |  -> if "Hallucinated" -> double-checked by an LLM judge (claim-by-claim) before retrying
     |  -> if genuinely weak -> regenerate with a stricter grounding prompt
     v
Display Verdict (smarter than raw eval verdict) + SQLite Logging + Cache
```

## Stack

- **Retrieval:** FAISS (vector) + BM25 (keyword), combined and reranked with a cross-encoder
- **Extraction:** PyMuPDF with column-aware block ordering (correctly handles two-column academic PDFs)
- **Chunking:** Structure-aware — splits on chapter/section headers rather than fixed character counts; supports multiple source PDFs, each chunk tagged with `[Document: ...]` and `[Section: ...]`
- **Generation:** Groq API (Llama 3.3 70B)
- **Self-evaluation:** A separate FastAPI service (NLI, cosine similarity, BERTScore, fluency) + custom retrieval-quality checks (Context Relevance, Context Recall) + a secondary LLM-judge for disputed "Hallucinated" verdicts
- **Storage:** SQLite (query history, 24-hour result caching)
- **UI:** Streamlit (combined single-file deployment for hosting; also available as a separate FastAPI + Streamlit split for local development)

## Running locally

**Option A — single file (simplest):**
```bash
pip install -r requirements.txt
python ingest.py          # builds faiss_index/ and bm25_chunks.pkl from data/*.pdf
streamlit run streamlit_app.py
```

**Option B — FastAPI + Streamlit split (demonstrates API/frontend separation):**
```bash
python ingest.py
uvicorn main:app --reload --port 8001
streamlit run dashboard.py
```

Set your Groq API key in a `.env` file (not committed) or as an environment variable:
```
GROQ_API_KEY=your_key_here
```

## Known limitations and fixes (found through testing, not assumed)

1. **Two-column academic PDFs jumble text.** Standard extractors (PyPDFLoader) read left-to-right across the full page width, interleaving text from separate columns mid-sentence. Fixed by extracting text as positioned blocks (PyMuPDF) and reading each column top-to-bottom before moving to the next.

2. **NLI struggles with numeric facts, lists, and long multi-claim answers.** The eval framework's NLI component frequently labels correct numeric answers (e.g., "accuracy of 99.6%"), list-format answers, or long compound-sentence answers as "neutral" or even "Hallucinated," dragging the overall verdict down even when cosine similarity and BERTScore both agree the answer is faithful. Fixed two ways:
   - A `display_verdict` override: if cosine and BERTScore are both >=0.75, an "Unverifiable" NLI dissent is overridden to "Faithful."
   - A secondary LLM-judge check: if NLI flags "Hallucinated," a separate claim-by-claim LLM check verifies the answer before a retry is triggered, since NLI is known to misfire on this exact case.

3. **Reranking can favor surface wording over the authoritative answer.** A query like "what dataset was used for training" initially retrieved a chunk about the 70/15/15 train/test split (matching the word "training") instead of the actual dataset-source section, because the correct chunk didn't share that keyword as prominently. Fixed by widening the retrieval window (`final_k` from 3 to 10) so lower-ranked but correct chunks still make it into context.

4. **Retrieval quality can vary by question phrasing even on a clean, single-topic index.** One specific question ("what features were selected for DDoS detection") consistently returned a low Context Recall score (0.4) regardless of whether an unrelated second document was present in the index or not -- ruling out cross-document vocabulary collision as the cause. The root cause was not fully isolated; documented here as an open edge case rather than assumed-and-left-unverified.

5. **Multi-document indexing risks vocabulary collision between topically similar documents.** Indexing a DDoS detection report alongside an unrelated ML paper (XGBoost) that also discusses "features" and "training" caused measurable retrieval degradation on document-specific questions. Testing with a topically distant second document (blockchain/OCR) did not reproduce this issue, suggesting the effect is specific to vocabulary overlap between documents, not multi-document indexing itself. A production fix would add document-aware retrieval filtering; not implemented here as a deliberate scope decision given time constraints -- noted as future work.

6. **Caching must be invalidated when the underlying index changes.** The query-result cache has no awareness of when `ingest.py` is re-run with a different or updated document set, so a stale cached answer can be served after reindexing. Mitigated procedurally (clear the database after any reindex); a more robust fix would hash the current index and include it as part of the cache key.

## Project story

This project extends an existing standalone LLM hallucination-detection framework into a live, closed-loop RAG system: retrieval and generation are now self-checked and self-corrected in real time, using that same eval framework as one component rather than a standalone tool. Development proceeded through systematic testing against a real, structurally complex document (a 53-page technical report with numbered chapters/sections), with each fix driven by a specific, reproduced failure rather than speculative hardening.