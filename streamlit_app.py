import os
import math
import pickle
import requests
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from db import init_db, log_query, get_all_logs, get_cached_result

# =========================================================
# SETUP (runs once, cached by Streamlit so it's not reloaded
# on every interaction)
# =========================================================

st.set_page_config(page_title="Self-Correcting RAG Pipeline", layout="wide")

EVAL_API_URL = "https://sha6th-llm-eval-ap.hf.space/evaluate-llm"

@st.cache_resource
def load_pipeline_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    with open("bm25_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return embeddings, vectorstore, reranker, groq_client, all_chunks, bm25


embeddings, vectorstore, reranker, groq_client, all_chunks, bm25 = load_pipeline_components()
init_db()

# =========================================================
# PIPELINE LOGIC (same as genrate.py, kept in-process —
# no HTTP calls to a separate FastAPI service needed)
# =========================================================


def retrieve_with_rerank(query, wide_k=25, final_k=10):
    vector_candidates = vectorstore.similarity_search(query, k=wide_k)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:wide_k]
    bm25_candidates = [all_chunks[i] for i in top_bm25_indices]

    seen = set()
    combined = []
    for doc in vector_candidates + bm25_candidates:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    pairs = [[query, doc.page_content] for doc in combined]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(combined, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:final_k]]


def generate_answer(query, context_chunks, strict=False):
    context_text = "\n\n".join([doc.page_content for doc in context_chunks])
    if strict:
        instruction = "Answer in full, complete sentences (not a list). Use ONLY the context below."
    else:
        instruction = "Answer the question using ONLY the context below."
    prompt = f"""{instruction}
If the answer isn't in the context, say "I cannot find this in the provided context."

The context below contains "[Section: ...]" labels marking which part of the document each
piece of text came from. These labels are for your reference only — do NOT mention them,
cite section numbers, or reference "[Section: ...]" in your answer. Just answer naturally,
as if you were simply reading the document.

Context:
{context_text}

Question: {query}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def evaluate_answer(query, context_text, answer):
    payload = {
    "question": query,
    "context": context_text,
    "llm_response": answer
}
 
    response = requests.post(
        EVAL_API_URL,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    generation = result["generation_evaluation"]

    # Compute generation verdict ourselves
    if generation["nli"]["verdict"] == "Hallucinated":
        verdict = "Hallucinated"

    elif (
        generation["nli"]["verdict"] == "Faithful"
        and generation["bert_score"]["score"] >= 0.70
    ):
        verdict = "Faithful"

    elif (
        generation["cosine"]["verdict"] == "Irrelevant"
        and len(answer.split()) > 2
    ):
        verdict = "Irrelevant"

    else:
        verdict = "Unverifiable"

    generation["final_verdict"] = verdict

    return generation

def check_context_relevance(query, chunks):
    pairs = [[query, doc.page_content] for doc in chunks]
    raw_scores = reranker.predict(pairs)
    normalized_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
    avg_relevance = float(sum(normalized_scores) / len(normalized_scores))
    return {
        "avg_relevance_score": avg_relevance,
        "verdict": "Relevant" if avg_relevance > 0.5 else "Weak Relevance"
    }


def check_context_recall(query, context_text):
    prompt = f"""Given this context and question, rate how much information the context provides to answer the question.
Give a score from 0 to 10 (0 = context has nothing relevant, 10 = context fully answers the question).
Respond in this exact format: SCORE: <number> | VERDICT: <Sufficient or Insufficient>

Context:
{context_text}

Question: {query}

Response:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        score_part = raw.split("SCORE:")[1].split("|")[0].strip()
        verdict_part = raw.split("VERDICT:")[1].strip()
        score = float(score_part) / 10
    except Exception:
        score, verdict_part = None, raw
    return {"score": score, "verdict": verdict_part}


def llm_judge_faithfulness(query, context_text, answer):
    prompt = f"""You are fact-checking an AI-generated answer against a source context.
Go through the answer claim by claim. For each claim, check if it is directly
supported by the context. Ignore phrasing differences — only flag a claim as
unsupported if it states something that contradicts or is absent from the context.

Context:
{context_text}

Answer to check:
{answer}

After checking all claims, respond in this exact format:
VERDICT: <Faithful or Hallucinated>
REASON: <one sentence explaining why>"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        verdict_part = raw.split("VERDICT:")[1].split("\n")[0].strip()
        reason_part = raw.split("REASON:")[1].strip() if "REASON:" in raw else ""
    except Exception:
        verdict_part, reason_part = "Unknown", raw
    return {"verdict": verdict_part, "reason": reason_part}


def needs_retrieval_retry(recall_result, relevance_result):
    recall_score = recall_result.get("score")
    if recall_score is not None and recall_score < 0.5:
        return True
    if relevance_result.get("verdict") == "Weak Relevance":
        return True
    return False


def needs_generation_retry(eval_result):
    verdict = eval_result.get("final_verdict")
    cosine_score = eval_result.get("cosine", {}).get("score", 0)
    bert_score = eval_result.get("bert_score", {}).get("score", 0)
    if verdict == "Hallucinated":
        return True
    if verdict == "Unverifiable":
        return cosine_score < 0.75 or bert_score < 0.75
    return False


def compute_display_verdict(eval_result):
    raw_verdict = eval_result.get("final_verdict")
    cosine_score = eval_result.get("cosine", {}).get("score", 0)
    bert_score = eval_result.get("bert_score", {}).get("score", 0)
    if raw_verdict and raw_verdict.startswith("Faithful (judge-verified"):
        return raw_verdict
    if raw_verdict == "Hallucinated":
        return "Hallucinated"
    if raw_verdict == "Irrelevant":
        return "Irrelevant"
    if raw_verdict == "Unverifiable" and cosine_score >= 0.75 and bert_score >= 0.75:
        return "Faithful (NLI dissent overridden)"
    if raw_verdict == "Unverifiable":
        return "Unverifiable"
    return raw_verdict or "Unknown"


def run_pipeline(query, max_retries=1):
    chunks = retrieve_with_rerank(query, wide_k=25, final_k=10)
    context_text = "\n\n".join([doc.page_content for doc in chunks])

    relevance = check_context_relevance(query, chunks)
    recall = check_context_recall(query, context_text)

    total_retries = 0
    retry_log = []

    if needs_retrieval_retry(recall, relevance) and total_retries < max_retries:
        chunks = retrieve_with_rerank(query, wide_k=20, final_k=5)
        context_text = "\n\n".join([doc.page_content for doc in chunks])
        relevance = check_context_relevance(query, chunks)
        recall = check_context_recall(query, context_text)
        total_retries += 1
        retry_log.append({
            "stage": "retrieval",
            "reason": "Weak Context Relevance/Recall on first attempt",
            "action": "Widened search"
        })

    answer = generate_answer(query, chunks, strict=False)
    eval_result = evaluate_answer(query, context_text, answer)

    if eval_result.get("final_verdict") == "Hallucinated":
        judge_result = llm_judge_faithfulness(query, context_text, answer)
        if judge_result["verdict"] == "Faithful":
            eval_result["final_verdict"] = "Faithful (judge-verified, NLI false-flag)"

    retry_log.append({
        "stage": "generation",
        "attempt": 1,
        "answer": answer,
        "verdict": eval_result.get("final_verdict"),
        "cosine": eval_result.get("cosine", {}).get("score"),
        "bert_score": eval_result.get("bert_score", {}).get("score")
    })

    while needs_generation_retry(eval_result) and total_retries < max_retries:
        answer = generate_answer(query, chunks, strict=True)
        eval_result = evaluate_answer(query, context_text, answer)

        if eval_result.get("final_verdict") == "Hallucinated":
            judge_result = llm_judge_faithfulness(query, context_text, answer)
            if judge_result["verdict"] == "Faithful":
                eval_result["final_verdict"] = "Faithful (judge-verified, NLI false-flag)"

        total_retries += 1
        retry_log.append({
            "stage": "generation",
            "attempt": total_retries + 1,
            "answer": answer,
            "verdict": eval_result.get("final_verdict"),
            "cosine": eval_result.get("cosine", {}).get("score"),
            "bert_score": eval_result.get("bert_score", {}).get("score")
        })

    display_verdict = compute_display_verdict(eval_result)
    return answer, eval_result, relevance, recall, total_retries, display_verdict, retry_log


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🔎 Self-Correcting RAG Pipeline")
st.caption("Retrieval + Generation + Self-Evaluation + Automatic Retry")

tab1, tab2 = st.tabs(["Ask a Question", "History"])

with tab1:
    query = st.text_input("Ask a question about your document:")
    max_retries = st.slider("Max retries allowed", 0, 3, 1)
    use_cache = st.checkbox("Use cache for repeated questions", value=True)

    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving, generating, and self-evaluating..."):
            from_cache = False
            cached = get_cached_result(query, max_age_hours=24) if use_cache else None

            if cached:
                answer = cached["answer"]
                final_verdict = cached["final_verdict"]
                display_verdict = cached["display_verdict"]
                retries = cached["retries_used"]
                relevance = {
                    "avg_relevance_score": cached["context_relevance_score"],
                    "verdict": cached["context_relevance_verdict"]
                }
                recall = {
                    "score": cached["context_recall_score"],
                    "verdict": cached["context_recall_verdict"]
                }
                import json
                full_eval = json.loads(cached["full_generation_eval"])
                retry_log = []
                from_cache = True
            else:
                answer, eval_result, relevance, recall, retries, display_verdict, retry_log = run_pipeline(
                    query, max_retries=max_retries
                )
                final_verdict = eval_result.get("final_verdict")
                full_eval = eval_result
                log_query(query, answer, eval_result, relevance, recall, retries, display_verdict)

            st.subheader("Answer")
            if from_cache:
                st.caption("⚡ Served from cache (identical question asked recently)")
            st.write(answer)

            if display_verdict.startswith("Faithful"):
                st.success(f"✅ Verdict: {display_verdict}")
            elif display_verdict == "Unverifiable":
                st.warning(f"⚠️ Verdict: {display_verdict}")
            elif display_verdict == "Irrelevant":
                st.info(f"ℹ️ Verdict: {display_verdict} (question not answerable from document)")
            else:
                st.error(f"❌ Verdict: {display_verdict}")

            st.caption(f"Raw eval framework verdict: {final_verdict} · Retries used: {retries}")

            if retry_log:
                with st.expander(f"📋 See all {len(retry_log)} attempt(s)"):
                    for i, entry in enumerate(retry_log):
                        if entry.get("stage") == "retrieval":
                            st.markdown(f"**Attempt {i+1} — Retrieval Retry**")
                            st.write(f"Reason: {entry.get('reason')}")
                            st.write(f"Action: {entry.get('action')}")
                        else:
                            st.markdown(f"**Attempt {i+1} — Generation (verdict: {entry.get('verdict')})**")
                            st.write(entry.get("answer"))
                            st.caption(f"Cosine: {entry.get('cosine')} · BERTScore: {entry.get('bert_score')}")
                        st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Retrieval Evaluation**")
                st.json({"Context Relevance": relevance, "Context Recall": recall})
            with col2:
                st.markdown("**Generation Evaluation**")

                st.json({
                "Final Verdict": eval_result["final_verdict"],
                "Cosine": eval_result["cosine"],
                "BERTScore": eval_result["bert_score"],
                "NLI": eval_result["nli"],
                "Fluency": eval_result["fluency"]
})

with tab2:
    st.subheader("Past Queries")

    if st.button("Refresh History"):
        st.rerun()

    logs = get_all_logs(limit=100)
    if not logs:
        st.info("No queries logged yet. Ask something in the first tab!")
    else:
        df = pd.DataFrame(logs)
        st.markdown("**Verdict Distribution (Display Verdict)**")
        verdict_col = "display_verdict" if "display_verdict" in df.columns else "final_verdict"
        st.bar_chart(df[verdict_col].value_counts())

        st.markdown("**Query Log**")
        display_cols = [
            "timestamp", "query", "display_verdict", "final_verdict",
            "cosine_score", "bert_score", "nli_label",
            "context_relevance_score", "context_recall_score",
            "retries_used"
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)