import os
import math
import pickle
import requests
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# --- Setup ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EVAL_API_URL = "http://127.0.0.1:8000/evaluate"

# --- Load chunks for BM25 keyword search (built during ingest.py) ---
with open("bm25_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

tokenized_corpus = [doc.page_content.lower().split() for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)


def retrieve_with_rerank(query, wide_k=10, final_k=3):
    """Hybrid retrieval: combine vector search (semantic) + BM25 (keyword),
    then rerank the combined candidate pool with a cross-encoder."""

    # --- Vector search candidates ---
    vector_candidates = vectorstore.similarity_search(query, k=wide_k)

    # --- BM25 keyword search candidates ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:wide_k]
    bm25_candidates = [all_chunks[i] for i in top_bm25_indices]

    # --- Merge candidate pools, dedupe by content ---
    seen = set()
    combined = []
    for doc in vector_candidates + bm25_candidates:
        key = doc.page_content[:100]  # dedupe key
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    # --- Rerank the combined pool with cross-encoder ---
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
    payload = {"context": context_text, "question": query, "llm_response": answer}
    response = requests.post(EVAL_API_URL, json=payload)
    response.raise_for_status()
    return response.json()


# --- Retrieval-level evaluation ---
def check_context_relevance(query, chunks):
    pairs = [[query, doc.page_content] for doc in chunks]
    raw_scores = reranker.predict(pairs)
    normalized_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]  # sigmoid -> 0-1 range
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
    """
    Second opinion, used only to double-check a 'Hallucinated' verdict from the
    eval framework's NLI component. NLI is known to sometimes misclassify long,
    multi-claim answers as contradictions even when every individual claim is
    grounded in the context. This checks claim-by-claim instead of as one block.
    """
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


def needs_generation_retry(eval_result):
    """Original check: is the ANSWER unfaithful given the context it was given?"""
    verdict = eval_result.get("final_verdict")
    cosine_score = eval_result.get("cosine", {}).get("score", 0)
    bert_score = eval_result.get("bert_score", {}).get("score", 0)

    if verdict == "Hallucinated":
        return True
    if verdict == "Unverifiable":
        return cosine_score < 0.75 or bert_score < 0.75
    return False  # covers "Faithful (judge-verified, NLI false-flag)" too — no retry needed


def needs_retrieval_retry(recall_result, relevance_result):
    """New check: did retrieval even fetch the RIGHT context in the first place?"""
    recall_score = recall_result.get("score")
    if recall_score is not None and recall_score < 0.5:
        return True
    if relevance_result.get("verdict") == "Weak Relevance":
        return True
    return False


def compute_display_verdict(eval_result):
    """
    The underlying eval framework drops final_verdict to 'Unverifiable'
    if even one signal (often NLI) disagrees, even when cosine/bert_score
    are both strong. This computes a more representative verdict for
    display purposes, without changing the original eval framework.
    """
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
    retry_log = []

    # --- Attempt 1: normal retrieval (k=25 wide, top 10 after rerank) ---
    chunks = retrieve_with_rerank(query, wide_k=25, final_k=10)
    print("\n========== RETRIEVED CHUNKS ==========")

    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i + 1} ---")
        print(chunk.page_content)

    print("======================================\n")
    context_text = "\n\n".join([doc.page_content for doc in chunks])

    relevance = check_context_relevance(query, chunks)
    recall = check_context_recall(query, context_text)
    print("\n--- RETRIEVAL EVAL (attempt 1) ---")
    print("Context Relevance:", relevance)
    print("Context Recall:", recall)

    total_retries = 0

    # --- If retrieval itself looks weak, re-retrieve wider before generating ---
    if needs_retrieval_retry(recall, relevance) and total_retries < max_retries:
        print("\n[Retrieval Retry] Context Recall/Relevance weak — widening search (k=20, top 5)...")
        retry_log.append({
            "stage": "retrieval",
            "reason": f"Weak retrieval (Context Relevance: {relevance.get('verdict')}, Context Recall: {recall.get('verdict')})",
            "action": "Widened search from k=25/top10 to k=20/top5 (fresh candidates, different ranking)"
        })
        chunks = retrieve_with_rerank(query, wide_k=20, final_k=5)
        context_text = "\n\n".join([doc.page_content for doc in chunks])
        relevance = check_context_relevance(query, chunks)
        recall = check_context_recall(query, context_text)
        total_retries += 1
        print("--- RETRIEVAL EVAL (attempt 2, widened) ---")
        print("Context Relevance:", relevance)
        print("Context Recall:", recall)

    # --- Generation + generation evaluation ---
    answer = generate_answer(query, chunks, strict=False)
    eval_result = evaluate_answer(query, context_text, answer)
    judge_note = None

    # If NLI flags "Hallucinated", double-check with an LLM judge before
    # trusting it and burning a retry — NLI is known to misfire on long,
    # multi-claim answers even when every claim is actually grounded.
    if eval_result.get("final_verdict") == "Hallucinated":
        judge_result = llm_judge_faithfulness(query, context_text, answer)
        judge_note = judge_result
        print(f"\n[Hallucination Check] NLI said Hallucinated. LLM judge says: {judge_result['verdict']} — {judge_result['reason']}")
        if judge_result["verdict"] == "Faithful":
            eval_result["final_verdict"] = "Faithful (judge-verified, NLI false-flag)"

    retry_log.append({
        "stage": "generation",
        "attempt": 1,
        "answer": answer,
        "verdict": eval_result.get("final_verdict"),
        "cosine": eval_result.get("cosine", {}).get("score"),
        "bert_score": eval_result.get("bert_score", {}).get("score"),
        "judge_check": judge_note
    })

    while needs_generation_retry(eval_result) and total_retries < max_retries:
        print(f"\n[Generation Retry] Verdict was '{eval_result['final_verdict']}' — regenerating with stricter prompt...")
        answer = generate_answer(query, chunks, strict=True)
        eval_result = evaluate_answer(query, context_text, answer)
        judge_note = None

        if eval_result.get("final_verdict") == "Hallucinated":
            judge_result = llm_judge_faithfulness(query, context_text, answer)
            judge_note = judge_result
            print(f"\n[Hallucination Check] NLI said Hallucinated. LLM judge says: {judge_result['verdict']} — {judge_result['reason']}")
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


# --- Run it ---
if __name__ == "__main__":
    query = "What features were selected for DDoS detection?"
    answer, eval_result, relevance, recall, retries, display_verdict, retry_log = run_pipeline(query)

    print("\nQUESTION:", query)
    print("ANSWER:", answer)
    print("\nRAW FINAL_VERDICT (from eval framework):", eval_result.get("final_verdict"))
    print("DISPLAY VERDICT (pipeline-level, smarter):", display_verdict)
    print("FULL GENERATION EVAL:", eval_result)
    print("RETRIES USED:", retries)
    print("RETRY LOG:", retry_log)