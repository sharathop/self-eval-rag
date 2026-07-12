from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# Load your existing FAISS index (no changes here)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load the reranker model (downloads once, then cached)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_with_rerank(query, wide_k=10, final_k=3):
    # Step A: cast a wide net using basic vector similarity
    candidates = vectorstore.similarity_search(query, k=wide_k)

    # Step B: rerank those candidates more carefully
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    # Step C: sort by rerank score, keep top final_k
    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:final_k]]

    return top_docs

# Test it
query = "What features were selected for DDoS detection?"
results = retrieve_with_rerank(query)

for i, doc in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:300])