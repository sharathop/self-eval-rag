import pickle

with open("bm25_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks: {len(chunks)}\n")

# Search for any chunk that mentions "Dataset Source" or "CICDDoS"
found = False
for i, doc in enumerate(chunks):
    if "Dataset Source" in doc.page_content or "CICDDoS" in doc.page_content:
        found = True
        print(f"--- Chunk {i} (section: {doc.metadata.get('section')}) ---")
        print(doc.page_content[:400])
        print()

if not found:
    print("NOT FOUND: No chunk contains 'Dataset Source' or 'CICDDoS' anywhere.")
    print("This means the structural splitter failed to isolate this section correctly.")