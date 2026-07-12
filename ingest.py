import re
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Load document
loader = PyPDFLoader("data/doc.pdf")
docs = loader.load()
print(f"Total pages loaded: {len(docs)}")

# 2. Join all pages into one big text blob (keeps section headers from being
#    split across page boundaries)
full_text = ""
for doc in docs:
    full_text += doc.page_content + "\n"

# 3. Split on structural headers: "CHAPTER 1", "5.2.1 Dataset Source", "1.3 Objectives", etc.
header_pattern = re.compile(
    r'(CHAPTER\s+\d+[A-Z\s]*|^\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n]{3,60})',
    re.MULTILINE
)

matches = list(header_pattern.finditer(full_text))
print(f"Found {len(matches)} structural headers")

sections = []
for i, match in enumerate(matches):
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
    header_text = match.group().strip()
    section_text = full_text[start:end].strip()
    sections.append({"header": header_text, "text": section_text})

# Fallback: if no headers were found, treat the whole document as one section
if not sections:
    sections = [{"header": "Full Document", "text": full_text}]

# 4. Within each section, further split into manageable chunks.
#    Each chunk carries its section header as metadata AND has the header
#    prepended to its text, so both embeddings and keyword search see it.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []
for section in sections:
    sub_chunks = splitter.split_text(section["text"])
    for chunk_text in sub_chunks:
        enriched_text = f"[Section: {section['header']}]\n{chunk_text}"
        chunks.append(Document(
            page_content=enriched_text,
            metadata={"section": section["header"]}
        ))

print(f"Created {len(chunks)} chunks across {len(sections)} sections")

# 5. Embed + build FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Index saved to faiss_index/")

# 6. Save raw chunks for BM25 (keyword search) use later
with open("bm25_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Chunks saved to bm25_chunks.pkl (for hybrid search)")