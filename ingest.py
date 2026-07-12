import os
import re
import glob
import pickle
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DATA_DIR = "data"


def extract_text_column_aware(pdf_path):
    """
    Extracts text from a PDF in correct reading order, even for
    multi-column layouts. Standard extractors (like PyPDFLoader) read
    left-to-right across the FULL page width, which jumbles text from
    a left column with text from a right column mid-sentence.

    This function uses PyMuPDF's block-level extraction (each block has
    its own bounding box), then splits the page into left/right halves
    based on block x-position, and reads all of the left column top-to-
    bottom before moving to the right column -- matching how a human
    actually reads a two-column paper.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # each: (x0, y0, x1, y1, text, block_no, block_type)
        if not blocks:
            continue

        page_width = page.rect.width
        midpoint = page_width / 2

        left_blocks = [b for b in blocks if b[0] < midpoint]
        right_blocks = [b for b in blocks if b[0] >= midpoint]

        # If almost everything is on one "side", this page is likely
        # single-column -- just sort all blocks top-to-bottom instead.
        if len(left_blocks) < 2 or len(right_blocks) < 2:
            ordered_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        else:
            left_blocks.sort(key=lambda b: b[1])   # top to bottom
            right_blocks.sort(key=lambda b: b[1])  # top to bottom
            ordered_blocks = left_blocks + right_blocks

        page_text = "\n".join(b[4] for b in ordered_blocks if b[4].strip())
        full_text += page_text + "\n"

    doc.close()
    return full_text


def chunk_document(full_text, source_name):
    """
    Splits on structural headers (CHAPTER N, numbered sections like
    '5.2.1 Dataset Source'), then further splits long sections into
    manageable chunks. Each chunk is tagged with both its SOURCE
    document and its SECTION, and both are prepended into the chunk
    text itself so embeddings and keyword search can see them.
    """
    header_pattern = re.compile(
        r'(CHAPTER\s+\d+[A-Z\s]*|^\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n]{3,60})',
        re.MULTILINE
    )
    matches = list(header_pattern.finditer(full_text))

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append({"header": match.group().strip(), "text": full_text[start:end].strip()})

    if not sections:
        sections = [{"header": "Full Document", "text": full_text}]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for section in sections:
        for chunk_text in splitter.split_text(section["text"]):
            enriched_text = f"[Document: {source_name}] [Section: {section['header']}]\n{chunk_text}"
            chunks.append(Document(
                page_content=enriched_text,
                metadata={"source": source_name, "section": section["header"]}
            ))
    return chunks


def ingest_all_pdfs():
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}/ -- add at least one PDF first.")

    print(f"Found {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")

    all_chunks = []
    for pdf_path in pdf_paths:
        source_name = os.path.basename(pdf_path)
        print(f"\nProcessing: {source_name}")

        full_text = extract_text_column_aware(pdf_path)
        chunks = chunk_document(full_text, source_name)
        print(f"  -> {len(chunks)} chunks created")

        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Index saved to faiss_index/")

    with open("bm25_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("Chunks saved to bm25_chunks.pkl (for hybrid search)")


if __name__ == "__main__":
    ingest_all_pdfs()