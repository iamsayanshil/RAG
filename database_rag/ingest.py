import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "rag_documents"
QDRANT_PATH = "./qdrant_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_file(file_path: str) -> str:
    """Extracts raw text from PDF, TXT, or CSV files."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif ext in [".txt", ".csv"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return text

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks for better embedding retention."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def process_and_store_document(file_path: str):
    """Parses, embeds, and uploads file chunks to Qdrant."""
    print(f"Reading file: {file_path}...")
    raw_text = extract_text_from_file(file_path)
    chunks = create_chunks(raw_text)
    
    print(f"Generated {len(chunks)} text chunks.")

    client = QdrantClient(path=QDRANT_PATH)
    
    # Re-create collection if not exists
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    points = []
    source_filename = os.path.basename(file_path)

    for idx, chunk in enumerate(chunks):
        vector = embedding_model.encode(chunk).tolist()
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source_filename,
                    "chunk_id": idx
                }
            )
        )

    print("Uploading vector embeddings to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Document successfully indexed into Qdrant!")

if __name__ == "__main__":
    # Test with any local PDF or file
    sample_file = "sample.pdf"  # আপনার ফোল্ডারে থাকা কোনো পিডিএফ ফাইলের নাম দিন
    if os.path.exists(sample_file):
        process_and_store_document(sample_file)
    else:
        print(f"Please place a file named '{sample_file}' in your directory to test ingestion.")