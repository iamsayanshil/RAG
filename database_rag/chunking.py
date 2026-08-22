from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def main():
    # Step 1: Load Dataset (Updated dataset path format)
    print("1. Loading dataset from Hugging Face...")
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter short lines (taking first 100 valid entries for testing)
    documents = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50][:100]
    print(f"Loaded {len(documents)} valid text documents.")

    # Step 2: Chunk Text
    print("2. Chunking text using RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents(documents)
    chunk_texts = [doc.page_content for doc in docs]
    print(f"Generated {len(chunk_texts)} chunks.")

    # Step 3: Load Embedding Model
    print("3. Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Initialize Qdrant Local Database
    print("4. Setting up local Qdrant Vector Database...")
    client = QdrantClient(path="./qdrant_db")
    
    COLLECTION_NAME = "rag_documents"
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    # Step 5: Generate Embeddings
    print("5. Generating embeddings for chunks...")
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

    # Step 6: Push to Qdrant Database
    print("6. Pushing chunks to Qdrant...")
    points = []
    for idx, (text, vector) in enumerate(zip(chunk_texts, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"text": text}
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("\n✅ Success! All chunks have been vectorized and stored in Qdrant!")

if __name__ == "__main__":
    main()