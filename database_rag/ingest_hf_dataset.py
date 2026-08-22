from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "rag_documents"
QDRANT_PATH = "./qdrant_db"

print("1. Loading SentenceTransformer Model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_msmarco_bengali(limit=1000):
    print("2. Streaming dataset from HuggingFace (ai4bharat/MSMARCO-XI)...")
    
    # Using 'default' configuration in streaming mode to process data dynamically
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)

    client = QdrantClient(path=QDRANT_PATH)
    
    # Create Qdrant collection if it does not exist
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    points = []
    count = 0

    print(f"3. Filtering and processing Bengali passages into Qdrant Vector DB (Target: {limit} chunks)...")
    
    for idx, example in enumerate(dataset):
        if count >= limit:
            break
            
        # Check target language metadata to filter Bengali content
        target_lang = str(example.get("target_lang", ""))
        
        # Filter records for Bengali language codes (e.g., ben_Beng, bn, etc.)
        if "ben" in target_lang or "bn" in target_lang or target_lang == "":
            passages = example.get("passages", {}).get("Translated_passages", [])
            
            for p_idx, passage in enumerate(passages):
                if not passage or len(passage.strip()) < 10:
                    continue

                # Generate vector embedding for each passage chunk
                vector = embedding_model.encode(passage).tolist()

                points.append(
                    PointStruct(
                        id=count,
                        vector=vector,
                        payload={
                            "text": passage,
                            "query_id": example.get("query_id"),
                            "source": "MSMARCO-XI-Bengali"
                        }
                    )
                )
                count += 1
                if count >= limit:
                    break

    print(f"4. Uploading {len(points)} passage chunks to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ MSMARCO Bengali Dataset successfully indexed into Qdrant!")

if __name__ == "__main__":
    # Ingest the first 1000 passages for testing
    ingest_msmarco_bengali(limit=1000)