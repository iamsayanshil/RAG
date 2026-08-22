from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def query_rag(query_text, top_k=3):
    # 1. Local Qdrant Database connect
    client = QdrantClient(path="./qdrant_db")
    COLLECTION_NAME = "rag_documents"

    # 2. Embedding Model load
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Convert query text to vector
    print(f"\nSearching for: '{query_text}'...")
    query_vector = embedding_model.encode(query_text).tolist()

    # 4. Search in Qdrant Vector DB
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    # 5. Display results
    print("\n--- Search Results ---")
    for idx, hit in enumerate(results, 1):
        print(f"\n[Result {idx}] (Score: {hit.score:.4f})")
        print(hit.payload['text'])

if __name__ == "__main__":
    # Test with a question
    user_query = "Tell me about history or historical events"
    query_rag(user_query, top_k=3)