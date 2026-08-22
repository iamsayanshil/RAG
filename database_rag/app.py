import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_active_chat_model(groq_client):
    try:
        models = groq_client.models.list()
        for m in models.data:
            model_id = m.id.lower()
            if "guard" not in model_id and "whisper" not in model_id and "vision" not in model_id:
                if any(name in model_id for name in ["llama", "gemma", "mixtral", "deepseek", "qwen"]):
                    return m.id
        for m in models.data:
            if "guard" not in m.id.lower():
                return m.id
        return models.data[0].id
    except Exception:
        return "llama-3.3-70b-versatile"

def ask_rag_with_citations(user_query: str) -> dict:
    client = QdrantClient(path="./qdrant_db")
    COLLECTION_NAME = "rag_documents"

    # 1. Query Embedding
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode(user_query).tolist()

    # 2. Retrieve Documents
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    results = response.points

    if not results:
        return {
            "answer": "No relevant documents found in the database.",
            "citations": []
        }

    # 3. Process Context & Build Citations
    retrieved_chunks = []
    citations = []

    for idx, hit in enumerate(results):
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "Unknown Document")
        chunk_id = hit.payload.get("chunk_id", idx)
        
        retrieved_chunks.append(text)
        citations.append({
            "source": source,
            "chunk_id": chunk_id,
            "snippet": text[:150] + "..."  # First 150 chars preview
        })

    context = "\n---\n".join(retrieved_chunks)

    # 4. Generate Answer via Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    active_model = get_active_chat_model(groq_client)

    chat_completion = groq_client.chat.completions.create(
        model=active_model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise AI assistant. Answer the question strictly using only the provided context. If the answer cannot be deduced from the context, state that you do not know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
            }
        ],
        temperature=0.2
    )

    answer = chat_completion.choices[0].message.content

    return {
        "answer": answer,
        "citations": citations
    }

if __name__ == "__main__":
    query = "Summarize the key information from the uploaded documents."
    result = ask_rag_with_citations(query)
    
    print("\n=== AI ANSWER ===")
    print(result["answer"])
    
    print("\n=== SOURCES & CITATIONS ===")
    for c in result["citations"]:
        print(f"📄 Source: {c['source']} (Chunk #{c['chunk_id']})")
        print(f"   Snippet: {c['snippet']}\n")