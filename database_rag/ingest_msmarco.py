import chromadb
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("1. Downloading lightweight dataset (Will take a few seconds)...")
# SQuAD is a fast, lightweight Q&A dataset
dataset = load_dataset("rajpurkar/squad", split="train[:300]")

print("2. Setting up Vector Database...")
chroma_client = chromadb.PersistentClient(path="./voice_rag_db")
collection = chroma_client.get_or_create_collection(name="voice_knowledge_base")

# Chunking Setup
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

documents, metadatas, ids = [], [], []

print("3. Chunking the text data...")
for index, item in enumerate(dataset):
    context = item.get("context", "")
    title = item.get("title", "")
    
    if context.strip():
        chunks = text_splitter.split_text(context)
        for c_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"title": title, "doc_id": index})
            ids.append(f"doc_{index}_c{c_idx}")

print(f"4. Saving {len(documents)} chunks to Vector DB...")
batch_size = 100
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i + batch_size],
        metadatas=metadatas[i:i + batch_size],
        ids=ids[i:i + batch_size]
    )

print("\n✅ Data Ingestion Complete! Vector DB saved in './voice_rag_db'")