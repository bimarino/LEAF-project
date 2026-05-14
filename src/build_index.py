import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} prompts")
    return data

def build_documents_and_metadata(data):
    texts, ids, metadatas = [], [], []
    for item in data:
        title = item.get("title", "")
        content = item.get("content", "")
        difficulty = item.get("difficulty", "")

        texts.append(f"{title}\n\n{content}")
        ids.append(item["id"])
        metadatas.append({
            "title": title,
            "category": item.get("category", ""),
            "difficulty": difficulty,  # <-- this line
            "likes": item.get("likes", 0),
            "upvotes": item.get("upvotes", 0),
            "author_reputation": item.get("author_reputation", 0),
            "views": item.get("views", 0),
            "uses": item.get("uses", 0),
        })
    print(f"Built {len(texts)} documents")
    return ids, texts, metadatas

def build_chroma_index(ids, texts, metadatas):
    print(f"Loading model: {EMBEDDING_MODEL_NAME}")
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="prompts", embedding_function=embedding_function)
    batch_size = 512
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f"  Adding batch {start}-{end}...")
        collection.add(ids=ids[start:end], documents=texts[start:end], metadatas=metadatas[start:end])
    print("Done. Chroma index saved.")

if __name__ == "__main__":
    data = load_dataset()
    ids, texts, metadatas = build_documents_and_metadata(data)
    build_chroma_index(ids, texts, metadatas)
