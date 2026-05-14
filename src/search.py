import argparse
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from sentence_transformers import CrossEncoder
# Paths and config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker

def get_collection():
    """Connect to the existing Chroma collection."""
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="prompts",
        embedding_function=embedding_function,
    )
    return collection


def search(query: str, top_k: int = 5, use_reranker: bool = False):
    """Run a semantic search over the prompts, optionally with reranking."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    if not use_reranker:
        return results

    # Rerank the retrieved candidates
    reranker = get_reranker()
    docs = results["documents"][0]
    ids = results["ids"][0]
    metas = results["metadatas"][0]

    # Prepare (query, doc) pairs and get reranker scores
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    # --- Difficulty-based metadata bonus ---
    q = query.lower()
    if any(word in q for word in ["easy", "beginner", "basic", "intro"]):
        target_difficulty = "beginner"
    elif any(word in q for word in ["advanced", "hard", "expert", "difficult"]):
        target_difficulty = "advanced"
    else:
        target_difficulty = "intermediate"

    BONUS = 0.05

    # Build scored_items WITH the bonus already included in the score
    scored_items = []
    for id_, doc, meta, score in zip(ids, docs, metas, scores):
        meta_diff = meta.get("difficulty") if isinstance(meta, dict) else None
        bonus = BONUS if meta_diff == target_difficulty else 0.0
        base_score = -float(score)          # lower distance -> higher base_score
        scored_items.append((id_, doc, meta, base_score + bonus))

    # Sort by score + bonus (descending) — fix: use scored_items, not raw scores
    ranked = sorted(
        scored_items,
        key=lambda x: x[3],
        reverse=True,
    )

    # Build results dict in the same format as Chroma
    reranked_results = {
        "ids": [[r[0] for r in ranked]],
        "documents": [[r[1] for r in ranked]],
        "metadatas": [[r[2] for r in ranked]],
        "scores": [[float(r[3]) for r in ranked]],
    }
    return reranked_results

def print_results(results):
    """Pretty-print results returned by Chroma."""
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("scores", [[]])[0]  # use 'scores' because we reranked

    for rank, (id_, doc, meta, score) in enumerate(
        zip(ids, docs, metas, scores), start=1
    ):
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        print(f"\nRank {rank}")
        print(f"ID: {id_}")
        print(f"Title: {title}")
        print(f"Score: {score}")
        print("Snippet:")
        print(doc[:300], "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over prompts")
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results to return (top_k)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use cross-encoder reranker on top of vector search",
    )
    args = parser.parse_args()

    print(f"Running search for: {args.query!r}")
    res = search(args.query, top_k=args.k, use_reranker=args.rerank)
    print_results(res)