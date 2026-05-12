# LEAF-BIANCA: Semantic Search over Prompt Dataset

## 1. Project overview
- One paragraph: what the system does (semantic search over a dataset of prompts, optional reranker to improve ranking).

## 2. Repository structure
- Brief bullets explaining:
  - `data/` (dataset.json, FIELDS.md)
  - `chroma_db/` (vector index built by build_index.py)
  - `src/build_index.py` (builds Chroma index from dataset)
  - `src/search.py` (runs search with optional reranker)
  - `evaluation.ipynb` (computes Precision@5 and MRR with vs without reranker)
  - `requirements.txt` (Python dependencies)

## 3. Setup and installation
- Explain:
  - Python version you used
  - How to create and activate `.venv`
  - How to install packages from `requirements.txt`
  ## 3. Setup and installation

1. Clone or download this repository into a local folder (for example on your Desktop).
2. Open the folder in VS Code.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

All commands should be run from the project root folder (`LEAF-BIANCA`) in a terminal where the `.venv` environment is active.

## 4. How to run

### 4.1 Build the index (once)

From the project root (`LEAF-BIANCA`) with the virtual environment active:

```bash
python src/build_index.py
```

This reads `data/dataset.json` and builds the Chroma index in `chroma_db/`.

### 4.2 Run search from the terminal

Vector-only search:

```bash
python src/search.py --query "Explain research in simple words to my grandma" --k 5
```

Search with reranker:

```bash
python src/search.py --query "Explain research in simple words to my grandma" --k 5 --rerank
```

### 4.3 Run the Streamlit demo

To launch the web demo:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`), type a query, choose `k`, and optionally enable the reranker to see ranked results.

## 5. Model and reranker
- Short explanation of:
  - Embedding model name
  - Cross-encoder reranker model name
  - Conceptual role of reranker (reorders initial results).

## 6. Evaluation results
- Mention:
  - small labeled set (2 queries)
  - Precision@5 and MRR for vector vs rerank (using the averages from `evaluation.ipynb`).

## 7. Future work / extensions
- One or two bullets:
  - metadata-aware ranking (likes, upvotes)
  - Streamlit demo UI
