# LEAF-BIANCA - Semantic Search over Prompt Dataset

## Project overview

This project implements a semantic search system over a dataset of prompts. Given a natural-language query, the system retrieves semantically similar prompts using an embedding model and a Chroma vector database for semantic search. A  metadata‑aware based on prompt "difficulty" is integrated into the reranking step to better match the user's intention.

## Repository structure
- data/
  - dataset.json: original prompt dataset
  - FIELDS.md: description of the dataset fields
- notebooks/
  - evaluation.ipynb: computes Precision@5 and Mean Reciprocal Rank (MRR) for vector‑only retrieval vs reranker + difficulty on a small labeled test set
  - demo.ipynb: code-based demo that runs example queries and prints ranked results (same behaviour as the app)
- src/
  - build_index.py: builds the Chroma index from data/dataset.json
  - search.py: runs semantic search with optional cross‑encoder reranker and difficulty metadata bonus
- app.py: Streamlit web app that provides a UI to run queries and see ranked prompts
- requirements.txt: Python dependencies

## Setup and installation

1. Download this repository into a local folder 
2. Open the folder (`LEAF-BIANCA`) in VS Code or terminal
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

All commands below assume they are run from the project root folder (`LEAF-BIANCA`) in a terminal where the `.venv` environment is active.

## How to run

### Build the index 

From the project root:

```bash
python src/build_index.py
```

This reads data/dataset.json, computes embeddings, and builds the Chroma index in chroma_db/

### Run search from the terminal

Vector‑only search:

```bash
python src/search.py --query "Explain research in simple words to my grandma" --k 5
```

Search with cross‑encoder reranker + difficulty metadata:

```bash
python src/search.py --query "Explain research in simple words to my grandma" --k 5 --rerank
```

### Run the Streamlit demo (UI)

To launch the web demo:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501), type a natural‑language query, choose k, and  enable the reranker to see ranked prompt results.

If preferred, run a code‑based demo:

- Open notebooks/demo.ipynb and run the cells to execute example queries and print ranked results.

## Models and reranking

- **Embedding model**: a sentence‑embedding model from Hugging Face is used to encode prompts and queries into dense vectors. These embeddings are stored and queried via the Chroma vector database using cosine similarity.
- **Reranker model**: a cross‑encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) is used as a reranker. It scores pairs (query, prompt) and reorders the top‑k candidates returned by the vector search.
- **Role of the reranker**: the embedding model efficiently finds a shortlist of semantically similar prompts. The cross‑encoder reranker then refines the ranking by looking at the full query–prompt pair often improving the position of the most relevant prompts.

## Evaluation results

Quantitative evaluation is implemented in notebooks/evaluation.ipynb on a small labeled set of two example queries. For each query the notebook compares:

- **Vector‑only retrieval** (embeddings + Chroma).
- **Reranker + difficulty metadata** (cross‑encoder, plus the difficulty bonus described below).

Using k = 5 the average metrics over the two test queries are:

- Precision@5: vector‑only ≈ 0.60, reranker + difficulty ≈ 0.60.
- MRR: vector‑only ≈ 0.75, reranker + difficulty ≈ 1.00.

This means both systems retrieve a similar number of relevant prompts in the top 5, but the reranker tends to move a relevant prompt to the very top position more often to improve the user experience.

Qualitative examples and full ranked lists for specific queries are shown in notebooks/demo.ipynb

## Metadata‑aware reranking 

In addition to the embedding‑based reranker, a simple metadata‑aware reranking step was implemented using the prompt difficulty field. For each query, the system looks for keywords such as “easy/beginner/basic/intro” and “advanced/hard/expert/difficult” and maps them to a target difficulty level (easy, medium, or hard). After the initial dense retrieval and reranking, prompts whose metadata difficulty matches this target receive a small positive bonus added to their reranker score, and the results are resorted. This keeps semantic similarity as the main signal while making the final ranking more aligned with the user’s explicit difficulty intent when it is expressed in the query.

## Further improvements

- Extend metadata‑aware reranking to include popularity signals in addition to difficulty.
- Add more labeled test queries to the evaluation notebook to obtain more robust metrics.
- Enrich the Streamlit UI with additional controls