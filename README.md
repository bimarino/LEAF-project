# LEAF-BIANCA - Semantic Search over Prompts Dataset

## Project overview

The aim of this project is to optimize a prompts dataset by implementing a semantic search tool to ensure interpretability and coherence to the user’s experience. The semantic search consists in a process where prompts are retrieved based on semantic similarity after providing a query in natural-language. This is done through the use of an embedding model and Chroma vector database. Lastly, to enhance the user’s experience, a metadata aware based on difficulty was incorporated within the reranking step. 


## Repository structure
- data/
  - dataset.json: original prompt dataset
  - FIELDS.md: description of the dataset fields
- chroma_db/
  - Persistent Chroma vector index built from dataset.json by src/build_index.py
- notebooks/
  - evaluation.ipynb: computes Precision@5 and Mean Reciprocal Rank (MRR) for vector‑only retrieval vs reranker + difficulty on a small labeled test set
  - demo.ipynb: code-based demo that runs example queries and prints ranked results (same behaviour as the app)
- src/
  - build_index.py: builds the Chroma index from data/dataset.json using SentenceTransformerEmbeddingFunction
  - search.py: runs semantic search with optional cross‑encoder reranker and difficulty metadata bonus
- app.py: Streamlit web app that provides a UI to run queries and see ranked prompts
- requirements.txt: Python dependencies

## Setup and installation

1. Download this repository into a local folder 
2. Open the folder (LEAF-BIANCA) in terminal
3. Create the virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

4. Install dependencies:

   pip install -r requirements.txt

(To run commands below assume to run from the project root folder (LEAF-BIANCA) in a terminal where the `.venv` environment is active)

##To run
From project root:

python src/build_index.py

This reads data/dataset.json, computes embeddings, and builds the Chroma index in chroma_db/

### Run search from the terminal

Vector‑only search:

python src/search.py --query "Explain research in simple words to my grandma" --k 5

Search with cross‑encoder reranker + difficulty metadata:

python src/search.py --query "Explain research in simple words to my grandma" --k 5 --rerank

### Run the Streamlit demo (UI) by

streamlit run app.py

It will show something like (http://localhost:8501) then type a natural‑language query.
You can choose (number)k and see ranked prompt results 

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

After the embedding reranker a metadata aware was added on prompt difficulty. When looking at each query the system will try to kind the following words: “easy/beginner/ibasic/intro” , “advanced/hard/expert/difficult”. Then allocates these to a certain difficulty level among (easy,medium,hard). After the reranking the prompts that were matched with the metadata correctly will be added a bonus to the reranker score. With this increase, the scores will be logically reranked.  This is done in order to unsure semantic similarity , to improve the final reranking and keep coherence with the user's difficulty query. 

## Further improvements

- Extend metadata‑aware reranking to include popularity signals in addition to difficulty.
- Add more labeled test queries to the evaluation notebook to obtain more robust metrics.
- Enrich the Streamlit UI with additional controls