# RAG (Vector + Graph) Demo

A minimal Retrieval-Augmented Generation ingestion/retrieval service with:

- Document upload and embedding.
- Vector retrieval (TF-IDF cosine similarity).
- Graph links between related chunks (sequence + shared-term edges).
- Web interface.
- **No access control / no login** (as requested).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

## Notes

- Upload plain-text-like files (`.txt`, `.md`, `.csv`, `.log`).
- Data is persisted in `rag_store.db`.
- This is a demo RAG backend/UI scaffold (not production hardening).
