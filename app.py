from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH = Path("rag_store.db")

app = FastAPI(title="RAG Vector + Graph Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class HybridStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    source_chunk_id INTEGER NOT NULL,
                    target_chunk_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (source_chunk_id, target_chunk_id, relation)
                );
                """
            )

    def add_document(self, filename: str, content: str) -> None:
        text = content.strip()
        if not text:
            return

        chunks = self._chunk_text(text)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=256)
        matrix = vectorizer.fit_transform(chunks)
        embeddings = matrix.toarray()

        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (filename, content) VALUES (?, ?)",
                (filename, text),
            )
            doc_id = int(cursor.lastrowid)
            chunk_ids: list[int] = []

            for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings, strict=False)):
                c = conn.execute(
                    """
                    INSERT INTO chunks (doc_id, position, content, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doc_id, idx, chunk_text, json.dumps(vector.tolist())),
                )
                chunk_ids.append(int(c.lastrowid))

            for i in range(len(chunk_ids) - 1):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO graph_edges (source_chunk_id, target_chunk_id, relation, weight)
                    VALUES (?, ?, 'sequence', 1.0)
                    """,
                    (chunk_ids[i], chunk_ids[i + 1]),
                )

            self._link_keyword_neighbors(conn, chunk_ids, chunks)

    def _link_keyword_neighbors(self, conn: sqlite3.Connection, chunk_ids: list[int], chunks: list[str]) -> None:
        token_map: dict[str, list[int]] = {}
        for idx, chunk_text in enumerate(chunks):
            tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", chunk_text)}
            for token in tokens:
                token_map.setdefault(token, []).append(chunk_ids[idx])

        for token, ids in token_map.items():
            if len(ids) < 2:
                continue
            for i in range(len(ids) - 1):
                for j in range(i + 1, len(ids)):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO graph_edges (source_chunk_id, target_chunk_id, relation, weight)
                        VALUES (?, ?, 'shared_term', 0.7)
                        """,
                        (ids[i], ids[j]),
                    )

    def _chunk_text(self, text: str, max_chunk_words: int = 120) -> list[str]:
        words = text.split()
        if len(words) <= max_chunk_words:
            return [text]
        return [" ".join(words[i : i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

    def _get_chunks(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, doc_id, position, content, embedding FROM chunks"
            ).fetchall()
        return rows

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        rows = self._get_chunks()
        if not rows:
            return []

        corpus = [r["content"] for r in rows]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=256)
        matrix = vectorizer.fit_transform(corpus + [query])
        query_vec = matrix[-1].toarray()[0]

        scored: list[tuple[sqlite3.Row, float]] = []
        for i, row in enumerate(rows):
            chunk_vec = matrix[i].toarray()[0]
            score = self._cosine(query_vec, chunk_vec)
            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        graph = self._load_graph()
        results: list[dict] = []
        for row, score in top:
            neighbors = [
                {
                    "id": n,
                    "relation": graph[row["id"]][n]["relation"],
                    "weight": graph[row["id"]][n]["weight"],
                }
                for n in graph.neighbors(row["id"])
            ] if graph.has_node(row["id"]) else []
            results.append(
                {
                    "chunk_id": row["id"],
                    "doc_id": row["doc_id"],
                    "position": row["position"],
                    "score": round(score, 4),
                    "content": row["content"],
                    "neighbors": neighbors[:3],
                }
            )
        return results

    def _load_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        with self._conn() as conn:
            chunks = conn.execute("SELECT id FROM chunks").fetchall()
            for row in chunks:
                graph.add_node(row["id"])
            edges = conn.execute(
                "SELECT source_chunk_id, target_chunk_id, relation, weight FROM graph_edges"
            ).fetchall()
            for edge in edges:
                graph.add_edge(
                    edge["source_chunk_id"],
                    edge["target_chunk_id"],
                    relation=edge["relation"],
                    weight=edge["weight"],
                )
        return graph

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "edges": edges}


store = HybridStore(DB_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stats": store.stats(),
            "results": [],
            "query": "",
            "message": "No authentication/login is enabled for this demo.",
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1", errors="ignore")

    store.add_document(file.filename or "uploaded.txt", content)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stats": store.stats(),
            "results": [],
            "query": "",
            "message": f"Indexed document: {file.filename}",
        },
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, query: str = Form(...)) -> HTMLResponse:
    results = store.search(query)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stats": store.stats(),
            "results": results,
            "query": query,
            "message": "Hybrid retrieval across vector similarity + graph links.",
        },
    )
