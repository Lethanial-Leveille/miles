"""Sentence embeddings for semantic memory retrieval.

Keyword search has a ceiling that no amount of tuning clears. "Who am I
closest to" never matches "Marlo is who he goes to when something is wrong",
because they share no words, and "fuzzy" appears in nine memories while "time"
appears in eight, so frequency cannot tell a company name from filler. Both
failures are the same thing: word overlap is a proxy for relevance and it
leaks.

An embedding maps text to a point in space chosen so that similar meanings land
near each other, and similarity is the cosine between two of those points.
That is the same operation Resemblyzer already performs on voice clips in this
codebase, on text instead of audio.

Model is all-MiniLM-L6-v2: 384 dimensions, around 80MB, 31ms per query on this
Pi, which is 0.7 percent of a 4298ms turn.

Vectors live in SQLite as float32 blobs. At 384 dimensions that is 1536 bytes
each, so the whole corpus is well under a megabyte, and brute force cosine over
a few hundred rows in numpy costs less than the SQLite round trip that fetched
them. A vector index would be machinery bought for a problem that does not
exist at this size.
"""

import sqlite3

import numpy as np

from config import DB_PATH, EMBEDDING_MODEL

_model = None


def get_model():
    """Load the model once, on first use rather than at import.

    Import time matters here: database.py is imported by the CLI tools and the
    test suite, and neither should pay a multi second model load to list a
    memory or assert on prompt assembly."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(texts):
    """Unit length float32 vectors, one row per text.

    Normalising here means cosine similarity is a plain dot product later, so
    nothing downstream has to remember to divide by magnitudes."""
    vectors = get_model().encode(list(texts), normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def store(memory_id, vector, conn=None):
    owns = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (memory_id, model, vector) "
        "VALUES (?, ?, ?)",
        (memory_id, EMBEDDING_MODEL, vector.astype(np.float32).tobytes()))
    if owns:
        conn.commit()
        conn.close()


def backfill(verbose=True):
    """Embed every active memory that does not yet have a current vector.

    Keyed on model as well as memory id, so changing EMBEDDING_MODEL re embeds
    rather than silently mixing vectors from two different spaces, which would
    produce similarity scores that look fine and mean nothing."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT m.id, m.content FROM memories m "
        "LEFT JOIN memory_embeddings e "
        "  ON e.memory_id = m.id AND e.model = ? "
        "WHERE m.status = 'active' AND e.memory_id IS NULL", (EMBEDDING_MODEL,)
    ).fetchall()
    if not rows:
        conn.close()
        return 0

    vectors = encode([content for _, content in rows])
    for (memory_id, _), vector in zip(rows, vectors):
        store(memory_id, vector, conn)
    conn.commit()
    conn.close()
    if verbose:
        print(f"embedded {len(rows)} memories with {EMBEDDING_MODEL}")
    return len(rows)


def search(query, limit=5, tier=None, min_similarity=0.25):
    """Rank active memories by cosine similarity. Returns (id, content, category).

    min_similarity exists because cosine always returns a nearest neighbour.
    Ask about the weather and something is still closest; it is just closest to
    nothing in particular. Without a floor, semantic search answers every
    question confidently, which is the failure mode keyword search does not
    have and the one people forget to guard.
    """
    conn = sqlite3.connect(DB_PATH)
    sql = ("SELECT m.id, m.content, m.category, e.vector "
           "FROM memory_embeddings e JOIN memories m ON m.id = e.memory_id "
           "WHERE e.model = ? AND m.status = 'active' ")
    params = [EMBEDDING_MODEL]
    if tier is not None:
        sql += "AND m.tier = ? "
        params.append(tier)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    if not rows:
        return []

    matrix = np.frombuffer(b"".join(r[3] for r in rows),
                           dtype=np.float32).reshape(len(rows), -1)
    similarities = matrix @ encode([query])[0]

    order = np.argsort(-similarities)[:limit]
    return [(rows[i][0], rows[i][1], rows[i][2])
            for i in order if similarities[i] >= min_similarity]
