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

import database
from config import EMBEDDING_MODEL

_model = None


def _db(db_path):
    """database.DB_PATH read at call time, not config's copy taken at import.
    The test fixture points database at a throwaway file; a copy taken here at
    import would keep pointing at the real one, and write vectors into it."""
    return db_path or database.DB_PATH


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
    conn = conn or sqlite3.connect(_db(None))
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (memory_id, model, vector) "
        "VALUES (?, ?, ?)",
        (memory_id, EMBEDDING_MODEL, vector.astype(np.float32).tobytes()))
    if owns:
        conn.commit()
        conn.close()


def backfill(verbose=True, db_path=None):
    """Embed every active or pending memory that does not yet have a current
    vector.

    Keyed on model as well as memory id, so changing EMBEDDING_MODEL re embeds
    rather than silently mixing vectors from two different spaces, which would
    produce similarity scores that look fine and mean nothing.

    Called before every search since Sep 28 2026. Until then nothing called it
    at all: it had been run by hand, so every memory stored after Sep 13 had no
    vector, and search by meaning could not find a single thing he had told Nova
    since. Running it where vectors are read, rather than at each place a memory
    is written, covers every writer, including ones added later. The check is
    one query and finds nothing on almost every turn.

    Pending as well as active, so a new fact can be compared against one still
    waiting for review."""
    conn = sqlite3.connect(_db(db_path))
    rows = conn.execute(
        "SELECT m.id, m.content FROM memories m "
        "LEFT JOIN memory_embeddings e "
        "  ON e.memory_id = m.id AND e.model = ? "
        "WHERE m.status IN ('active', 'pending') AND e.memory_id IS NULL",
        (EMBEDDING_MODEL,)
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


def search(query, limit=5, tier=None, min_similarity=0.25, db_path=None):
    """Rank active memories by cosine similarity. Returns (id, content, category).

    min_similarity exists because cosine always returns a nearest neighbour.
    Ask about the weather and something is still closest; it is just closest to
    nothing in particular. Without a floor, semantic search answers every
    question confidently, which is the failure mode keyword search does not
    have and the one people forget to guard.
    """
    conn = sqlite3.connect(_db(db_path))
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


def _vectors(db_path, statuses):
    marks = ", ".join("?" for _ in statuses)
    conn = sqlite3.connect(_db(db_path))
    rows = conn.execute(
        "SELECT m.id, m.content, m.status, e.vector "
        "FROM memory_embeddings e JOIN memories m ON m.id = e.memory_id "
        f"WHERE e.model = ? AND m.status IN ({marks}) ORDER BY m.id",
        (EMBEDDING_MODEL, *statuses)).fetchall()
    conn.close()
    if not rows:
        return rows, None
    return rows, np.frombuffer(b"".join(r[3] for r in rows),
                               dtype=np.float32).reshape(len(rows), -1)


def nearest(text, db_path=None):
    """The stored memory closest in meaning to text, active or pending, as
    (id, content, similarity), or None when nothing is stored.

    Backfills first, so a memory written a second ago is compared too: on Sep 28
    2026 the pass queued a rewording of a fact the turn had stored two seconds
    earlier."""
    backfill(verbose=False, db_path=db_path)
    rows, matrix = _vectors(db_path, ("active", "pending"))
    if not rows:
        return None
    similarities = matrix @ encode([text])[0]
    best = int(np.argmax(similarities))
    return rows[best][0], rows[best][1], float(similarities[best])


def similar_pairs(threshold, db_path=None, dismissed=frozenset()):
    """Every pair of active memories at or above threshold, closest first, as
    (similarity, (id, content), (id, content)).

    A list for him to read, never something acted on: measured Sep 28 2026, a
    duplicate and two different facts about the same person score the same, so
    only a reader can say which a pair is. See docs/BRAIN.md.

    dismissed holds frozensets of ids he has already called different facts."""
    backfill(verbose=False, db_path=db_path)
    rows, matrix = _vectors(db_path, ("active",))
    if not rows:
        return []
    scores = matrix @ matrix.T
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if scores[i, j] >= threshold and \
                    frozenset((rows[i][0], rows[j][0])) not in dismissed:
                pairs.append((float(scores[i, j]), rows[i][:2], rows[j][:2]))
    return sorted(pairs, key=lambda p: -p[0])


def closest_stored(memory_ids, threshold, db_path=None, dismissed=frozenset()):
    """For each of memory_ids, the closest active memory at or above threshold,
    as {id: (id, content, similarity)}. Ids with nothing close are left out.

    For the review queue: a pending fact is shown beside what it may repeat or
    update, so he decides with both in front of him. Active memories only, all
    tiers, since a fact the prompt never carries can still be repeated."""
    backfill(verbose=False, db_path=db_path)
    wanted = set(memory_ids)
    rows, matrix = _vectors(db_path, ("active", "pending"))
    if not rows:
        return {}
    active = [k for k, row in enumerate(rows) if row[2] == "active"]
    out = {}
    for k, row in enumerate(rows):
        if row[0] not in wanted or not active:
            continue
        scores = matrix[active] @ matrix[k]
        for rank in np.argsort(-scores):
            other = rows[active[rank]]
            if other[0] == row[0] or frozenset((row[0], other[0])) in dismissed:
                continue
            if scores[rank] >= threshold:
                out[row[0]] = (other[0], other[1], float(scores[rank]))
            break
    return out
