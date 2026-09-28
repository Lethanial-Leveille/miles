import os
import sys

# Test files live in src/tests/; the modules under test live in src/, one
# directory up. Insert it so plain `import database` etc. resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point database.py at a throwaway sqlite file for the duration of one
    test and run migrations against it, so tests never touch the real
    ~/miles/data/miles.db."""
    test_db_path = str(tmp_path / "test_miles.db")
    monkeypatch.setattr(database, "DB_PATH", test_db_path)
    database.init_db()
    return database


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Words hashed into 64 buckets stand in for the sentence model: sentences
    sharing words point the same way, and nothing loads 80MB of weights."""
    import zlib
    import numpy as np
    import embeddings

    def encode(texts):
        out = []
        for text in texts:
            v = np.zeros(64, dtype=np.float32)
            for word in text.lower().replace(".", "").split():
                v[zlib.crc32(word.encode()) % 64] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out)

    monkeypatch.setattr(embeddings, "encode", encode)
    return encode
