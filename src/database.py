import json
import math
import re
import sqlite3
from datetime import datetime
from config import DB_PATH, RECALL_MIN_DF


# ── Schema migrations ──
# Each entry is (version, function). init_db() runs any migration whose
# version is greater than what is recorded in schema_version, in order,
# once. Add new schema changes by appending a new (version, function) pair
# here rather than hand running ALTER TABLE on the Pi.

def _migration_001_memories_v2(conn):
    """Replace the v1 memories table (content/source/created_at/last_referenced/
    relevance_score only) with the richer schema. The table is empty in
    production as of this migration, so drop and recreate instead of
    migrating data."""
    conn.execute("DROP TABLE IF EXISTS memories")
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT,
            source TEXT NOT NULL DEFAULT 'explicit',
            created_at TEXT NOT NULL,
            references_date TEXT,
            volatile INTEGER NOT NULL DEFAULT 0,
            confidence TEXT NOT NULL DEFAULT 'high',
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by INTEGER,
            last_referenced TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0
        )
    """)


def _migration_002_verification_log_v2(conn):
    """Add embedded_duration_seconds, wake_confidence, and turn_type to
    verification_log. The table has zero rows in production as of this
    migration (Phase 1 logging just went live and hasn't logged an
    attempt yet), so drop and recreate instead of migrating data."""
    conn.execute("DROP TABLE IF EXISTS verification_log")
    conn.execute("""
        CREATE TABLE verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            similarity REAL NOT NULL,
            accepted INTEGER NOT NULL,
            threshold_used REAL NOT NULL,
            transcript TEXT,
            duration_seconds REAL NOT NULL,
            embedded_duration_seconds REAL NOT NULL,
            wake_confidence REAL,
            turn_type TEXT NOT NULL
        )
    """)


def _migration_003_verification_outcome(conn):
    """Add an outcome column so attempts that never produced a real embedding
    are separable from genuine accept/reject decisions.

    A clip that trims to no voiced audio still yields a similarity score, and
    that score is meaningless: one such row scored 0.426 and was the only
    rejection in the first ten attempts, which made the sample look like a
    false rejection problem when it was not.

    ALTER rather than recreate, since this table now holds real collected
    data. Existing rows predate the guard and were all genuinely scored."""
    conn.execute(
        "ALTER TABLE verification_log ADD COLUMN outcome TEXT NOT NULL DEFAULT 'scored'"
    )


def _migration_004_acoustic_measures(conn):
    """Log the acoustic correlates of each attempt so distance, loudness, and
    vocal effort can be separated by regression instead of by block labels,
    which depend on reproducing "projecting" consistently by feel.

    snr_db is the one that matters most: simulation puts embedding cosine at
    0.98 for 25dB SNR and 0.68 for 10dB, which is the whole observed score
    range. spectral_tilt is the production side marker, since a close normal
    voice and a far projected voice can arrive at the same level but not with
    the same tilt."""
    for column in ("rms_dbfs REAL", "snr_db REAL", "spectral_tilt REAL"):
        conn.execute(f"ALTER TABLE verification_log ADD COLUMN {column}")


def _migration_005_timing_log(conn):
    """Per turn stage timings.

    The existing "(Total: 5.14s)" print starts after recording ends, so it
    cannot see the endpointing delay, which the user experiences as part of
    the wait. Everything here is measured from the moment the user stopped
    talking instead.

    Every stage column is nullable: an action turn has an action_ms and a
    plain turn does not, a turn that never reaches TTS has no perceived
    latency, and a stage that errors leaves its column empty rather than
    dropping the whole row."""
    conn.execute("""
        CREATE TABLE timing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            turn_type TEXT NOT NULL,
            action_fired INTEGER NOT NULL DEFAULT 0,
            transcript TEXT,
            speech_end_to_endpoint_ms REAL,
            transcribe_ms REAL,
            verify_ms REAL,
            claude_ttft_ms REAL,
            claude_total_ms REAL,
            tts_ttfb_ms REAL,
            tts_first_audio_ms REAL,
            action_ms REAL,
            total_perceived_ms REAL
        )
    """)


def _migration_006_timing_model(conn):
    """Record which model served each turn, and what it said.

    model supports the TTFT comparison. response is here so answer quality can
    be judged by reading: conversation_history has the text but carries no
    model attribution, so without this column there is no way to tell which
    arm produced a given answer."""
    for column in ("model TEXT", "response TEXT"):
        conn.execute(f"ALTER TABLE timing_log ADD COLUMN {column}")


def _migration_007_response_length_and_cache(conn):
    """Response word count, cache effectiveness, and the sentence assembly gap.

    response_words makes the prompt length change measurable instead of a
    matter of feel. Baseline before the change: median 76 words, p90 117,
    max 120, which is roughly thirty seconds of speech per turn.

    cache_read_tokens exists because prompt caching fails silently. Below the
    model's minimum cacheable prefix the API caches nothing, reports nothing,
    and raises nothing, so a zero here is the only signal.

    first_sentence_ms closes the instrumentation gap that showed up as 754ms
    of unaccounted time: the wait between Claude's first token and the first
    complete sentence reaching the TTS queue."""
    for column in ("response_words INTEGER", "cache_read_tokens INTEGER",
                   "cache_creation_tokens INTEGER", "first_sentence_ms REAL"):
        conn.execute(f"ALTER TABLE timing_log ADD COLUMN {column}")


def _migration_008_max_pause(conn):
    """Longest pause the speaker took mid utterance and then spoke through.

    SILENCE_LIMIT has to clear this value or turns end on hesitation rather
    than on a finished thought. The 0.63s figure it was set against came from
    reading a scripted enrollment phrase, and spontaneous speech pauses longer
    than read speech, so the constant needs tuning from real conversation."""
    conn.execute("ALTER TABLE timing_log ADD COLUMN max_pause_ms REAL")


def _migration_009_recording_path(conn):
    """Where the archived copy of this recording lives.

    command.wav is overwritten every turn, so audio decisions have had to be
    validated against enrollment recordings rather than real commands. This
    links each logged attempt, with its SNR and transcript, back to the audio
    that produced it."""
    conn.execute("ALTER TABLE verification_log ADD COLUMN recording_path TEXT")


def _migration_010_tool_use(conn):
    """Tool call timing on timing_log, plus a log of the calls themselves.

    tool_ms is the tool's own execution: the HTTP round trips for weather, the
    thread spawn for a timer. second_ttft_ms is time to first token on the
    follow up call that speaks about the result, which only happens for tools
    with returns_to_model set. Separating them answers the question the old
    single action_ms could not: when an action turn feels slow, was it the tool
    or was it Claude reading the result back.

    tool_call_log is a sibling of verification_log and timing_log, one row per
    call. Tool results are neither conversation nor memory: putting them in
    conversation_history would replay last Tuesday's weather into the prompt as
    context, and putting them in memories would mix facts about the world at one
    instant with facts about Lethanial. They are here for the same reason the
    other two logs exist, to make a subsystem tunable from data rather than
    from feel.

    arguments and result are stored as text. result is truncated on write; the
    full payload is not worth keeping and a calendar entry or a workout log is
    personal data that should not accumulate without a reason."""
    for column in ("tool_ms REAL", "second_ttft_ms REAL"):
        conn.execute(f"ALTER TABLE timing_log ADD COLUMN {column}")

    conn.execute("""
        CREATE TABLE tool_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments TEXT,
            result TEXT,
            is_error INTEGER NOT NULL DEFAULT 0,
            duration_ms REAL,
            model TEXT
        )
    """)


def _migration_011_alert_log(conn):
    """Timer and reminder alerts, and how they got delivered.

    Alerts were the one thing in the system with no record at all. There was no
    way to tell how often one collided with speech, or how late a deferred one
    would be, which meant the delay threshold could only ever be tuned by feel.

    delay_ms is the gap between the alert firing and Lethanial hearing it.
    mode is 'folded' when Nova worked it into a response she was already
    giving, 'spoken' when it was announced on its own."""
    conn.execute("""
        CREATE TABLE alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT,
            fired_at TEXT,
            delivered_at TEXT NOT NULL,
            delay_ms REAL,
            mode TEXT NOT NULL
        )
    """)


def _migration_012_pronunciations(conn):
    """How to say words the synthesizer gets wrong.

    ElevenLabs reads "Lethanial" as spelled, which is not how it sounds. The
    alias is a respelling fed to the synthesizer in place of the real word.

    ipa and arpabet are stored but unused: ElevenLabs only supports phoneme tags
    on the turbo and flash v2 English models, and this runs on flash v2.5.
    Recording them now means the data is already there if a future model can
    take them, and it documents the intended pronunciation for a human reading
    the table.

    verified marks aliases actually listened to. An unverified alias is a guess,
    and a wrong guess sounds worse than the original spelling."""
    conn.execute("""
        CREATE TABLE pronunciations (
            id INTEGER PRIMARY KEY,
            grapheme TEXT NOT NULL UNIQUE,
            alias TEXT NOT NULL,
            ipa TEXT,
            arpabet TEXT,
            verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    now = datetime.now().isoformat()
    conn.execute(
        """INSERT INTO pronunciations
           (grapheme, alias, ipa, arpabet, verified, created_at, updated_at)
           VALUES (?, ?, ?, ?, 1, ?, ?)""",
        ("Lethanial", "Luhthanyul", "l\u0259\u02c8\u03b8\u00e6nj\u0259l",
         "L AH0 TH AE1 N Y AH0 L", now, now)
    )


def _migration_013_supersede_and_expiry(conn):
    """Make correction and shelf life real.

    Three columns describing this already existed and none of them did anything.
    superseded_by was declared and never written, volatile and references_date
    were written and never read. So the store was append only in practice: a
    memory recorded wrong stayed wrong, and the only remedy was deleting the row
    and losing the fact that it ever changed.

    No new columns are needed for supersede. status is free text and every
    retrieval already filters on 'active', so retiring a row to 'superseded' or
    'expired' removes it from the prompt with no query changes.

    superseded_at is added for the audit trail: superseded_by says what replaced
    a fact, and this says when, which is what makes a chain readable later."""
    conn.execute("ALTER TABLE memories ADD COLUMN superseded_at TEXT")


def _migration_014_memory_tiers_and_search(conn):
    """Split the corpus into what is always in the prompt and what is looked up.

    The corpus outgrew the prompt. Not on cost or latency, both of which were
    measured and are fine well past a thousand memories, but on two things that
    degrade much earlier: the remember tool decides store versus supersede by
    reading the whole list, and that judgment falls apart long before the
    context window does; and a model holding two hundred facts volunteers the
    irrelevant ones.

    tier 1 is resident, tier 2 is retrievable. Everything defaults to 1, so this
    migration changes nothing on its own. A memory only leaves the prompt when
    something deliberately demotes it, because a migration that silently removed
    facts from Nova's context would be a miserable thing to debug.

    The index is FTS5 in external content mode: memories_fts holds the inverted
    index and reaches back into memories by rowid rather than storing a second
    copy of every fact. That costs more careful triggers and buys exactly one
    place where the text lives, which is worth it for a store whose contents are
    personal enough to have been the subject of a history rewrite.

    Triggers cover all three mutations. The delete form is FTS5's, where you
    insert a 'delete' command carrying the old values rather than issuing a
    DELETE, because an external content table cannot read a row that is already
    gone. Getting this wrong leaves the index quietly stale, so if it is ever
    suspect, INSERT INTO memories_fts(memories_fts) VALUES('rebuild') is the
    recovery.

    Nothing filters on status here. Search joins back to memories and filters
    there, which keeps one definition of what active means."""
    conn.execute("ALTER TABLE memories ADD COLUMN tier INTEGER NOT NULL DEFAULT 1")

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            content='memories',
            content_rowid='id'
        )""")

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_insert
        AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END""")
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_delete
        AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END""")
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_fts_update
        AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END""")

    # Backfill. The triggers only see mutations from here on, so every row that
    # already exists has to be indexed once by hand.
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")


def _migration_015_retrieval_log(conn):
    """Log every retrieval so the decision about embeddings can be made on data.

    A miss cannot be logged directly, because it is invisible: an empty result
    means either the question was not about him or the words did not line up,
    and nothing in the query distinguishes those. So every retrieval is
    recorded and judged afterwards by hand.

    terms is stored separately from query because both retrieval bugs so far
    were visible in the terms before they were visible in the results. Seeing
    ['did', 'build', 'accelerometer'] explains the wrong answer immediately.

    verdict stays null until a human fills it in. Once enough rows are labelled
    this stops being a log and becomes an eval set built from real questions,
    which is what turns 'add a vector store' from a guess into a measurement."""
    conn.execute("""
        CREATE TABLE retrieval_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            query TEXT NOT NULL,
            terms TEXT NOT NULL,
            returned_ids TEXT NOT NULL,
            scores TEXT,
            hit_count INTEGER NOT NULL,
            verdict TEXT,
            note TEXT
        )
    """)


MIGRATIONS = [
    (1, _migration_001_memories_v2),
    (2, _migration_002_verification_log_v2),
    (3, _migration_003_verification_outcome),
    (4, _migration_004_acoustic_measures),
    (5, _migration_005_timing_log),
    (6, _migration_006_timing_model),
    (7, _migration_007_response_length_and_cache),
    (8, _migration_008_max_pause),
    (9, _migration_009_recording_path),
    (10, _migration_010_tool_use),
    (11, _migration_011_alert_log),
    (12, _migration_012_pronunciations),
    (13, _migration_013_supersede_and_expiry),
    (14, _migration_014_memory_tiers_and_search),
    (15, _migration_015_retrieval_log),
]

# Tool results are capped rather than kept whole. Weather from three weeks ago
# has no value, and once Calendar lands these rows carry real event contents.
TOOL_RESULT_MAX_CHARS = 500


def _run_migrations(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        current = 0
    else:
        current = row[0]

    for version, migration_fn in MIGRATIONS:
        if version > current:
            migration_fn(conn)
            conn.execute("UPDATE schema_version SET version = ?", (version,))
            conn.commit()
            print(f"Applied migration {version}: {migration_fn.__name__}", flush=True)
            current = version


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_device TEXT NOT NULL DEFAULT 'pi'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            due_at TEXT,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()

    _run_migrations(conn)

    conn.close()


# ── Memories ──

def save_memory(content, source="explicit", status="active", category=None,
                 confidence="high", volatile=False, references_date=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    existing = c.execute(
        "SELECT id FROM memories WHERE content = ? AND status = 'active'",
        (content,)
    ).fetchone()
    if existing:
        print(f"Duplicate memory skipped (matches active id {existing[0]}): {content}", flush=True)
        conn.close()
        return False

    c.execute(
        """INSERT INTO memories
           (content, category, source, created_at, references_date, volatile, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, category, source, datetime.now().isoformat(), references_date,
         int(volatile), confidence, status)
    )
    conn.commit()
    conn.close()
    print(f"Memory saved ({source}, {status}): {content}", flush=True)
    return True


def get_seed_memories(tier=1):
    """Resident seed memories, grouped for prompt assembly by category then id.

    tier 1 only by default. Tier 2 exists but is reached through
    search_memories rather than by sitting in every prompt. Pass tier=None for
    the whole corpus, which is what the seeder and the CLI want."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if tier is None:
        rows = c.execute(
            "SELECT id, content, category FROM memories "
            "WHERE source = 'seed' AND status = 'active' "
            "ORDER BY category, id"
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT id, content, category FROM memories "
            "WHERE source = 'seed' AND status = 'active' AND tier = ? "
            "ORDER BY category, id", (tier,)
        ).fetchall()
    conn.close()
    return rows


# FTS5 treats a bare multi word MATCH as an implicit AND, so "sister anime"
# asks for one memory containing both words and finds nothing, even though a
# memory about each exists. Queries here are spoken questions rather than
# search syntax, so terms are ORed and ranking decides what matters. Anything
# that would be read as an operator is dropped, since a stray quote or NEAR in
# a transcript is a syntax error rather than a search.
# The stopword list has to be thorough rather than approximate. A first pass
# missed "did", and the query "what did I build with the accelerometer" then
# ranked a memory about VS Code first, because that memory happened to contain
# both "did" and "build" while the MotionSense memory only carried the one rare
# word. bm25 was right; the query was junk. With 209 short documents the IDF
# spread is too compressed for a rare term to outweigh two function word hits,
# so anything that carries no meaning has to be dropped before it reaches MATCH.
_FTS_SAFE = re.compile(r"[A-Za-z0-9']+")
_FTS_STOPWORDS = frozenset("""
a about after all also am an and any are as at back be because been before
being between both but by can come could did do does doing done down during
each even every few first for from get give go had has have having he her here
hers him his how i if in into is it its just know like make many me might more
most much must my never no nor not now of off on once one only or other our
out over own said same see she should so some such take tell than that the
their them then there these they thing think this those through to too under
until up us use used very was way we well were what when where which while who
whom why will with would you your yours
date day time today tomorrow tonight morning evening week month year
""".split())


def _log_retrieval(query, terms, rows, scores):
    """Record one retrieval. Never raises: a logging failure must not cost a turn."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO retrieval_log (created_at, query, terms, returned_ids, "
            "scores, hit_count) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), query, json.dumps(terms),
             json.dumps([r[0] for r in rows]),
             json.dumps([round(s, 3) for s in scores]), len(rows)))
        conn.commit()
        conn.close()
    except Exception:
        pass


def search_memories(query, limit=5, tier=None, log=False):
    """Rank memories against a spoken query. Empty list when nothing matches.

    Returns (id, content, category) so results drop straight into the same
    prompt blocks the resident memories use."""
    terms = [t for t in _FTS_SAFE.findall(query.lower())
             if t not in _FTS_STOPWORDS and len(t) > 1]
    if not terms:
        if log:
            _log_retrieval(query, [], [], [])
        return []

    conn = sqlite3.connect(DB_PATH)
    sql = ("SELECT m.id, m.content, m.category "
           "FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
           "WHERE memories_fts MATCH ? AND m.status = 'active' ")
    params = [" OR ".join(terms)]
    if tier is not None:
        sql += "AND m.tier = ? "
        params.append(tier)

    try:
        rows = conn.execute(sql, params).fetchall()
        # Rank on rarity alone, deliberately not on bm25.
        #
        # bm25 normalises by document length, and for "what did I build with
        # the accelerometer" that ranked a 251 character memory about VS Code
        # above the 380 character one about MotionSense, because the long
        # document was penalised for its length while the short one happened to
        # contain "build". Length normalisation is there to stop a long
        # document winning by sheer surface area, which is a real problem for
        # web pages and no problem at all for a corpus of one sentence facts.
        # Here it only punishes the memories that carry the most detail, which
        # are exactly the ones worth retrieving.
        #
        # So: score a row by how rare the query terms it contains are, and
        # ignore how long it is. One document frequency lookup per term, over a
        # corpus in the hundreds.
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0] or 1
        weights = {}
        for term in terms:
            df = conn.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?",
                (term,)).fetchone()[0]
            weights[term] = math.log(total / df) if df else 0.0

        def score(row):
            words = set(_FTS_SAFE.findall(row[1].lower()))
            return sum(w for t, w in weights.items() if t in words)

        # A floor, because any match at all is not evidence of relevance.
        # "What time is it" survived the stopword list on the single word
        # "time", matched two memories that happened to contain "for a time",
        # and attached his grandfather and Alejandra to a question about the
        # clock. That is precisely the irrelevant volunteering the tier split
        # existed to stop.
        #
        # The floor is expressed in document frequency rather than as a raw
        # number, so it keeps meaning as the corpus grows: a row has to match
        # terms at least as informative as one appearing in RECALL_MIN_DF
        # documents. Note this cannot be tuned to perfection, since idf does
        # not separate a useful term from a useless one here. "azarieyah"
        # scores lower than "time" because she appears in eleven memories and
        # it appears in eight. Names are good search terms and common nouns are
        # not, and nothing in the statistics knows that. This is the ceiling of
        # keyword retrieval, and the reason to log misses rather than trust it.
        floor = math.log(total / RECALL_MIN_DF) if total > RECALL_MIN_DF else 0.0
        ranked = sorted(((score(r), r) for r in rows), key=lambda p: p[0],
                        reverse=True)
        kept = [(s, r) for s, r in ranked if s >= floor][:limit]
        rows = [r for _, r in kept]
        if log:
            _log_retrieval(query, terms, rows, [s for s, _ in kept])
    except sqlite3.OperationalError:
        # A query that still parses badly should cost Nova nothing. Losing the
        # retrieval is survivable; raising inside prompt assembly is not.
        rows = []
        if log:
            _log_retrieval(query, terms, [], [])
    conn.close()
    return rows


def memory_manifest(tier=2):
    """Category counts for the non resident tier, for the prompt's index.

    Nova cannot search for something she does not know exists, so the prompt
    carries the shape of what is retrievable even when the facts themselves are
    not present. Roughly a hundred tokens, and without it search either never
    fires or fires on everything."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT category, COUNT(*) FROM memories "
        "WHERE status = 'active' AND tier = ? "
        "GROUP BY category ORDER BY COUNT(*) DESC", (tier,)
    ).fetchall()
    conn.close()
    return rows


def get_episodic_memories(limit=20):
    """Most recent active explicit or approved-implicit memories. Ordered by
    id, not created_at: seed rows share a timestamp and ties on created_at
    have no defined order in SQLite.

    Includes source='implicit' as well as 'explicit': an implicit row only
    reaches status='active' by being approved through the pending queue, so
    including it here is already correct without rewriting its source,
    which stays as provenance for tracking approved vs rejected model
    generated memories."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content FROM memories "
        "WHERE source IN ('explicit', 'implicit') AND status = 'active' "
        # Expiry is enforced here rather than by a sweep. A volatile memory
        # whose date has passed stops being retrieved the moment it passes,
        # with no job to schedule and nothing to fall out of sync. expire_memories
        # only marks what this already hides, for the sake of the listing.
        "  AND NOT (volatile = 1 AND references_date IS NOT NULL "
        "           AND references_date < ?) "
        "ORDER BY id DESC LIMIT ?",
        (datetime.now().isoformat(), limit)
    ).fetchall()
    conn.close()
    return rows


def get_pending_memories(limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content, category, source, confidence FROM memories "
        "WHERE status = 'pending' ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_active_memories(limit=50, offset=0):
    """All active memories (seed and explicit/approved-implicit) with full
    metadata, paginated, for the GET /memories listing endpoint. The seed
    corpus alone can run over a hundred rows, so this always returns the
    total active count alongside the page so a caller can tell when there
    is more."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
    rows = c.execute(
        "SELECT id, content, category, source, confidence, status FROM memories "
        "WHERE status = 'active' ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return rows, total


def supersede_memory(old_id: int, new_content: str, **fields):
    """Replace a memory with a corrected one, keeping the link between them.

    Deleting and re-adding loses the fact that something changed. An exam that
    moved from Friday to Thursday is different information from an exam that was
    always Thursday, and only one of those survives a delete.

    The old row goes to status 'superseded', which every retrieval already
    filters out, so the prompt sees only the current answer. superseded_by points
    at the replacement so the chain can be walked later.

    Returns the new row id, or None when old_id does not exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    old = c.execute(
        "SELECT source, category, confidence, volatile, references_date "
        "FROM memories WHERE id = ?", (old_id,)
    ).fetchone()
    if old is None:
        conn.close()
        return None

    # Inherit the old row's classification unless the caller overrides it. A
    # correction is usually the same kind of fact, and re-specifying every field
    # to fix a typo is how fields drift.
    source, category, confidence, volatile, references_date = old
    now = datetime.now().isoformat()
    c.execute(
        """INSERT INTO memories
           (content, category, source, created_at, references_date, volatile,
            confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
        (new_content,
         fields.get("category", category),
         fields.get("source", source),
         now,
         fields.get("references_date", references_date),
         int(fields.get("volatile", volatile)),
         fields.get("confidence", confidence))
    )
    new_id = c.lastrowid
    c.execute(
        "UPDATE memories SET status = 'superseded', superseded_by = ?, "
        "superseded_at = ? WHERE id = ?",
        (new_id, now, old_id)
    )
    conn.commit()
    conn.close()
    print(f"Memory {old_id} superseded by {new_id}: {new_content}", flush=True)
    return new_id


def get_memory_chain(memory_id: int):
    """Walk a memory back through everything it replaced, newest first.

    This is the payoff for not deleting. It answers "what did this used to say"
    and "when did it change", which a delete cannot."""
    conn = sqlite3.connect(DB_PATH)
    chain = []
    seen = set()
    current = memory_id
    while current is not None and current not in seen:
        seen.add(current)
        row = conn.execute(
            "SELECT id, content, status, created_at, superseded_by, superseded_at "
            "FROM memories WHERE id = ?", (current,)
        ).fetchone()
        if row is None:
            break
        chain.append(row)
        # Walk backwards: find whatever this row replaced.
        prev = conn.execute(
            "SELECT id FROM memories WHERE superseded_by = ?", (current,)
        ).fetchone()
        current = prev[0] if prev else None
    conn.close()
    return chain


def expire_memories():
    """Mark volatile memories whose date has passed.

    Bookkeeping only. get_episodic_memories already excludes these at read time,
    so this changes nothing about what Nova sees; it exists so an expired memory
    shows as expired in a listing rather than looking active and mysteriously
    absent from her answers.

    Returns how many were marked."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE memories SET status = 'expired' "
        "WHERE status = 'active' AND volatile = 1 "
        "  AND references_date IS NOT NULL AND references_date < ?",
        (datetime.now().isoformat(),)
    )
    marked = c.rowcount
    conn.commit()
    conn.close()
    if marked:
        print(f"Expired {marked} volatile memories past their date", flush=True)
    return marked


def approve_memory(memory_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE memories SET status = 'active' WHERE id = ? AND status = 'pending'",
        (memory_id,)
    )
    updated = c.rowcount
    conn.commit()
    conn.close()
    return updated > 0


def delete_memory(memory_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


# ── Conversation history ──

def save_message(role, content, device="pi"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_history (role, content, created_at, source_device) VALUES (?, ?, ?, ?)",
        (role, content, datetime.now().isoformat(), device)
    )
    conn.commit()
    conn.close()


def get_recent_messages(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def get_history(limit: int = 50, offset: int = 0) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, created_at, source_device FROM conversation_history"
        " ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    return [
        {"role": r, "content": c, "created_at": t, "source_device": d}
        for r, c, t, d in rows
    ]


# ── Timing log ──

def log_timing(turn_type, action_fired, transcript, speech_end_to_endpoint_ms,
               transcribe_ms, verify_ms, claude_ttft_ms, claude_total_ms,
               tts_ttfb_ms, tts_first_audio_ms, action_ms, total_perceived_ms,
               model=None, response=None, cache_read_tokens=None,
               cache_creation_tokens=None, first_sentence_ms=None,
               max_pause_ms=None, tool_ms=None, second_ttft_ms=None):
    # Derived here rather than at every call site so the count and the text it
    # describes can never drift apart.
    response_words = len(response.split()) if response else None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO timing_log
           (created_at, turn_type, action_fired, transcript,
            speech_end_to_endpoint_ms, transcribe_ms, verify_ms,
            claude_ttft_ms, claude_total_ms, tts_ttfb_ms, tts_first_audio_ms,
            action_ms, total_perceived_ms, model, response, response_words,
            cache_read_tokens, cache_creation_tokens, first_sentence_ms, max_pause_ms,
            tool_ms, second_ttft_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), turn_type, int(action_fired), transcript,
         speech_end_to_endpoint_ms, transcribe_ms, verify_ms,
         claude_ttft_ms, claude_total_ms, tts_ttfb_ms, tts_first_audio_ms,
         action_ms, total_perceived_ms, model, response, response_words,
         cache_read_tokens, cache_creation_tokens, first_sentence_ms,
         max_pause_ms, tool_ms, second_ttft_ms)
    )
    conn.commit()
    conn.close()


def log_tool_call(tool_name, arguments, result, is_error=False,
                  duration_ms=None, model=None):
    """One row per tool call. Truncates the result on write.

    Failures are logged too, with is_error set. A tool that errored is the most
    interesting row in the table: it is the one that explains why Nova said
    something strange."""
    text = "" if result is None else str(result)
    if len(text) > TOOL_RESULT_MAX_CHARS:
        text = text[:TOOL_RESULT_MAX_CHARS] + "..."

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO tool_call_log
           (created_at, tool_name, arguments, result, is_error, duration_ms, model)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), tool_name, json.dumps(arguments or {}),
         text, int(is_error), duration_ms, model)
    )
    conn.commit()
    conn.close()


def log_alert(kind, content, fired_at, mode, delay_ms=None):
    """One row per delivered alert. Written at delivery, not at firing, so a
    queued alert that is still waiting never appears as though it landed."""
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO alert_log
           (created_at, kind, content, fired_at, delivered_at, delay_ms, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (now, kind, content, fired_at, now, delay_ms, mode)
    )
    conn.commit()
    conn.close()


def get_pronunciations():
    """Every grapheme and alias, longest grapheme first.

    The ordering is the substitution rule, not a display preference. Applying
    "Lethanial Leveille" before "Lethanial" is what lets a multi word entry win
    over a single word one that is a prefix of it. Sorting here rather than at
    each call site means the rule cannot be forgotten by a new caller."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT grapheme, alias, arpabet FROM pronunciations "
        "ORDER BY LENGTH(grapheme) DESC"
    ).fetchall()
    conn.close()
    return rows


def upsert_pronunciation(grapheme, alias, ipa=None, arpabet=None, verified=False):
    """Add or replace one entry at runtime, no migration needed.

    grapheme is UNIQUE, so this is an update when it already exists. created_at
    is preserved on update and only updated_at moves, so the table records when
    a pronunciation was first needed as well as when it was last corrected."""
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO pronunciations
           (grapheme, alias, ipa, arpabet, verified, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(grapheme) DO UPDATE SET
             alias = excluded.alias,
             ipa = excluded.ipa,
             arpabet = excluded.arpabet,
             verified = excluded.verified,
             updated_at = excluded.updated_at""",
        (grapheme, alias, ipa, arpabet, int(verified), now, now)
    )
    conn.commit()
    conn.close()


def prune_tool_call_log(max_rows=2000):
    """Cap the table by row count, oldest first, the way ARCHIVE_MAX_FILES caps
    recordings. Called opportunistically rather than on a schedule."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM tool_call_log WHERE id NOT IN "
        "(SELECT id FROM tool_call_log ORDER BY id DESC LIMIT ?)",
        (max_rows,)
    )
    conn.commit()
    conn.close()


# ── Verification log ──

def log_verification(similarity, accepted, threshold_used, transcript, duration_seconds,
                      embedded_duration_seconds, turn_type, wake_confidence=None,
                      outcome='scored', rms_dbfs=None, snr_db=None, spectral_tilt=None,
                      recording_path=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO verification_log
           (created_at, similarity, accepted, threshold_used, transcript, duration_seconds,
            embedded_duration_seconds, wake_confidence, turn_type, outcome,
            rms_dbfs, snr_db, spectral_tilt, recording_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), similarity, int(accepted), threshold_used,
         transcript, duration_seconds, embedded_duration_seconds, wake_confidence,
         turn_type, outcome, rms_dbfs, snr_db, spectral_tilt, recording_path)
    )
    conn.commit()
    conn.close()
