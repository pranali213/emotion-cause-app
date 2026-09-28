import sqlite3, os

DATABASE = os.path.join(os.path.dirname(__file__), "ecpe.db")

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    db = get_db()
    c  = db.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS utterances (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        turn_index      INTEGER NOT NULL,
        speaker         TEXT NOT NULL,
        text            TEXT NOT NULL,
        emotion         TEXT DEFAULT 'neutral',
        confidence      REAL DEFAULT 0.0,
        model_used      TEXT DEFAULT 'Random Forest',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS emotion_cause_pairs (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id      INTEGER NOT NULL,
        emotion_utterance_id INTEGER NOT NULL,
        cause_utterance_id   INTEGER NOT NULL,
        emotion              TEXT NOT NULL,
        cause_span           TEXT,
        confidence           REAL DEFAULT 0.0,
        explanation          TEXT,
        FOREIGN KEY (conversation_id)      REFERENCES conversations(id) ON DELETE CASCADE,
        FOREIGN KEY (emotion_utterance_id) REFERENCES utterances(id)    ON DELETE CASCADE,
        FOREIGN KEY (cause_utterance_id)   REFERENCES utterances(id)    ON DELETE CASCADE
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS analysis_runs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        model_used      TEXT NOT NULL,
        pairs_found     INTEGER DEFAULT 0,
        accuracy        REAL DEFAULT 0.0,
        f1_score        REAL DEFAULT 0.0,
        ran_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS model_metrics (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name  TEXT NOT NULL,
        accuracy    REAL,
        precision   REAL,
        recall      REAL,
        f1          REAL,
        cv_f1_mean  REAL,
        cv_f1_std   REAL,
        confusion_matrix TEXT,
        labels      TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── NEW: training examples (custom dataset) ───────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS training_examples (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT NOT NULL,
        emotion    TEXT NOT NULL,
        source     TEXT DEFAULT 'manual',   -- 'manual' | 'csv' | 'json' | 'builtin'
        dataset_id INTEGER,                 -- FK to datasets (nullable for manual)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── NEW: uploaded dataset registry ───────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS datasets (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        file_type    TEXT NOT NULL,   -- 'csv' | 'json'
        row_count    INTEGER DEFAULT 0,
        uploaded_by  TEXT DEFAULT 'admin',
        uploaded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── NEW: model version snapshots (before/after retrain comparison) ────────
    c.execute("""CREATE TABLE IF NOT EXISTS model_versions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        version_tag   TEXT NOT NULL,          -- e.g. 'v1', 'v2-retrained'
        model_name    TEXT NOT NULL,
        accuracy      REAL,
        precision     REAL,
        recall        REAL,
        f1            REAL,
        cv_f1_mean    REAL,
        cv_f1_std     REAL,
        training_size INTEGER DEFAULT 0,
        notes         TEXT,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    db.commit()
    db.close()
