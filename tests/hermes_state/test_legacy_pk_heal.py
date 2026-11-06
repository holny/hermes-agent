"""Unconditional sessions/messages PK heals (#148742).

Legacy builds created ``sessions`` and ``messages`` with a bare ``id`` column (no
PRIMARY KEY). On messages, ``append_message`` omits ``id`` — every insert stores
NULL, ids and rowids drift apart, and the compaction split's ``int(row["id"])``
crashes while indexing walks skip the rows. On sessions, ``create_session``'s
``ON CONFLICT(id)`` upsert fails outright ("does not match any PRIMARY KEY or
UNIQUE constraint"). SQLite cannot ALTER a primary key, so both tables are
rebuilt once, unconditionally, on open — same pattern as the gateway_routing and
session_model_usage PK heals.
"""

import sqlite3

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION

# A legacy messages table: every SCHEMA_SQL column the fixture writes, but ``id``
# is a bare INTEGER column — inserts that omit it (append_message's shape) store
# NULL instead of taking the next autoincrement value.
LEGACY_MESSAGES_SQL = """
    CREATE TABLE messages (
        id INTEGER,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        role TEXT NOT NULL,
        content TEXT,
        tool_call_id TEXT,
        tool_calls TEXT,
        tool_name TEXT,
        effect_disposition TEXT,
        timestamp REAL NOT NULL,
        token_count INTEGER,
        finish_reason TEXT,
        reasoning TEXT,
        reasoning_content TEXT,
        reasoning_details TEXT,
        codex_reasoning_items TEXT,
        codex_message_items TEXT,
        platform_message_id TEXT,
        observed INTEGER DEFAULT 0,
        _compressed_summary INTEGER NOT NULL DEFAULT 0,
        active INTEGER NOT NULL DEFAULT 1,
        compacted INTEGER NOT NULL DEFAULT 0,
        api_content TEXT,
        display_kind TEXT,
        display_metadata TEXT,
        display_identity BLOB,
        display_order INTEGER,
        message_uid TEXT,
        absorbed_message_uids TEXT,
        tool_call_uids TEXT,
        tool_call_uid TEXT
    )
"""

# A legacy sessions table: the full current column set but ``id`` is bare (no PRIMARY
# KEY, no UNIQUE) — nothing enforces session-id uniqueness and ON CONFLICT(id) cannot
# match. Derived from the heal's own DDL so the fixture stays in lockstep with the
# shipped column list.
def _legacy_sessions_sql():
    from hermes_state_schema import SessionSchemaMixin

    lines = []
    for line in SessionSchemaMixin._SESSIONS_HEAL_DDL.splitlines():
        stripped = line.strip()
        if stripped.startswith("FOREIGN KEY"):
            continue
        if stripped.startswith("id TEXT PRIMARY KEY"):
            line = line.replace("id TEXT PRIMARY KEY", "id TEXT", 1)
        lines.append(line)
    ddl = "\n".join(lines)
    # Rename onto the live table name and drop the comma that sat before the FOREIGN KEY
    # lines we removed.
    return ddl.replace("sessions_pk_heal_new", "sessions", 1).replace(",\n)", "\n)")


def _pk_cols(db, table):
    rows = db._conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]


def _make_born_current_db(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", "cli")
    db.close()
    return db_path


class TestMessagesPkHeal:
    def test_bare_id_messages_rebuilt_with_integer_pk_and_collision_free_remapping(self, tmp_path):
        """NULL/text ids are remapped above every preserved integer id; preserved integer
        ids survive verbatim (messages_fts is external content keyed by id — renumbering
        healthy rows would desync the index); ordering is kept; every row survives."""
        db_path = _make_born_current_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE messages")
        conn.execute(LEGACY_MESSAGES_SQL)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            [
                (5, "s1", "user", "kept integer id", 1.0),
                (None, "s1", "user", "null id row 1", 2.0),
                (None, "s1", "user", "null id row 2", 3.0),
                ("legacy-text", "s1", "user", "text id row", 4.0),
            ],
        )
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            assert _pk_cols(db, "messages") == ["id"]
            rows = db._conn.execute(
                "SELECT id, content FROM messages ORDER BY timestamp"
            ).fetchall()
            assert [r[1] for r in rows] == [
                "kept integer id", "null id row 1", "null id row 2", "text id row",
            ]
            ids = [r[0] for r in rows]
            # The preserved integer id survives verbatim.
            assert rows[0][0] == 5
            # Remapped ids are integers, unique, and strictly above the preserved max.
            assert all(isinstance(i, int) for i in ids[1:])
            assert len(set(ids)) == 4
            assert all(i > 5 for i in ids[1:])
        finally:
            db.close()

    def test_healed_db_opens_again_as_noop(self, tmp_path):
        """The heal is idempotent: a second open finds the shipped PK shape and returns."""
        db_path = _make_born_current_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE messages")
        conn.execute(LEGACY_MESSAGES_SQL)
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (NULL, 's1', 'user', 'x', 1.0)"
        )
        conn.commit()
        conn.close()
        first = SessionDB(db_path=db_path)
        ids_after_first = first._conn.execute("SELECT id FROM messages").fetchall()
        first.close()
        second = SessionDB(db_path=db_path)
        try:
            assert [r[0] for r in second._conn.execute("SELECT id FROM messages").fetchall()] == [
                r[0] for r in ids_after_first
            ]
            assert _pk_cols(second, "messages") == ["id"]
        finally:
            second.close()


class TestSessionsPkHeal:
    def test_bare_id_sessions_rebuilt_with_text_pk_and_create_session_works(self, tmp_path):
        """Ids are preserved verbatim, and the create_session upsert path (ON CONFLICT(id))
        — which fails outright on a bare column — works after the heal."""
        db_path = _make_born_current_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE sessions")
        conn.execute(_legacy_sessions_sql())
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)",
            [("legacy-a", 1.0), ("legacy-b", 2.0)],
        )
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            assert _pk_cols(db, "sessions") == ["id"]
            kept = {r[0] for r in db._conn.execute("SELECT id FROM sessions").fetchall()}
            assert {"legacy-a", "legacy-b"} <= kept
            # The upsert path that failed on the bare column (foma-agent's report):
            db.create_session("s-new", "cli")
            assert "s-new" in {r[0] for r in db._conn.execute("SELECT id FROM sessions").fetchall()}
        finally:
            db.close()

    def test_child_fks_still_name_sessions_after_heal(self, tmp_path):
        """The rename-last rebuild must leave every child FK clause pointing at ``sessions`` —
        an ALTER RENAME-first rebuild would repoint them at the dropped legacy copy."""
        db_path = _make_born_current_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE sessions")
        conn.execute(_legacy_sessions_sql())
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('legacy-a', 'cli', 1.0)")
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            fk_rows = db._conn.execute("PRAGMA foreign_key_list(messages)").fetchall()
            assert fk_rows, "messages must keep an FK clause"
            assert all(r[2] == "sessions" for r in fk_rows), fk_rows
        finally:
            db.close()
