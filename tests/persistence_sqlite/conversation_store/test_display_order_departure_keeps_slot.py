"""A display-identity group keeps its slot when a member leaves (#152603)."""
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    get_messages,
    replace_messages,
)
from persistence.sqlite.conversation._store.sessions import create_session


def _twin_pair(db, sid="s"):
    create_session(db, sid, source="desktop")
    original = append_message(db, sid, role="assistant", content="twin", timestamp=100.0)
    later = append_message(db, sid, role="assistant", content="later turn", timestamp=200.0)
    create_session(db, "noise", source="desktop")
    for index in range(400):
        append_message(db, "noise", role="assistant", content=f"n{index}", timestamp=1.0)
    twin = append_message(db, sid, role="assistant", content="twin", timestamp=100.0)
    return original, later, twin


def _slots(path, *row_ids):
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT id, display_order FROM messages WHERE id IN (%s)" % ",".join("?" * len(row_ids)),
            row_ids).fetchall()
    return dict(rows)


@pytest.mark.parametrize("mode", ["archive", "delete"])
def test_departing_original_keeps_display_order_after_upgrade(mode, tmp_path):
    path = tmp_path / "state.db"
    with StoreHolder(path) as db:
        original, later, twin = _twin_pair(db)

    with sqlite3.connect(path) as conn:
        # Historical compacted copies survive an active-only transcript rewrite.
        conn.execute(
            "UPDATE messages SET active = 0, compacted = 1 WHERE id IN (?, ?)",
            (later, twin),
        )
        # Seed the pre-fix trigger definitions on disk, not a mocked initializer.
        for name in ("messages_display_visibility_update", "messages_display_identity_delete"):
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
            ).fetchone()[0]
            conn.execute(f"DROP TRIGGER {name}")
            conn.execute(sql.replace("MIN(COALESCE(peer.display_order, peer.id))", "MIN(peer.id)"))

    slots = _slots(path, original, later, twin)
    group_slot = slots[original]
    assert slots[twin] == group_slot
    assert slots[later] > group_slot

    # Opening an existing database must upgrade its installed triggers as well.
    with StoreHolder(path) as db:
        assert [m["id"] for m in get_messages(db, "s")] == [original]
        replace_messages(db, "s", [], active_only=True, archive_dropped=mode == "archive")
        remaining = get_messages(db, "s", include_inactive=True)
        assert (original in {m["id"] for m in remaining}) == (mode == "archive")
        assert get_messages(db, "s") == []
        assert [m["content"] for m in get_messages(db, "s", include_compacted=True)] == [
            "twin", "later turn",
        ]
        assert [m["id"] for m in get_messages(
            db, "s", include_compacted=True, limit=1, offset=0,
        )] == [twin]
        assert [m["id"] for m in get_messages(
            db, "s", include_compacted=True, limit=1, latest=True,
        )] == [later]

    after = _slots(path, later, twin)
    assert after[twin] == group_slot
    assert after[twin] < after[later]
