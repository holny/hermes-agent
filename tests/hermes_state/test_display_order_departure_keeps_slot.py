"""A display-identity group that loses a member keeps its surviving peers' slot.

The write-hand absorb writes every message twice (turn flush, then the absorb
rewrite), so a logical message's twin pair straddles the id domain: the original
lands early, the twin lands after every other session's inserts advanced the
global AUTOINCREMENT. Both share the original's slot (the insert trigger's
sibling inheritance). When a rewind/truncation archives the original, the
survivor must keep that slot — re-pinning it to MIN(peer.id) would stamp an
id-domain value hundreds of slots above rows written minutes later, so the
message renders at the end of the transcript (#152603).
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _twin_pair(db, sid="s"):
    db.create_session(sid, source="desktop")
    original = db.append_message(sid, role="assistant", content="twin", timestamp=100.0)
    later = db.append_message(sid, role="assistant", content="later turn", timestamp=200.0)
    db.create_session("noise", source="desktop")
    for index in range(400):
        db.append_message("noise", role="assistant", content=f"n{index}", timestamp=1.0)
    twin = db.append_message(sid, role="assistant", content="twin", timestamp=100.0)
    return original, later, twin


def _slots(db, *row_ids):
    rows = db._read_all(
        "SELECT id, display_order FROM messages WHERE id IN (%s)" % ",".join("?" * len(row_ids)),
        list(row_ids))
    return {row["id"]: row["display_order"] for row in rows}


@pytest.mark.parametrize("mode", ["archive", "delete"])
def test_archiving_a_twin_original_keeps_the_survivor_slot(db, mode):
    original, later, twin = _twin_pair(db)
    slots = _slots(db, original, later, twin)
    group_slot = slots[original]
    assert slots[twin] == group_slot, "the twin joins the original's slot, not its own id"
    assert slots[later] > group_slot

    if mode == "archive":
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET active = 0 WHERE id = ?", (original,)))
    else:
        db._execute_write(lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (original,)))

    after = _slots(db, later, twin)
    assert after[twin] == group_slot, (
        "the surviving twin keeps the group slot; an id-domain re-pin would move it "
        "hundreds of slots above the later turn")
    assert after[twin] < after[later], "the early message still renders before the later turn"
