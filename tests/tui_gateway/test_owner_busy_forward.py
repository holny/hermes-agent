"""A Desktop prompt forwarded while the owner's session runs a turn follows that session's own
busy policy (#149598) instead of always joining the owner's FIFO behind it: ``steer`` joins the
running turn, ``interrupt`` takes a bare stop phrase as a stop that reaches no agent, and
``queue`` keeps the FIFO admission. Real: the SessionDB, the started ledger claim, the legacy
``prompt.submit`` and its ledger gate, and the owner (``SessionAuthority`` +
``AuthorityConnection``: attach, the busy config read, the correction, and settlement
publication). Replaced: only the owner's discovery and socket dial, by an in-process client
speaking to the same ``AuthorityConnection``."""
import contextlib
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input, list_session_admissions,
)
from tui_gateway import server
from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient, _legacy
from tests.waiting import wait_until


def _running_owner(monkeypatch, db, key, epoch, mode):
    """The ready owner with the session's busy mode set the way a ``config.set key=busy`` would."""
    from gateway.session_authority import LiveSession, SessionAuthority
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda _k: None)
    authority = SessionAuthority(runner, profile_id="owned", instance_id="gateway", db=db, epoch=epoch)
    authority.sessions[key] = LiveSession(SimpleNamespace(platform=None, user_id="desktop"), "route")
    authority.sessions[key].busy_input_mode = mode
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    @contextlib.asynccontextmanager
    async def connect(_home, endpoint):
        yield _OwnerClient(authority)
    monkeypatch.setattr("tui_gateway.owner_forward.discover_owner", lambda _home: "ready-owner")
    monkeypatch.setattr("tui_gateway.owner_forward.connect_owner", connect)
    return authority


def _started_turn(db, key):
    """The gateway is running a turn on this session: a claimed (started) ledger admission."""
    epoch = begin_runtime_epoch(db, instance_id="gateway")
    bot = admit_session_input(db, epoch=epoch, principal_id="bot", session_id=key, request_id="dm-1",
                              payload={"text": "a message"})
    claimed = claim_session_input(db, epoch=epoch, session_id=key)
    return epoch, bot, claimed


def test_a_forward_in_steer_mode_joins_the_running_turn(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        epoch, bot, claimed = _started_turn(db, key)
        authority = _running_owner(monkeypatch, db, key, epoch, "steer")
        from gateway.session_contract import SessionRef
        ref = SessionRef("owned", key)
        authority._stamp_claim(ref, authority.sessions[key], claimed)

        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "focus on the refund path"}})

        assert "error" not in reply, reply
        assert reply["result"]["status"] == "steered" and reply["result"]["forwarded"] is True
        # Joined the turn, not the FIFO: the ledger still holds exactly the started admission, and
        # the words are latched for the agent this claim adopts.
        assert [r["status"] for r in list_session_admissions(db, session_id=key)] == ["started"]
        assert authority.pending_steers[key] == (claimed["generation"], ["focus on the refund path"])
        assert turns == [] and db.get_messages_as_conversation(key) == []

        # The running turn's completion settles the Desktop bubble.
        from gateway.session_settlement_recovery import commit_and_publish
        live = authority.sessions[key]
        with live.event_stream.lock:
            authority.executing.pop(key, None)
        commit_and_publish(authority, live, ref, claimed, "THE ANSWER", "completed", None, {})

        wait_until(lambda: any(e == "message.complete" for e, _s, _p in emitted), timeout=10)
        assert [p for e, s, p in emitted if s == sid and e == "message.complete"] == [
            {"text": "THE ANSWER", "status": "complete"}]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_a_forwarded_stop_phrase_in_interrupt_mode_stops_without_queueing(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        epoch, bot, claimed = _started_turn(db, key)
        authority = _running_owner(monkeypatch, db, key, epoch, "interrupt")
        from gateway.session_contract import SessionRef
        authority._stamp_claim(SessionRef("owned", key), authority.sessions[key], claimed)

        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "stop"}})

        assert "error" not in reply, reply
        assert reply["result"].get("voice_stopped") is True
        # The stop reached no agent and queued nothing: the ledger keeps the started admission only,
        # and the stop is latched for the agent this claim adopts (adopt_agent fires it).
        assert [r["status"] for r in list_session_admissions(db, session_id=key)] == ["started"]
        assert authority.pending_stops[key] == claimed["generation"]
        assert turns == [] and db.get_messages_as_conversation(key) == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_queue_mode_still_admits_behind_a_running_turn(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        epoch, _bot, claimed = _started_turn(db, key)
        authority = _running_owner(monkeypatch, db, key, epoch, "queue")
        from gateway.session_contract import SessionRef
        authority._stamp_claim(SessionRef("owned", key), authority.sessions[key], claimed)

        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "my own question"}})

        assert "error" not in reply, reply
        assert reply["result"]["status"] == "queued" and reply["result"]["forwarded"] is True
        rows = list_session_admissions(db, session_id=key)
        assert [r["status"] for r in rows] == ["started", "queued"]
        assert rows[1]["payload"]["text"] == "my own question"
        assert turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()
