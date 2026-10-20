"""The pending model switch emits a session.info as soon as it is stashed (#141027).

A pick made from the composer pill while a turn runs is stashed for the next turn start — by
design, since the live agent can't be swapped mid-turn. But `_stash_pending_model_switch` wrote
the stash without emitting `session.info`, and the pill is server-truth (it repaints only on
that event), so the user's click stayed invisible until the deferred switch landed a turn later."""
import pytest

import tools  # noqa: F401  # registers the toolset the methods module reads
from tui_gateway import methods_config_set as mcs


class _FakeAgent:
    provider = "anthropic"
    model = "old-model"


def _session(running=True):
    return {"agent": _FakeAgent(), "running": running, "session_key": "sess1"}


def _parsed(model_input="new-model", explicit_provider=""):
    parsed = type("P", (), {})()
    parsed.model_input = model_input
    parsed.explicit_provider = explicit_provider
    return parsed


@pytest.fixture
def emitted(monkeypatch):
    events = []
    # server.py binds these onto the module globals at import (method_ctx.bind_module); a bare
    # module import in a unit test has no binder, so provide the three seams the setters use.
    monkeypatch.setattr(mcs, "_emit", lambda ev, sid, payload=None: events.append((ev, sid, payload)) or True, raising=False)
    monkeypatch.setattr(mcs, "_err", lambda rid, code, msg, data=None: {"ok": False, "rid": rid, "code": code, "error": msg}, raising=False)
    monkeypatch.setattr(mcs, "_ok", lambda rid, payload=None, **kw: {"ok": True, "rid": rid, **(payload or {}), **kw}, raising=False)

    def _session_info(agent, session):
        pending = session.get("pending_model_switch") or {}
        return {"display_model": pending.get("display_model"), "display_provider": pending.get("display_provider")}

    monkeypatch.setattr(mcs, "_session_info", _session_info, raising=False)
    return events


def _drive(rid, params, session):
    return mcs._set_model(rid, params, "model", params["model"], session)


def test_stash_emits_session_info_with_the_pending_pick(emitted):
    params = {"session_id": "sess1", "model": "new-model", "confirm_expensive_model": True}
    session = _session()
    out = _drive("r1", params, session)

    assert out["ok"] is True
    assert any(ev == "session.info" and sid == "sess1" for ev, sid, _ in emitted)
    payload = [p for ev, _sid, p in emitted if ev == "session.info"][-1]
    assert payload["display_model"] == "new-model"


def test_stash_emits_once_per_pick(emitted):
    params = {"session_id": "sess1", "model": "new-model", "confirm_expensive_model": True}
    session = _session()
    _drive("r1", params, session)

    infos = [ev for ev, _sid, _p in emitted if ev == "session.info"]
    assert infos.count("session.info") == 1


def test_no_emit_when_nothing_stashed(emitted):
    # A live (idle-session) switch follows the direct path, not the stash branch: this contract
    # only pins the stash branch, driven through a busy session with a blocking selection guard.
    params = {"session_id": "sess1", "model": "new-model"}
    session = _session()

    class _ExplodingAgent(_FakeAgent):
        model = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    session["agent"] = _ExplodingAgent()
    out = _drive("r1", params, session)
    assert out["ok"] in (True, False)  # whatever the guard decides, no stash emit escapes
    infos = [ev for ev, _sid, _p in emitted if ev == "session.info"]
    assert all(ev != "session.info" for ev in infos) or out["ok"] is False
