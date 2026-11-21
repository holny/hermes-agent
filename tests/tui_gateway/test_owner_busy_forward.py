"""Regression for #149598: a forwarded correction follows the owner's busy policy."""
import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _legacy, _owner
from tui_gateway import server
from tests.waiting import async_wait_until


def test_forwarded_steer_joins_the_running_owner(monkeypatch, tmp_path):
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = "steer"
        authority._stamp_claim(SessionRef("owned", key), live, claimed)
        delivered = []
        authority.adopt_agent(key, claimed["generation"], SimpleNamespace(steer=lambda text: delivered.append(text) or True))
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "correct this"}})
        assert reply.get("result", {}).get("status") == "steered", reply
        assert delivered == ["correct this"]
        assert len(db.admissions.list(session_id=key)) == 1
        assert turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
def test_explicit_queued_submit_bypasses_owner_correction(monkeypatch, tmp_path, mode):
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = mode
        authority._stamp_claim(SessionRef("owned", key), live, claimed)
        delivered = []
        authority.adopt_agent(key, claimed["generation"], SimpleNamespace(
            steer=lambda text: delivered.append(text) or True,
            redirect=lambda text: delivered.append(text) or True))
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "next", "queued": True}})
        assert reply.get("result", {}).get("status") == "queued", reply
        assert delivered == [] and turns == []
        assert [row.payload["text"] for row in db.admissions.list(session_id=key)] == ["first", "next"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("mode,reason", [("steer", "unsupported_control"),
                                         ("interrupt", "unsupported_control"),
                                         ("steer", "execution_not_ready"),
                                         ("interrupt", "execution_not_ready")])
def test_pre_delivery_correction_refusal_admits_to_owner_fifo(monkeypatch, tmp_path, mode, reason):
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef
    from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient
    from tui_gateway import owner_forward

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = mode
        authority._stamp_claim(SessionRef("owned", key), live, claimed)
        if reason == "unsupported_control":
            authority._managed_workers = {key: object()}
        else:
            authority.adopt_agent(key, claimed["generation"], SimpleNamespace())
        calls, refusals = [], []

        class RecordingClient(_OwnerClient):
            async def rpc(self, method, **params):
                calls.append(method)
                try:
                    return await super().rpc(method, **params)
                except Exception as exc:
                    if method in ("session.steer", "session.redirect"):
                        refusals.append(exc.data.get("reason"))
                    raise

        @contextlib.asynccontextmanager
        async def connect(_home, _endpoint):
            yield RecordingClient(authority)
        monkeypatch.setattr(owner_forward, "connect_owner", connect)
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "next"}})
        assert reply.get("result", {}).get("status") == "queued", (reason, reply)
        assert refusals == [reason]
        assert calls[-2:] == ["session.steer" if mode == "steer" else "session.redirect", "prompt.submit"]
        assert [row.payload["text"] for row in db.admissions.list(session_id=key)] == ["first", "next"]
        assert turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_lost_correction_receipt_after_real_owner_effect(monkeypatch, tmp_path):
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.gateway_client import GatewayClient
    from tui_gateway import owner_forward
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = "interrupt"
        authority._stamp_claim(SessionRef("owned", key), live, claimed)
        delivered, calls = [], []
        authority.adopt_agent(key, claimed["generation"], SimpleNamespace(redirect=lambda text: delivered.append(text) or True))

        @contextlib.asynccontextmanager
        async def wire_owner(_home, _endpoint):
            async def handler(ws):
                socket = SimpleNamespace(write=lambda frame: True, close=lambda: None)
                conn = AuthorityConnection(authority, socket, {"user_id": "desktop"})
                async for raw in ws:
                    request = json.loads(raw)
                    calls.append(request["method"])
                    reply = await conn.dispatch(request)
                    if request["method"] == "session.redirect":
                        await ws.close()
                        return
                    await ws.send(json.dumps(reply))
            async with serve(handler, "127.0.0.1", 0) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    async with GatewayClient(ws) as client:
                        yield client
        monkeypatch.setattr(owner_forward, "connect_owner", wire_owner)
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "correct this"}})
        assert delivered == ["correct this"]
        assert reply["error"]["data"]["reason"] == "SESSION_OWNER_UNCONFIRMED"
        assert "prompt.submit" not in calls
        assert len(db.admissions.list(session_id=key)) == 1 and turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_relay_filters_real_owner_event_generations(tmp_path):
    from gateway.session_events import SessionEvents
    from tui_gateway import owner_forward

    async def exercise():
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "chat", "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        generations = []
        for request_id in ("first", "second", "third"):
            db.admissions.admit(epoch=epoch, principal_id="bot", session_id="chat",
                                request_id=request_id, payload={"text": request_id})
            row = db.admissions.claim(epoch=epoch, session_id="chat")
            generations.append(row.generation)
            db.admissions.settle(epoch=epoch, admission_id=row.admission_id,
                                 generation=row.generation, outcome="completed")
        assert len(set(generations)) == 3
        events = asyncio.Queue()
        stream = SessionEvents()
        stream.fanout.write = lambda frame, **kwargs: events.put_nowait(frame) or True
        for generation in (generations[0], generations[2], generations[1]):
            stream.publish("chat", {"text": str(generation)}, event_type="message.complete",
                           execution={"execution_generation": generation})
        emitted = []
        relay = owner_forward._Relay("ui", {}, "chat", tmp_path, None, "text",
                                     lambda *args: emitted.append(args))
        await relay._relay(SimpleNamespace(events=events), None, generations[1])
        assert emitted == [("message.complete", "ui", {"text": str(generations[1])})]
        db.close()
    asyncio.run(exercise())


def test_forwarded_turn_relay_waits_for_its_real_completion(monkeypatch, tmp_path):
    """The owner drain, not a test publisher, produces both generation-stamped completions."""
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.gateway_client import GatewayClient
    from tui_gateway import owner_forward
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, legacy_turns, emitted = _legacy(monkeypatch, db)

    async def exercise():
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = "queue"
        first_started, second_started = asyncio.Event(), asyncio.Event()
        release_first, release_second = asyncio.Event(), asyncio.Event()
        calls, owner_completions = [], []

        class Agent:
            def run_conversation(self, text):
                calls.append(text)
                return f"answer: {text}"

        agent = Agent()

        async def handle(event):
            gate = release_first if event.text == "first" else release_second
            (first_started if event.text == "first" else second_started).set()
            await gate.wait()
            return agent.run_conversation(event.text)

        authority.runner._handle_message = handle
        authority.runner._adapter_for_source = lambda source: None
        monkeypatch.setattr(authority, "_schedule", SessionAuthority._schedule.__get__(authority))

        @contextlib.asynccontextmanager
        async def wire_owner(_home, _endpoint):
            async def handler(ws):
                loop = asyncio.get_running_loop()

                class Socket:
                    def write(self, frame):
                        if frame.get("params", {}).get("type") == "message.complete":
                            owner_completions.append(frame["params"])
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(ws.send(json.dumps(frame))))
                        return True

                    def close(self):
                        return None

                connection = AuthorityConnection(authority, Socket(), {"user_id": "desktop"})
                try:
                    async for raw in ws:
                        await ws.send(json.dumps(await connection.dispatch(json.loads(raw))))
                finally:
                    await connection.close()

            async with serve(handler, "127.0.0.1", 0) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as ws:
                    async with GatewayClient(ws) as client:
                        yield client

        monkeypatch.setattr(owner_forward, "connect_owner", wire_owner)
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="first", payload={"text": "first"})
        authority._schedule(SessionRef("owned", key))
        await asyncio.wait_for(first_started.wait(), 10)

        reply = await asyncio.to_thread(server.handle_request, {
            "id": "p", "method": "prompt.submit",
            "params": {"session_id": sid, "text": "second"}})
        assert reply.get("result", {}).get("status") == "queued", reply
        rows = db.admissions.list(session_id=key)
        assert len(rows) == 2 and rows[0].generation != rows[1].generation
        assert legacy_turns == []

        release_first.set()
        await asyncio.wait_for(second_started.wait(), 10)
        await async_wait_until(lambda: db.admissions.get(admission_id=rows[0].admission_id).status == "terminal",
                               timeout=10)
        await async_wait_until(lambda: len(owner_completions) == 1, timeout=10)
        assert owner_completions[0]["execution_generation"] == rows[0].generation
        assert not any(event == "message.complete" for event, _sid, _payload in emitted)
        assert calls == ["first"]
        target = db.admissions.get(admission_id=rows[1].admission_id)
        assert target.status == "started" and target.generation != rows[0].generation

        release_second.set()
        await asyncio.wait_for(live.task, 10)
        await async_wait_until(lambda: any(event == "message.complete" for event, _sid, _payload in emitted),
                               timeout=10)
        assert [frame["execution_generation"] for frame in owner_completions] == [
            rows[0].generation, target.generation]
        completions = [payload for event, forwarded_sid, payload in emitted
                       if event == "message.complete" and forwarded_sid == sid]
        assert calls == ["first", "second"]
        assert completions == [{"text": "answer: second", "status": "complete"}]
        assert [row.status for row in db.admissions.list(session_id=key, pending_only=False)] == [
            "terminal", "terminal"]

    try:
        asyncio.run(exercise())
    finally:
        server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("mode,failure", [(mode, failure) for mode in ("steer", "interrupt")
                                          for failure in ("exception", "malformed")])
def test_correction_effect_without_valid_receipt_is_not_retried(monkeypatch, tmp_path, mode, failure):
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef
    from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient
    from tui_gateway import owner_forward

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        authority = _owner(monkeypatch, db, key, epoch)
        live = authority.sessions[key]
        live.busy_input_mode = mode
        authority._stamp_claim(SessionRef("owned", key), live, claimed)
        delivered, calls = [], []
        authority.adopt_agent(key, claimed["generation"], SimpleNamespace(
            steer=lambda text: delivered.append(text) or True,
            redirect=lambda text: delivered.append(text) or True))

        class FaultClient(_OwnerClient):
            async def rpc(self, method, **params):
                calls.append(method)
                result = await super().rpc(method, **params)
                if method in ("session.steer", "session.redirect"):
                    if failure == "exception":
                        raise RuntimeError("receipt lost after effect")
                    return {"status": "unexpected"}
                return result

        @contextlib.asynccontextmanager
        async def connect(_home, _endpoint):
            yield FaultClient(authority)
        monkeypatch.setattr(owner_forward, "connect_owner", connect)
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "correction"}})
        assert delivered == ["correction"]
        assert reply["error"]["data"]["reason"] == "SESSION_OWNER_UNCONFIRMED"
        assert "prompt.submit" not in calls and len(db.admissions.list(session_id=key)) == 1
        assert turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_admission_effect_with_malformed_receipt_is_unconfirmed(monkeypatch, tmp_path):
    from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient
    from tui_gateway import owner_forward

    db = open_persistence(profile_scope(home=tmp_path))
    sid, key, turns, _ = _legacy(monkeypatch, db)
    try:
        seed_session(db, key, "test")
        epoch = db.runtime.begin_epoch(instance_id="gateway")
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        db.admissions.claim(epoch=epoch, session_id=key)
        authority = _owner(monkeypatch, db, key, epoch)

        class FaultClient(_OwnerClient):
            async def rpc(self, method, **params):
                result = await super().rpc(method, **params)
                return {} if method == "prompt.submit" else result

        @contextlib.asynccontextmanager
        async def connect(_home, _endpoint):
            yield FaultClient(authority)
        monkeypatch.setattr(owner_forward, "connect_owner", connect)
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "next"}})
        assert len(db.admissions.list(session_id=key)) == 2 and turns == []
        assert reply["error"]["data"]["reason"] == "SESSION_OWNER_UNCONFIRMED"
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_multiplex_owner_busy_policy_isolated_across_profiles(monkeypatch, tmp_path):
    import yaml
    from agent.secret_scope import set_multiplex_active
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef
    from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient
    from tui_gateway import owner_forward

    class Runner(GatewayConfigLoadersMixin):
        config = SimpleNamespace(multiplex_profiles=True)
        _busy_input_mode = "interrupt"
        _busy_text_mode = "interrupt"
        _draining = False

    runner = Runner()
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, mode in zip(homes, ("steer", "interrupt")):
        home.mkdir()
        (home / "config.yaml").write_text(f"display:\n  busy_input_mode: {mode}\n")
        runner._snapshot_profile_busy_modes(home.name, yaml.safe_load((home / "config.yaml").read_text()))
    set_multiplex_active(True)
    try:
        for index, home in enumerate((homes[0], homes[1], homes[0])):
            db = open_persistence(profile_scope(home=home))
            sid, key, turns, _ = _legacy(monkeypatch, db)
            server._sessions[sid]["profile_home"] = str(home)
            try:
                seed_session(db, key, "test")
                epoch = db.runtime.begin_epoch(instance_id=f"gateway-{index}")
                db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                                    request_id=f"running-{index}", payload={"text": "first"})
                claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
                authority = _owner(monkeypatch, db, key, epoch)
                authority.runner = runner
                live = authority.sessions[key]
                live.source.profile = home.name
                authority._stamp_claim(SessionRef("owned", key), live, claimed)
                delivered = []
                authority.adopt_agent(key, claimed["generation"], SimpleNamespace(
                    steer=lambda text: delivered.append("steer") or True,
                    redirect=lambda text: delivered.append("interrupt") or True))
                @contextlib.asynccontextmanager
                async def connect(_home, _endpoint):
                    assert _home.resolve() == home.resolve()
                    yield _OwnerClient(authority)
                monkeypatch.setattr(owner_forward, "connect_owner", connect)
                async def no_relay(*args):
                    return None
                monkeypatch.setattr(owner_forward._Relay, "_relay", no_relay)
                reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                               "params": {"session_id": sid, "text": "correction"}})
                expected = "steer" if home == homes[0] else "interrupt"
                assert delivered == [expected], reply
                assert reply["result"]["status"] == ("steered" if expected == "steer" else "redirected")
                assert turns == []
            finally:
                server._sessions.pop(sid, None)
                db.close()
    finally:
        set_multiplex_active(False)


@pytest.mark.parametrize("mode,voice", [(mode, voice) for mode in ("steer", "interrupt")
                                       for voice in (True, False)])
def test_owner_decides_whether_stop_ends_voice(monkeypatch, tmp_path, mode, voice):
    from gateway import run_api
    from gateway.session_authority import admission_row
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_session_local_reopen_on_turn import _ended_local_session
    from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _OwnerClient
    from tui_gateway import owner_forward

    authority, _, ref = asyncio.run(_ended_local_session(tmp_path, monkeypatch))
    db, key = authority.persistence, ref.session_id
    sid, _, _, _ = _legacy(monkeypatch, db)
    server._sessions[sid]["session_key"] = key
    try:
        epoch = authority.epoch
        db.admissions.admit(epoch=epoch, principal_id="bot", session_id=key,
                            request_id="running", payload={"text": "first"})
        claimed = admission_row(db.admissions.claim(epoch=epoch, session_id=key))
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        monkeypatch.setattr(owner_forward, "discover_owner", lambda home: "ready")
        class LocalClient(_OwnerClient):
            def __init__(self):
                self.loop, self.events = asyncio.get_running_loop(), asyncio.Queue()
                self.conn = AuthorityConnection(authority, self, {"user_id": "owner"})
        @contextlib.asynccontextmanager
        async def connect_owner(home, endpoint):
            yield LocalClient()
        monkeypatch.setattr(owner_forward, "connect_owner", connect_owner)
        live = authority.sessions[key]
        live.source.chat_id = key
        live.busy_input_mode = mode
        authority._stamp_claim(ref, live, claimed)
        runner = authority.runner
        runner._voice_mode = {key: "voice_only" if voice else "off"}
        runner._voice_key_for_source = lambda source: source.chat_id
        runner._save_voice_modes = lambda: None
        runner.session_authority = authority
        monkeypatch.setattr(run_api, "_ACTIVE_RUNNER", runner)
        monkeypatch.setattr(server, "_tts_stream_stop", lambda **kwargs: None)
        delivered = []
        agent = SimpleNamespace(steer=lambda text: delivered.append(text) or True,
                                redirect=lambda text: delivered.append(text) or True)
        authority.adopt_agent(key, claimed["generation"], agent)
        async def no_relay(*args):
            return None
        monkeypatch.setattr(owner_forward._Relay, "_relay", no_relay)
        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "stop"}})
        if voice:
            assert reply.get("result") == {"voice_stopped": True}, reply
            assert runner._voice_mode[key] == "off" and delivered == []
        else:
            assert reply.get("result", {}).get("status") == ("steered" if mode == "steer" else "redirected"), reply
            assert delivered == ["stop"]
    finally:
        server._sessions.pop(sid, None)
        from persistence_assembly import close_stores_under
        close_stores_under(db.scope.profile_dir)
