"""A gateway-boot discovery pass that dies mid-connect must leave the in-flight stdio server
either cleanly shut down or MANAGED (#158884).

Boot discovery (`discover_mcp_tools` → `_run_discovery_pass`) connects candidates on the MCP
loop under a wall-clock pass budget. A slow-cold-start server (uvx/npm resolvers easily take
minutes) can still be spawning when the budget fires: `_run_on_mcp_loop` cancels the pass
coroutine, and `_discover_and_register_server`'s `except BaseException` cleanup then tries
`await server.shutdown()` from inside a cancelled frame. Whatever survives that — a spawned
child with no adopted entry — has no lifecycle/recycle registration and no retry: it lives
until the gateway exits. Observed in the field as "two stdio servers spawned at boot, never
registered, idle_timeout_seconds inert" (#158884).
"""

import subprocess

import pytest
from tools import mcp_tool
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_lifecycle as _mcp_lifecycle
from tests.waiting import wait_until

# `sleep <oddball>` never speaks the MCP handshake — the transport's initialize wait
# runs out with a REAL child process on the table, the slow-cold-start shape.
_NEVER_READY_COMMAND = ["sleep", "15888459"]
_SLEEPER_PATTERN = "sleep 15888459"


def _leaked_sleepers() -> int:
    out = subprocess.run(["pgrep", "-f", _SLEEPER_PATTERN], capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.strip()])


@pytest.fixture
def mcp_state():
    _mcp_lifecycle.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()
    yield mcp_tool
    _mcp_lifecycle.shutdown_mcp_servers()
    # A ghost (unadopted run task) is invisible to shutdown_mcp_servers; its only outward
    # sign is the child process. Kill our own probe children by their exact oddball argv.
    subprocess.run(["pkill", "-f", _SLEEPER_PATTERN], capture_output=True)
    wait_until(lambda: _leaked_sleepers() == 0, timeout=10)


@pytest.mark.platforms("posix")
def test_connect_timeout_reaps_the_spawned_child(mcp_state, monkeypatch):
    """A per-server connect timeout (no cancellation involved) must not leave the child alive."""
    monkeypatch.setattr(mcp_state, "_MCP_AVAILABLE", True)

    result = _mcp_discovery.register_mcp_servers(
        {"never-ready": {"command": _NEVER_READY_COMMAND[0],
                         "args": _NEVER_READY_COMMAND[1:], "connect_timeout": 3}})
    assert result == []
    assert wait_until(lambda: _leaked_sleepers() == 0, timeout=15), (
        "a connect-timeout server left its spawned child process alive — "
        "no managed entry, no lifecycle, never reaped"
    )


@pytest.mark.platforms("posix")
def test_pass_timeout_leaves_the_spawned_child_managed(mcp_state, monkeypatch):
    """The #158884 shape: the PASS budget (2s) fires while the server is still cold-starting
    (connect_timeout 60 would never fire first). The cancellation path must adopt the in-flight
    task into the managed registry — a spawned child with no adopted entry is never recycled."""
    monkeypatch.setattr(mcp_state, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_state, "_MCP_DISCOVERY_PASS_MAX_SEC", 2)

    with pytest.raises(TimeoutError):
        _mcp_discovery.register_mcp_servers(
            {"never-ready": {"command": _NEVER_READY_COMMAND[0],
                             "args": _NEVER_READY_COMMAND[1:], "connect_timeout": 60}})

    # The caller's TimeoutError races the loop-side cancellation that runs the adoption.
    def _adopted() -> bool:
        with mcp_state._lock:
            return "never-ready" in mcp_state._servers
    assert wait_until(_adopted, timeout=10), (
        "a pass-timeout cancellation left its in-flight spawned child unadopted — "
        "no managed entry, no lifecycle recycle, no retry, process alive"
    )
