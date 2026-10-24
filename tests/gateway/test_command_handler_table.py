"""The eager slash-handler table must degrade, not brick every messaging turn.

``_command_handler_table`` runs on every idle-dispatched message before command/text
classification, so one name listed without a resolvable ``_handle_<name>_command`` used to
raise AttributeError for every inbound message on every platform (#143844).
"""

from gateway.run import GatewayRunner
from gateway.run_busy import GatewayBusySessionMixin


class _PartialCommands(GatewayBusySessionMixin):
    """Only some declared commands resolve — the drift shape from #143844."""

    async def _handle_status_command(self, event):
        return "status"

    async def _handle_set_home_command(self, event):
        return "home"


def test_missing_handler_is_skipped_not_raised():
    table = _PartialCommands()._command_handler_table(
        ("status", "sethome", "autoprune")
    )
    assert set(table) == {
        "status",
        "sethome",
    }  # autoprune has no handler: degrade, don't raise


def test_alias_entry_resolves_to_its_aliased_method():
    table = _PartialCommands()._command_handler_table(("sethome",))
    assert table["sethome"].__func__ is _PartialCommands._handle_set_home_command


def test_every_declared_command_resolves_on_gateway_runner():
    # Class-level drift guard: a name listed in _PLAIN_COMMANDS/_IDLE_COMMANDS whose handler
    # only exists on a CLI-side mixin would silently degrade to "unknown command" after the
    # fix above — this pins the full pairing so drift fails in CI, not in production.
    missing = []
    for name in (
        *GatewayBusySessionMixin._PLAIN_COMMANDS,
        *GatewayBusySessionMixin._IDLE_COMMANDS,
    ):
        handler = GatewayBusySessionMixin._COMMAND_HANDLER_ALIASES.get(
            name, f"_handle_{name.replace('-', '_')}_command"
        )
        if not hasattr(GatewayRunner, handler):
            missing.append(f"/{name} -> {handler}")
    assert not missing, (
        f"declared commands without a gateway handler: {', '.join(missing)}"
    )
