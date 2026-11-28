"""A secret source's bootstrap credential (the vault machine token Hermes itself reads
to fetch every other secret) must never reach a spawned child. Provider keys are Tier 2
and already stripped; the vault token outranks them and was not on any list, so a
``terminal("env")`` or any LSP / browser / TTS child inherited it. Each source hands the
token to its own CLI explicitly (``_run_bws_list``, ``_op_child_env``), so no child of
the generic spawn paths needs it."""

import pytest

from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource
from tests.tools._child_env_fixtures import child_env, observe_child
from tools.environments import local

# Default bootstrap env names declared by the bundled sources
# (agent/secret_sources/bitwarden.py, agent/secret_sources/onepassword.py).
VAULT_TOKENS = ["BWS_ACCESS_TOKEN", "OP_SERVICE_ACCOUNT_TOKEN"]
# Operator-shell credentials that stay inheritable on purpose (SECURITY.md §3.2 posture,
# local_env_policy.py AWS note); the fix must not widen the scrub to them.
OPERATOR_KEPT = ["AWS_SECRET_ACCESS_KEY", "MY_CUSTOM_VAR"]


def _seed(monkeypatch):
    for name in [*VAULT_TOKENS, *OPERATOR_KEPT, "OPENAI_API_KEY"]:
        monkeypatch.setenv(name, "fake-" + name)


@pytest.mark.parametrize("builder", [
    pytest.param(lambda: local._make_run_env({}), id="terminal"),
    pytest.param(lambda: local.build_subprocess_env(scrub_secrets=True), id="background"),
    pytest.param(lambda: local.hermes_subprocess_env(), id="helper"),
    pytest.param(lambda: local.hermes_subprocess_env(inherit_credentials=True), id="inherit_credentials"),
])
@pytest.mark.usefixtures("child_env")
def test_vault_bootstrap_tokens_never_reach_spawned_children(monkeypatch, builder):
    _seed(monkeypatch)
    observed = observe_child(builder(), [*VAULT_TOKENS, *OPERATOR_KEPT])
    assert {k: observed[k] for k in VAULT_TOKENS} == dict.fromkeys(VAULT_TOKENS)
    assert {k: observed[k] for k in OPERATOR_KEPT} == {k: "fake-" + k for k in OPERATOR_KEPT}


@pytest.mark.usefixtures("child_env")
def test_sources_still_hand_the_token_to_their_own_cli(monkeypatch):
    # The scrub is about *other* children. The vault CLIs get the token explicitly.
    from agent.secret_sources import onepassword
    from agent.secret_sources.base import source_child_env

    _seed(monkeypatch)
    assert onepassword._op_child_env("op-token")["OP_SERVICE_ACCOUNT_TOKEN"] == "op-token"
    bws_env = source_child_env()
    bws_env["BWS_ACCESS_TOKEN"] = "bws-token"  # what _run_bws_list does
    assert observe_child(bws_env, ["BWS_ACCESS_TOKEN"]) == {"BWS_ACCESS_TOKEN": "bws-token"}


@pytest.mark.usefixtures("child_env")
def test_launch_home_credentialed_children_keep_the_token(monkeypatch, tmp_path):
    """A gateway Hermes starts for its own home and a ``key_cmd`` helper (``op read``) authenticate
    to the vault themselves, so they keep the token, also when the process env (systemd, Docker -e)
    holds it. A routed profile's child never gets the launch home's token."""
    import shlex
    import sys

    from agent.command_token_source import _mint
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import get_hermes_home

    _seed(monkeypatch)
    other = tmp_path / "hermes" / "profiles" / "other"
    other.mkdir(parents=True)
    own = local.served_profile_child_env(target_home=str(get_hermes_home()), inherit_credentials=True)
    routed = local.served_profile_child_env(target_home=str(other), inherit_credentials=True)
    token = set_secret_scope({}, profile_home=str(other))  # a served profile's scope, no target
    try:
        scoped = local.served_profile_child_env(inherit_credentials=True)
    finally:
        reset_secret_scope(token)
    assert observe_child(own, VAULT_TOKENS) == {k: "fake-" + k for k in VAULT_TOKENS}
    assert observe_child(routed, VAULT_TOKENS) == dict.fromkeys(VAULT_TOKENS)
    assert observe_child(scoped, VAULT_TOKENS) == dict.fromkeys(VAULT_TOKENS)
    key_cmd = shlex.join([sys.executable, "-c", "import os; print(os.environ['OP_SERVICE_ACCOUNT_TOKEN'])"])
    assert _mint(key_cmd, "probe")[0] == "fake-OP_SERVICE_ACCOUNT_TOKEN"


# =========================================================================
# Plugin-registered secret sources (#161034)
# =========================================================================

class _PluginVaultSource(SecretSource):
    """A plugin source the way PluginContext.register_secret_source() delivers one: it registers
    during plugin discovery, after local_env_policy's import-time snapshots froze."""

    api_version = SECRET_SOURCE_API_VERSION
    shape = "bulk"
    name = "plugenvault"
    scheme = "plugenvault://"
    token_env_key = None
    default_token_env = "PLUGIN_VAULT_TOKEN"

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(secrets={})


@pytest.fixture
def plugin_vault_source():
    import agent.secret_sources.registry as reg
    reg._reset_registry_for_tests()
    source = _PluginVaultSource()
    assert reg.register_source(source), "plugin secret source registration failed"
    yield source
    reg._reset_registry_for_tests()


@pytest.mark.parametrize("builder", [
    pytest.param(lambda: local._make_run_env({}), id="terminal"),
    pytest.param(lambda: local.build_subprocess_env(scrub_secrets=True), id="background"),
    pytest.param(lambda: local.hermes_subprocess_env(), id="helper"),
    pytest.param(lambda: local.hermes_subprocess_env(inherit_credentials=True), id="inherit_credentials"),
])
@pytest.mark.usefixtures("child_env", "plugin_vault_source")
def test_plugin_source_bootstrap_token_never_reaches_children(monkeypatch, builder):
    """A plugin source registers after the frozen strip sets snapped, so its bootstrap token
    kept riding into terminal, background and credentialed children. Tier 1 is resolved at
    spawn time now; the built-ins' tokens were already covered by the tests above."""
    _seed(monkeypatch)
    monkeypatch.setenv("PLUGIN_VAULT_TOKEN", "fake-PLUGIN_VAULT_TOKEN")
    observed = observe_child(builder(), ["PLUGIN_VAULT_TOKEN", "BWS_ACCESS_TOKEN"])
    assert observed["PLUGIN_VAULT_TOKEN"] is None
    assert observed["BWS_ACCESS_TOKEN"] is None


@pytest.mark.usefixtures("child_env", "plugin_vault_source")
def test_plugin_source_token_cannot_be_passthrough_registered():
    """GHSA-rhgp-j443-p4rf's refusal must see a name that owns a bootstrap token NOW, even
    though the import-time blocklist never knew the source: a skill registering the token
    name before the plugin loads must not tunnel it into sandboxed children."""
    from tools.env_passthrough import register_env_passthrough
    refused = register_env_passthrough(["PLUGIN_VAULT_TOKEN", "UNRELATED_PASSTHROUGH_VAR"])
    assert refused == ["PLUGIN_VAULT_TOKEN"]
