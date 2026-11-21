"""Catalog ``default: true`` plugins: every home gets each one installed and enabled once.

Real config.yaml files under temp homes and the real catalog parser; the network install is a stand-in
that places the plugin directory (the live install is exercised end to end outside the unit suite).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
from hermes_cli import default_plugins
from hermes_cli.plugin_catalog import entry_from_mapping


def _entry(name: str, tier: str = "official"):
    return entry_from_mapping({"name": name, "repo": "https://github.com/NousResearch/hermes-official-plugins",
                               "sha": "a" * 40, "description": "d", "maintainer": "NousResearch", "tier": tier,
                                "default": True}, name)


_WINDOWS_CRASH_CHILD_SCRIPT = '''import os, sys, threading
from pathlib import Path
root = Path(sys.argv[1])
Path(root, "children", str(os.getpid())).touch()
while not Path(root, "stop").exists():
    threading.Event().wait(.02)
'''


def test_windows_crash_child_script_is_valid_python():
    ast.parse(_WINDOWS_CRASH_CHILD_SCRIPT)


@pytest.fixture
def installs(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # the suite turns fetching off
    calls: list[tuple[Path, str]] = []

    def install_into(home, consent=None):
        def install(name, discard=None):
            calls.append((home, name))
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    return calls


def test_default_is_installed_once_and_a_removal_sticks(tmp_path, installs):
    home = tmp_path / "home"
    home.mkdir()
    entries = [_entry("accent-picker")]

    assert default_plugins.install_defaults(home, entries=entries)["installed"] == ["accent-picker"]
    assert installs == [(home, "accent-picker")]

    (home / "plugins" / "accent-picker").rmdir()  # the user removes it
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": [], "owed": []}
    assert installs == [(home, "accent-picker")], "a removed default came back"
    assert not (home / "plugins" / "accent-picker").exists()

    # A default added to the catalog later still reaches the home.
    entries.append(_entry("second"))
    assert default_plugins.install_defaults(home, entries=entries)["installed"] == ["second"]


def test_a_refused_enable_is_retried_until_the_user_decides(tmp_path, monkeypatch):
    """Co-authored with @BearHuddleston's review patch on #154858."""
    import json
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = tmp_path / "home"
    for name in ("kept-off", "by-hand"):  # disabled by the user; installed by hand with the enable declined
        (home / "plugins" / name).mkdir(parents=True)
    config = home / "config.yaml"
    config.write_text("plugins:\n  disabled: [kept-off]\n", encoding="utf-8")
    calls: list[str] = []

    def install_into(h, consent=None):
        def install(name, discard=None):
            calls.append(name)
            installed = name.removeprefix("hermes-")  # hermes-user-off installs as user-off, like hermes-memory-wiki
            (h / "plugins" / installed).mkdir(parents=True, exist_ok=True)  # the clone lands before the enable
            sidecar = h / "plugins" / ".install-metadata.json"
            rows = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            sidecar.write_text(json.dumps({**rows, installed: {"catalog": {"name": name}}}))
            if calls.count(name) == 1:
                return {"ok": False, "error": "enable refused: dependency resolution failed"}
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    entries = [_entry(name) for name in ("kept-off", "by-hand", "accent-picker", "user-on", "hermes-user-off")]

    # An older install record (top-level catalog_name) under another dir name, disabled by the user.
    (home / "plugins" / "legacy-dir").mkdir()
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({"legacy-dir": {"catalog_name": "hermes-legacy"}}))
    config.write_text("plugins:\n  disabled: [kept-off, legacy-dir]\n", encoding="utf-8")
    entries.append(_entry("hermes-legacy"))

    first = default_plugins.install_defaults(home, entries=entries)
    assert first["owed"] == ["accent-picker", "user-on", "hermes-user-off"]
    # The user settles two of the leftover trees by hand before the next attempt (by their installed names).
    config.write_text("plugins:\n  enabled: [user-on]\n  disabled: [kept-off, user-off]\n", encoding="utf-8")
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": ["accent-picker"], "owed": []}
    assert calls == ["accent-picker", "user-on", "hermes-user-off", "accent-picker"], "retried the wrong trees"
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": [], "owed": []}

    # A tree a refused enable left, which the user then removes, is their removal: never reinstalled.
    entries.append(_entry("removed"))
    assert default_plugins.install_defaults(home, entries=entries)["owed"] == ["removed"]
    (home / "plugins" / "removed").rmdir()
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": [], "owed": []}
    assert calls.count("removed") == 1, "a removed leftover came back"


def test_a_home_that_turned_lazy_installs_off_fetches_nothing(tmp_path, installs):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    assert default_plugins.install_defaults(home, entries=[_entry("accent-picker")])["skipped"] == "lazy-installs-disabled"
    assert installs == []


def test_an_owed_boot_step_stays_owed_until_it_succeeds(tmp_path, monkeypatch):
    """An owed step that raises on a retry is still owed; only a success clears it."""
    from hermes_cli import boot_bootstrap, post_update
    monkeypatch.setattr(boot_bootstrap, "current_install_identity", lambda root: "a" * 40)
    script = iter([{"ok": True, "owed": True}, OSError("profiles dir unreadable"), {"ok": True}])

    def step():
        outcome = next(script)
        if isinstance(outcome, Exception):
            raise outcome
        if not outcome.get("owed"):
            from pm.environments import dependency_home_root
            settled = default_plugins._state(Path(dependency_home_root()), "settled")
            settled.parent.mkdir(parents=True, exist_ok=True)
            settled.write_text("a" * 40)
        return outcome
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", step),))
    root = tmp_path / "install"

    def boot():
        return boot_bootstrap.run_boot_bootstrap(root)["home"]

    assert boot()["install_default_plugins"]["owed"]
    assert boot()["install_default_plugins"]["owed"], "a retry that raised cleared the debt"
    assert boot()["install_default_plugins"] == {"ok": True}
    assert boot() == "skipped"


def test_only_official_and_verified_entries_can_be_default():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_plugin_catalog.py"
    spec = importlib.util.spec_from_file_location("validate_plugin_catalog", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    assert _entry("o").default and _entry("v", tier="verified").default
    assert _entry("c", tier="community").default is False
    errors, _ = validator.validate_entry({"name": "c", "repo": "https://github.com/a/b", "sha": "a" * 40,
                                          "description": "d", "maintainer": "m", "tier": "community",
                                          "default": True})
    assert any("default" in error for error in errors)


def test_a_new_home_or_a_failed_pass_unsettles_the_revision(tmp_path, monkeypatch):
    """@BearHuddleston's review patch on #154858."""
    import subprocess

    import hermes_cli.plugin_catalog
    import pm.plugins_state
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(default_plugins, "_identity", lambda: "rev-1")
    live: list = [None]  # the live catalog is unreadable until the first pass
    monkeypatch.setattr(hermes_cli.plugin_catalog, "fetch_live_catalog", lambda **kwargs: live[0])
    entries = [_entry("accent-picker")]
    monkeypatch.setattr(default_plugins, "default_entries", lambda: list(entries))
    homes = [tmp_path / "home"]
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: list(homes))
    online = [True]

    def install_into(home, consent=None):
        def install(name, discard=None):
            if not online[0]:
                raise OSError("fatal: unable to access 'https://github.com/'")
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    children: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: children.append(argv))

    # Offline: the in-tree default installs, but the pass stays owed until the live catalog is read.
    assert default_plugins.install_and_record(wait=1.0)["owed"] == ["live catalog"]
    assert default_plugins.step_install_default_plugins().get("owed")
    live[0] = {"entries": [{"name": "accent-picker"}]}  # the published doc is read
    default_plugins.install_and_record(wait=1.0)
    assert default_plugins.step_install_default_plugins() == {"ok": True}  # settled for this revision
    # A profile is created; its first boot runs before the detached install finishes.
    homes.append(tmp_path / "work")
    assert default_plugins.spawn_install(homes[-1]) and len(children) == 1
    assert default_plugins.step_install_default_plugins().get("owed"), "the new profile's boot closed early"

    default_plugins.install_and_record(wait=1.0)  # the child installs it: settled again
    assert default_plugins.step_install_default_plugins() == {"ok": True}
    entries.append(_entry("second"))  # a later foreground pass fails (an update's tail, offline)
    online[0] = False
    assert default_plugins.install_and_record(wait=1.0)["owed"] == ["second", "second"]
    assert default_plugins.step_install_default_plugins().get("owed"), "a failed pass left the revision settled"


def test_default_entries_keep_the_tier_and_platform_rules(monkeypatch):
    import dataclasses

    import hermes_cli.plugin_catalog as catalog
    from hermes_platform.host.facts import os_family
    elsewhere = "linux" if os_family() == "win32" else "windows"
    entries = [_entry("here"), dataclasses.replace(_entry("merged"), tier="community"),
               dataclasses.replace(_entry("elsewhere"), platforms=[elsewhere])]
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: entries)
    assert [entry.name for entry in default_plugins.default_entries()] == ["here"]


def test_running_installer_receives_request_without_another_child(tmp_path, monkeypatch):
    import subprocess
    import sys

    import pm.environments
    monkeypatch.setattr(pm.environments, "dependency_home_root", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    holder = subprocess.Popen(
        [sys.executable, "-c", "from hermes_cli.default_plugins import _machine_lock; "
         "import sys, pm.environments; pm.environments.dependency_home_root = lambda: sys.argv[1]; "
         "lock = _machine_lock(2); lock.__enter__(); "
         "print('locked', flush=True); sys.stdin.readline(); lock.__exit__(None, None, None)", str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        children = []
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: children.append(args))
        assert not default_plugins.spawn_install()
        assert children == []
        assert default_plugins._state(tmp_path, "rerun").exists()
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)


def test_concurrent_parents_start_only_one_child(tmp_path):
    import subprocess
    import sys

    script = """
import sys
from pathlib import Path
import hermes_cli._launchers as launchers
import pm.environments
from hermes_cli.default_plugins import spawn_install
root = Path(sys.argv[1])
pm.environments.dependency_home_root = lambda: root
launchers.runtime_command = lambda *args, **kwargs: [sys.executable, '-c',
    'import sys,threading; from pathlib import Path; Path(sys.argv[1]).open("a").write("child\\\\n"); threading.Event().wait(4)',
    str(root / 'children')]
sys.stdin.readline()
print(spawn_install(), flush=True)
"""
    parents = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for _ in range(2)]
    try:
        for parent in parents:
            parent.stdin.write("go\n")
            parent.stdin.flush()
        results = [parent.communicate(timeout=15) for parent in parents]
        assert all(parent.returncode == 0 for parent in parents), results
        assert sorted(result[0].strip() for result in results) == ["False", "True"]
        from tests.waiting import wait_until
        assert wait_until(lambda: (tmp_path / "children").exists(), timeout=5)
        assert (tmp_path / "children").read_text().splitlines() == ["child"]
    finally:
        for parent in parents:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=10)


def test_new_profile_during_cross_process_install_is_rescanned(tmp_path, monkeypatch):
    import subprocess
    import sys

    import pm.environments
    monkeypatch.setattr(pm.environments, "dependency_home_root", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    script = """
import sys
from pathlib import Path
import pm.environments, pm.plugins_state, hermes_cli.plugin_catalog
from hermes_cli import default_plugins as d
root = Path(sys.argv[1])
pm.environments.dependency_home_root = lambda: root
pm.plugins_state.dependency_homes = lambda: [root / 'default'] + ([root / 'new'] if (root / 'new').exists() else [])
hermes_cli.plugin_catalog.fetch_live_catalog = lambda: {'entries': [1]}
d.default_entries = lambda: [object()]
d.fetch_policy = lambda home: 'allowed'
d._identity = lambda: 'revision'
first = True
def install(home, **kwargs):
    global first
    print(home.name, flush=True)
    if first:
        first = False
        sys.stdin.readline()
    return {'installed': [], 'owed': []}
d.install_defaults = install
print(d.install_and_record(wait=5)['owed'], flush=True)
"""
    holder = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "default"
        (tmp_path / "new").mkdir()
        children = []
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: children.append(args))
        assert not default_plugins.spawn_install(tmp_path / "new")
        assert children == []
        holder.stdin.write("continue\n")
        holder.stdin.flush()
        output = holder.stdout.read().splitlines()
        assert holder.wait(timeout=10) == 0
        assert output == ["default", "new", "[]"]
        assert default_plugins._state(tmp_path, "settled").read_text() == "revision"
    finally:
        holder.stdin.close()
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_parent_dies_after_spawn_without_starting_a_second_child(tmp_path):
    import subprocess
    import sys

    (tmp_path / "children").mkdir()
    script = """
import os, sys
from pathlib import Path
import pm.environments, hermes_cli._launchers as launchers
from hermes_cli import default_plugins as d
root = Path(sys.argv[1])
pm.environments.dependency_home_root = lambda: root
launchers.runtime_command = lambda *a, **k: [sys.executable, '-c',
    'import os,sys,threading; from pathlib import Path; Path(sys.argv[1], str(os.getpid())).touch(); threading.Event().wait(8)',
    str(root / 'children')]
if sys.argv[2] == 'crash':
    original = d.subprocess.Popen
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(73)
    d.subprocess.Popen = crash
print(d.spawn_install(), flush=True)
"""
    first = subprocess.run([sys.executable, "-c", script, str(tmp_path), "crash"],
                           capture_output=True, text=True, timeout=15, check=False)
    assert first.returncode == 73, first.stderr
    from tests.waiting import wait_until
    assert wait_until(lambda: list((tmp_path / "children").iterdir()), timeout=5)
    second = subprocess.run([sys.executable, "-c", script, str(tmp_path), "retry"],
                            capture_output=True, text=True, timeout=15, check=False)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "False"
    assert len(list((tmp_path / "children").iterdir())) == 1


@pytest.mark.platforms("windows")
def test_windows_parent_crash_preserves_child_claim_and_recovers(tmp_path):
    import subprocess
    import sys

    from tests.waiting import wait_until

    (tmp_path / "children").mkdir()
    script = """
import os, sys
from pathlib import Path
import pm.environments, hermes_cli._launchers as launchers
from hermes_cli import default_plugins as d
root = Path(sys.argv[1])
pm.environments.dependency_home_root = lambda: root
child_script = CHILD_SCRIPT_PLACEHOLDER
launchers.runtime_command = lambda *a, **k: [sys.executable, '-c', child_script, str(root)]
if sys.argv[2] == 'crash':
    original = d.subprocess.Popen
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(73)
    d.subprocess.Popen = crash
print(d.spawn_install(), flush=True)
"""
    script = script.replace("CHILD_SCRIPT_PLACEHOLDER", repr(_WINDOWS_CRASH_CHILD_SCRIPT))
    ast.parse(script)

    def run(mode):
        return subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                              capture_output=True, text=True, timeout=15, check=False)

    try:
        first = run("crash")
        assert first.returncode == 73, first.stderr
        assert wait_until(lambda: len(list((tmp_path / "children").iterdir())) == 1, timeout=5)
        second = run("retry")
        assert second.returncode == 0, second.stderr
        assert second.stdout.strip() == "False"
        assert len(list((tmp_path / "children").iterdir())) == 1
        (tmp_path / "stop").touch()
        assert wait_until(lambda: not default_plugins._live_child(tmp_path), timeout=10)
        (tmp_path / "stop").unlink()
        third = run("retry")
        assert third.returncode == 0, third.stderr
        assert third.stdout.strip() == "True"
        assert wait_until(lambda: len(list((tmp_path / "children").iterdir())) == 2, timeout=5)
    finally:
        (tmp_path / "stop").touch()


def test_new_profile_backoff_keeps_existing_boot_record_owed(tmp_path, monkeypatch):
    from hermes_cli import boot_bootstrap, post_update
    import pm.environments
    from hermes_cli.memory_provider_migration import _note_failure

    monkeypatch.setattr(pm.environments, "dependency_home_root", lambda: tmp_path)
    monkeypatch.setattr(boot_bootstrap, "current_install_identity", lambda root: "revision")
    monkeypatch.setattr(default_plugins, "_identity", lambda: "revision")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = tmp_path / "new"
    home.mkdir()
    monkeypatch.setattr("pm.plugins_state.dependency_homes", lambda: [home])
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", default_plugins.step_install_default_plugins),))
    record = tmp_path / "boot.json"
    monkeypatch.setattr(boot_bootstrap, "record_path", lambda root: record)
    default_plugins._state(tmp_path, "settled").parent.mkdir(parents=True)
    default_plugins._state(tmp_path, "settled").write_text("revision")
    assert boot_bootstrap.run_boot_bootstrap(tmp_path)["home"]["install_default_plugins"] == {"ok": True}
    _note_failure(tmp_path, default_plugins._BACKOFF)
    assert not default_plugins.spawn_install(home)
    assert boot_bootstrap.run_boot_bootstrap(tmp_path)["home"]["install_default_plugins"]["owed"]


def test_uninspectable_child_claim_does_not_spawn_again(tmp_path, monkeypatch):
    import psutil
    import pm.environments

    monkeypatch.setattr(pm.environments, "dependency_home_root", lambda: tmp_path)
    record = default_plugins._state(tmp_path, "child")
    record.parent.mkdir(parents=True)
    record.write_text("123 456\n")
    monkeypatch.setattr(psutil, "Process", lambda pid: (_ for _ in ()).throw(psutil.AccessDenied(pid)))
    assert default_plugins._live_child(tmp_path)


def test_zombie_child_claim_does_not_block_retry(tmp_path, monkeypatch):
    import psutil

    record = default_plugins._state(tmp_path, "child")
    record.parent.mkdir(parents=True)
    record.write_text("123 456\n")

    class Zombie:
        def create_time(self):
            return 456.0

        def status(self):
            return psutil.STATUS_ZOMBIE

    monkeypatch.setattr(psutil, "Process", lambda pid: Zombie())
    assert not default_plugins._live_child(tmp_path)


@pytest.mark.platforms("linux")
def test_unreaped_linux_child_is_not_a_live_claim(tmp_path):
    import subprocess
    import sys

    from tests.waiting import wait_until

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        import psutil
        assert wait_until(lambda: psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE, timeout=5)
        record = default_plugins._state(tmp_path, "child")
        record.parent.mkdir(parents=True)
        record.write_text(f"{child.pid} {psutil.Process(child.pid).create_time()}\n")
        assert not default_plugins._live_child(tmp_path)
    finally:
        child.wait(timeout=5)


def test_an_unreadable_config_keeps_the_home_owed_and_a_repinned_leftover_is_replaced(tmp_path, monkeypatch):
    """The leftover is replaced by removing that dir, never by forcing over the destination (Greptile)."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("security: [unclosed\n", encoding="utf-8")
    calls: list[tuple] = []

    def install_into(h, consent=None):
        def install(name, discard=None):
            calls.append(name)
            if discard is not None:
                discard.rmdir()
            target = h / "plugins" / name
            if target.exists():
                return {"ok": False, "error": f"Plugin '{name}' already exists."}
            target.mkdir(parents=True, exist_ok=True)
            return {"ok": len(calls) > 1, "error": "enable refused: dependency resolution failed"}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    import dataclasses
    entry = _entry("accent-picker")

    # Fetch nothing, but stay owed: a repaired file is picked up later (unlike allow_lazy_installs: false).
    assert default_plugins.install_defaults(home, entries=[entry]) == {"installed": [], "owed": [str(home)]}
    assert calls == []
    config.write_text("model: {}\n", encoding="utf-8")
    assert default_plugins.install_defaults(home, entries=[entry])["owed"] == ["accent-picker"]  # enable refused
    # The catalog re-pins before the retry: the leftover tree is replaced at the new pin.
    repinned = dataclasses.replace(entry, sha="b" * 40)
    assert default_plugins.install_defaults(home, entries=[repinned])["installed"] == ["accent-picker"]
    assert calls == ["accent-picker", "accent-picker"]


def test_a_retry_that_removed_our_leftover_and_then_failed_stays_owed(tmp_path, monkeypatch):
    """Greptile: the retry removes the leftover with the real remove path, then the install fails before
    publishing (offline). The missing tree is ours, not the user's removal: the next pass tries again."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    attempts: list[str] = []

    def install_into(h, consent=None):
        def install(name, discard=None):
            attempts.append(name)
            if discard is not None:  # the real remove path, as the installer runs it
                from hermes_cli.plugins_cmd_remove import _remove_plugin_core
                from hermes_constants import reset_hermes_home_override, set_hermes_home_override
                token = set_hermes_home_override(h)
                try:
                    _remove_plugin_core(discard)
                finally:
                    reset_hermes_home_override(token)
            if len(attempts) == 1:  # the clone lands, then the enable is refused
                (h / "plugins" / name).mkdir(parents=True)
                (h / "plugins" / name / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
                return {"ok": False, "error": "enable refused: dependency resolution failed"}
            return {"ok": False, "error": "fatal: unable to access 'https://github.com/'"}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    entries = [_entry("accent-picker")]

    assert default_plugins.install_defaults(home, entries=entries)["owed"] == ["accent-picker"]
    assert default_plugins.install_defaults(home, entries=entries)["owed"] == ["accent-picker"]
    assert not (home / "plugins" / "accent-picker").exists()  # the retry removed our leftover
    assert default_plugins.install_defaults(home, entries=entries)["owed"] == ["accent-picker"]
    assert attempts == ["accent-picker"] * 3, "a tree we removed was read as the user's removal"
