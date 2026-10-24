"""Store recovery: a corrupt SQLite file must quarantine and rebuild, not fail forever.

A mangled header with an intact body (see the issue's byte probes) makes every
store touch raise ``file is not a database``. Before the self-heal, each tool
dispatch re-raised it and telemetry stayed down until a human moved the file.
"""

from __future__ import annotations

import sqlite3

from hermes_cli.observability.shared_metrics import SharedMetricsStore


def _resource() -> dict[str, str]:
    return {
        "architecture": "x86_64",
        "hermes_version": "test-version",
        "install_method": "git",
        "os_family": "linux",
    }


def _record_started(store: SharedMetricsStore) -> None:
    store.record_counter(
        "hermes.task_run.started",
        {"entrypoint": "interactive", "execution_surface": "cli"},
        _resource(),
    )


def _mangle_header(path) -> None:
    """Reproduce the observed corruption: keep a SQLite-looking prefix, diverge after it."""
    data = bytearray(path.read_bytes())
    data[:16] = b"SQLit\x17\x03\x03\x00\x13\x00\xa8\xa3\xab\x0e\xff\xce"
    path.write_bytes(data)


def test_corrupt_store_is_quarantined_and_rebuilt(tmp_path):
    database = tmp_path / "metrics.sqlite3"
    store = SharedMetricsStore(database, tmp_path / "outbox")
    _record_started(store)
    _mangle_header(database)

    rebuilt = SharedMetricsStore(database, tmp_path / "outbox")

    # The damaged file was kept aside for forensics, not silently deleted.
    [quarantined] = tmp_path.glob("metrics.sqlite3.corrupt_*")
    assert quarantined.is_file()
    # The fresh store is usable again on both sides of the API.
    assert rebuilt.counter_snapshot() == []
    _record_started(rebuilt)
    [row] = rebuilt.counter_snapshot()
    assert row["metric_name"] == "hermes.task_run.started"
    assert row["value"] == 1


def test_quarantine_drops_stale_sidecars_so_the_rebuild_stays_clean(tmp_path):
    database = tmp_path / "metrics.sqlite3"
    SharedMetricsStore(database, tmp_path / "outbox")
    _mangle_header(database)
    # A leftover WAL from the damaged file must not be replayed onto the rebuild.
    (tmp_path / "metrics.sqlite3-wal").write_bytes(b"stale wal")

    SharedMetricsStore(database, tmp_path / "outbox")

    assert not (tmp_path / "metrics.sqlite3-wal").exists()
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_quarantine_keeps_only_the_newest_copies(tmp_path):
    database = tmp_path / "metrics.sqlite3"
    SharedMetricsStore(database, tmp_path / "outbox")
    for stamp in ("20261001T000000Z", "20261005T000000Z", "20261010T000000Z",
                  "20261015T000000Z", "20261020T000000Z"):
        (tmp_path / f"metrics.sqlite3.corrupt_{stamp}").write_bytes(b"old")
    _mangle_header(database)

    SharedMetricsStore(database, tmp_path / "outbox")

    kept = sorted(path.name.rsplit("_", 1)[1] for path in tmp_path.glob("metrics.sqlite3.corrupt_*"))
    # The oldest pre-existing copy is rotated out; the fresh quarantine plus the
    # four newer survivors make up the cap of five.
    assert kept == ["20261005T000000Z", "20261010T000000Z", "20261015T000000Z",
                    "20261020T000000Z", kept[-1]]
    assert len(kept) == 5


def test_unrelated_database_errors_are_not_swallowed(tmp_path, monkeypatch):
    database = tmp_path / "metrics.sqlite3"
    store = SharedMetricsStore(database, tmp_path / "outbox")

    def unrelated_failure() -> None:
        raise sqlite3.DatabaseError("no such table: telemetry_state")

    monkeypatch.setattr(store, "_ensure_schema_once", unrelated_failure)
    try:
        store._ensure_schema()
    except sqlite3.DatabaseError as exc:
        assert "no such table" in str(exc)
    else:
        raise AssertionError("a non-corruption DatabaseError must propagate")
    assert list(tmp_path.glob("metrics.sqlite3.corrupt_*")) == []
