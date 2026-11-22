"""``hermes chat --tui --query-file`` seeds the interactive session like ``-q`` does.

Regression for #158875: the file used to be read only after ``_launch_tui`` had already
replaced the process, so the TUI opened empty and the flag vanished without a trace.
"""
import argparse
import pytest

from hermes_cli import main as cli_main


def _parse(argv):
    from hermes_cli._parser import _build_chat_parser
    parser = argparse.ArgumentParser()
    # The real chat parser is a subparser; build it the way main() does.
    import hermes_cli._parser as parser_mod
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="cmd")
    chat = _build_chat_parser(sub)
    return parser_mod, chat.parse_args(argv[1:])


@pytest.fixture
def launch_capture(monkeypatch):
    seen = {}

    def fake_launch(*args, **kwargs):
        seen.update(kwargs)
        seen["argv"] = args
        raise SystemExit(0)

    monkeypatch.setattr(cli_main, "_launch_tui", fake_launch)
    monkeypatch.setattr(cli_main, "_start_chat_background_prefetch", lambda: None)
    return seen


def test_query_file_seeds_the_tui(tmp_path, monkeypatch, launch_capture):
    seed = tmp_path / "seed.md"
    seed.write_text("Reply with exactly: SEEDED", encoding="utf-8")
    _parser_mod, args = _parse(["chat", "--tui", "--query-file", str(seed)])

    with pytest.raises(SystemExit):
        cli_main.cmd_chat(args)

    assert launch_capture.get("query") == "Reply with exactly: SEEDED"


@pytest.mark.parametrize("shape", ["missing", "empty"])
def test_query_file_fails_loudly_before_the_tui_launches(shape, tmp_path, monkeypatch, launch_capture):
    if shape == "missing":
        path = tmp_path / "nope.md"
    else:
        path = tmp_path / "empty.md"
        path.write_text("   \n", encoding="utf-8")
    _parser_mod, args = _parse(["chat", "--tui", "--query-file", str(path)])

    with pytest.raises(SystemExit) as exit_info:
        cli_main.cmd_chat(args)

    assert exit_info.value.code == 2
    assert "launched" not in launch_capture
