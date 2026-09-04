"""Tests for pedigree_graph._threads."""

import os
import subprocess
import sys

import pytest

from pedigree_graph._threads import _reset_thread_state, configure_threads, thread_budget

ENV_VAR = "PEDIGREE_GRAPH_THREADS"

PRINT_BUDGET = "from pedigree_graph._threads import thread_budget\nprint(thread_budget())\n"

CONFIGURE_THEN_PRINT = (
    "from pedigree_graph._threads import configure_threads, thread_budget\n"
    "configure_threads(3)\n"
    "print(thread_budget())\n"
)


@pytest.fixture(autouse=True)
def reset_thread_state():
    _reset_thread_state()
    yield
    _reset_thread_state()


def run_child(script, env_threads=None):
    env = dict(os.environ)
    env.pop(ENV_VAR, None)
    if env_threads is not None:
        env[ENV_VAR] = env_threads
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)


class TestDefault:
    def test_default_is_one(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert thread_budget() == 1


class TestConfigureThreads:
    def test_configured_value_is_returned(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(4)
        assert thread_budget() == 4

    def test_configure_beats_env_var(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "8")
        configure_threads(2)
        assert thread_budget() == 2

    def test_last_configure_before_commit_wins(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(2)
        configure_threads(5)
        assert thread_budget() == 5

    @pytest.mark.parametrize("n", [0, -1, True, 2.0, "2"])
    def test_invalid_n_raises_value_error(self, monkeypatch, n):
        monkeypatch.delenv(ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="requires an int >= 1"):
            configure_threads(n)

    def test_invalid_n_leaves_budget_open(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="requires an int >= 1"):
            configure_threads(0)
        configure_threads(3)
        assert thread_budget() == 3


class TestEnvVar:
    @pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("2", 2), ("16", 16)])
    def test_env_var_is_parsed(self, monkeypatch, raw, expected):
        monkeypatch.setenv(ENV_VAR, raw)
        assert thread_budget() == expected

    @pytest.mark.parametrize("raw", ["0", "-1", "2.5", "abc", "", " 2 ", "+2", "0x2"])
    def test_invalid_env_var_raises_value_error(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_VAR, raw)
        with pytest.raises(ValueError, match=ENV_VAR):
            thread_budget()


class TestCommit:
    def test_reconfiguring_same_value_after_commit_is_accepted(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(4)
        assert thread_budget() == 4
        configure_threads(4)
        assert thread_budget() == 4

    def test_reconfiguring_different_value_after_commit_raises(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(4)
        assert thread_budget() == 4
        with pytest.raises(RuntimeError, match="committed thread budget"):
            configure_threads(2)
        assert thread_budget() == 4

    def test_committed_default_also_rejects_a_different_value(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert thread_budget() == 1
        configure_threads(1)
        with pytest.raises(RuntimeError, match="committed thread budget"):
            configure_threads(2)

    def test_invalid_n_after_commit_raises_value_error(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(4)
        assert thread_budget() == 4
        with pytest.raises(ValueError, match="requires an int >= 1"):
            configure_threads(0)

    def test_env_var_change_after_commit_is_ignored(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "3")
        assert thread_budget() == 3
        monkeypatch.setenv(ENV_VAR, "9")
        assert thread_budget() == 3

    def test_invalid_env_var_after_commit_is_ignored(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "3")
        assert thread_budget() == 3
        monkeypatch.setenv(ENV_VAR, "abc")
        assert thread_budget() == 3

    def test_reset_reopens_the_budget(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        configure_threads(4)
        assert thread_budget() == 4
        _reset_thread_state()
        assert thread_budget() == 1


class TestFreshProcess:
    def test_env_var_alone_sets_the_budget(self):
        proc = run_child(PRINT_BUDGET, env_threads="6")
        assert proc.returncode == 0
        assert proc.stdout.strip() == "6"

    def test_configure_beats_env_var(self):
        proc = run_child(CONFIGURE_THEN_PRINT, env_threads="6")
        assert proc.returncode == 0
        assert proc.stdout.strip() == "3"

    def test_default_without_env_var_or_call(self):
        proc = run_child(PRINT_BUDGET)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "1"


ROOT_CONFIGURE_THEN_PRINT = (
    "from pedigree_graph import configure_threads\n"
    "from pedigree_graph._threads import thread_budget\n"
    "configure_threads(2)\n"
    "print(thread_budget())\n"
)


class TestRootExport:
    def test_root_export_is_the_same_object(self):
        import pedigree_graph

        assert pedigree_graph.configure_threads is configure_threads
        assert "configure_threads" in pedigree_graph.__all__

    def test_root_configure_beats_env_var_in_a_fresh_process(self):
        proc = run_child(ROOT_CONFIGURE_THEN_PRINT, env_threads="7")
        assert proc.returncode == 0
        assert proc.stdout.strip() == "2"
