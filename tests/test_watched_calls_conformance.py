"""The watched relationship calls take the steps the formal-validate models assume.

TLC checks the models ``RJob.tla`` and ``PyWatched.tla`` (``models/watched_calls/``
in the private companion repo formal-validate, ``external/formal-validate`` in
the simACE umbrella), not the code; these tests pin each model action to the
code it stands for, in the model's order. They read the source, because the
order of a store and a notify, or of a start and its cancel guard, is what the
models depend on and what a timing test rarely sees. When one fails, the code
no longer matches the model: update the model (or the code), rerun
``pixi run watched-calls`` in formal-validate (with ``PEDIGREE_GRAPH_ROOT``
pointing at this checkout; ``bash models/watched_calls/check.sh --mutants``
for the mutants), and move the sync commit in its
``.agents/skills/watched-calls/SKILL.md``.

Behaviour the models also assume is pinned elsewhere: the ``job.rs`` unit
tests (``pixi run -e r test-r-rust``), ``r/tests/testthat/test-progress.R``,
``tests/test_progress.py`` and ``tests/test_panic_boundary.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB_RS = ROOT / "r" / "src" / "rust" / "src" / "job.rs"
R_LIB_RS = ROOT / "r" / "src" / "rust" / "src" / "lib.rs"
PROGRESS_R = ROOT / "r" / "R" / "progress.R"
WRAPPERS_R = ROOT / "r" / "R" / "wrappers.R"
KERNEL_R = (ROOT / "r" / "R" / "kernels.R", ROOT / "r" / "R" / "moments.R")
PYTHON_LIB_RS = ROOT / "crates" / "python" / "src" / "lib.rs"
CORE_PROGRESS_RS = ROOT / "crates" / "core" / "src" / "relationships" / "progress.rs"
GUARD_C = ROOT / "r" / "src" / "guard.c"
ENTRYPOINT_C = ROOT / "r" / "src" / "entrypoint.c"

_RUST_CHAR = re.compile(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])'")


class Source:
    """A source file with comments removed and whitespace squashed to one space.

    ``code`` is that text; ``mask`` is the same text with the inside of every
    string and character literal replaced by ``_``, so brackets are counted on
    the mask and read from the code.
    """

    def __init__(self, path: Path) -> None:
        rust = path.suffix in {".rs", ".c"}  # C comments, strings and chars
        text = path.read_text(encoding="utf-8")
        code: list[str] = []
        mask: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if (rust and text.startswith("//", i)) or (not rust and ch == "#"):
                i = text.find("\n", i)
                i = n if i < 0 else i
                continue
            if rust and text.startswith("/*", i):
                i = text.index("*/", i) + 2
                continue
            if rust and ch == "'":
                literal = _RUST_CHAR.match(text, i)
                if literal is None:  # a lifetime
                    code.append(ch)
                    mask.append(ch)
                    i += 1
                    continue
                end = literal.end()
            elif ch == '"' or (not rust and ch in "'`"):
                end = i + 1
                while text[end] != ch:
                    end += 2 if text[end] == "\\" else 1
                end += 1
            else:
                space = ch.isspace()
                if space and code and code[-1] in " ([":
                    i += 1
                    continue
                if ch in ")]":  # no space before a closing bracket, no trailing comma
                    for drop in " ,":
                        if code and code[-1] == drop:
                            code.pop()
                            mask.pop()
                code.append(" " if space else ch)
                mask.append(" " if space else ch)
                i += 1
                continue
            code.extend(text[i:end])
            mask.extend(ch + "_" * (end - i - 2) + ch)
            i = end
        self.path = path
        self.code = "".join(code)
        self.mask = "".join(mask)

    def find(self, snippet: str, start: int = 0) -> int:
        """Where ``snippet`` (squashed) starts at or after ``start``, else -1."""
        return self.code.find(_squash(snippet), start)

    def closing(self, open_at: int) -> int:
        """The index just past the bracket that closes the one at ``open_at``."""
        pairs = {"{": "}", "(": ")"}
        opener = self.mask[open_at]
        closer = pairs[opener]
        depth = 0
        for i in range(open_at, len(self.mask)):
            depth += {opener: 1, closer: -1}.get(self.mask[i], 0)
            if depth == 0:
                return i + 1
        raise AssertionError(f"{self.path.name}: unbalanced {opener} at {open_at}")

    def block(self, header: str, within: str | None = None) -> str:
        """The braced block after the first ``header`` (inside the block after ``within``)."""
        start, stop = 0, len(self.code)
        if within is not None:
            start = self.find(within)
            assert start >= 0, f"{self.path.name}: no {within!r}; the code the model names has moved"
            stop = self.closing(self.mask.index("{", start))
        at = self.find(header, start)
        assert 0 <= at < stop, f"{self.path.name}: no {header!r}; the code the model names has moved"
        open_at = self.mask.index("{", at + len(_squash(header)))
        return self.code[open_at : self.closing(open_at)]


def _squash(text: str) -> str:
    """Whitespace as ``Source`` keeps it: one space, none inside brackets, no trailing comma."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([(\[]) ", r"\1", text)
    return re.sub(r" ?,? ?([)\]])", r"\1", text)


def _in_order(body: str, *steps: str) -> None:
    """Each step occurs in ``body``, after the end of the one before it."""
    at = 0
    for step in steps:
        found = body.find(_squash(step), at)
        assert found >= 0, f"expected {step!r} after the previous step in:\n{body}"
        at = found + len(_squash(step))


def _r_fn(name: str) -> str:
    return Source(PROGRESS_R).block(f"{name} <- function(")


# --- RJob.tla ----------------------------------------------------------------


def test_begin_guard_start_pgcall():
    """Begin, Start, PgCall: the native guard runs the start and then the watch (#68)."""
    body = _r_fn(".pg_run")
    _in_order(
        body,
        "report <- .pg_progress(progress, label, call = call)",
        ".External(pg_run_guarded, start, function(handle) {",
        ".pg_watch(.pg_call(handle, call = call), report, call = call)",
    )
    assert "on.exit" not in body, "the cancel must not depend on an R on.exit (#68)"


def test_guard_stores_the_handle_and_cancels_natively():
    """Start, Exit, CancelFlag: the handle is stored as start() returns; the cleanup is native.

    R runs R_ExecWithCleanup's cleanup as the context's `cend`, without
    eval(), on return and on every jump (R 4.5.3 src/main/context.c:128,
    :909), so no interrupt can skip the cancel. No R code runs between
    `start()` returning and the handle landing in the protected box.
    """
    src = Source(GUARD_C)
    _in_order(
        src.block("static SEXP run(void *data)"),
        "SET_VECTOR_ELT(g->box, 0, Rf_eval(start, R_GlobalEnv));",
        "Rf_lang2(g->watch, VECTOR_ELT(g->box, 0))",
        "Rf_eval(watch, R_GlobalEnv)",
    )
    _in_order(
        src.block("static void cancel(void *data)"),
        "if (TYPEOF(handle) == EXTPTRSXP) { wrap__job_cancel(handle); }",
    )
    _in_order(
        src.block("SEXP pg_run_guarded(SEXP args)"),
        "g.box = PROTECT(Rf_allocVector(VECSXP, 1));",
        "R_ExecWithCleanup(run, &g, cancel, &g)",
        "UNPROTECT(1);",
    )
    entry = Source(ENTRYPOINT_C)
    assert entry.find('{"pg_run_guarded", (DL_FUNC) &pg_run_guarded, 2}') >= 0
    _in_order(
        entry.block("void R_init_pedigreegraph(void *dll)"),
        "R_init_pedigreegraph_extendr(dll);",
        "R_registerRoutines(dll, NULL, NULL, NULL, external_routines);",
        "R_useDynamicSymbols(dll, FALSE);",
    )


def _start_closures() -> list[tuple[str, str]]:
    """(label, closure body) of every `.pg_run(progress, label, function() ...)`."""
    found = []
    for path in KERNEL_R:
        src = Source(path)
        for m in re.finditer(r'\.pg_run\(progress, "(\w+)", function\(\) ', src.code):
            at = m.end()
            end = src.closing(at) if src.code[at] == "{" else src.closing(src.mask.index("(", at))
            found.append((m.group(1), src.code[at:end]))
    return found


def test_each_start_is_one_wrapper_call():
    """Start: `start()` is one call of a `.native_start_*` wrapper and nothing else.

    RJob.tla's Start is one step that R cannot interrupt between the .Call's
    return and `handle <-` (R 4.5.3 checks for interrupts only on eval()
    entry and bytecode entries, calls and back branches). Code in the closure
    after the call would open that window.
    """
    closures = _start_closures()
    assert {label for label, _ in closures} == {
        "relationship_pairs",
        "relationship_counts",
        "relationship_burden",
        "relatives_per_person",
        "relationship_moments",
    }
    for label, body in closures:
        inner = body[2:-2] if body.startswith("{ ") else body
        assert re.fullmatch(r"\.native_start_\w+\(.*\)", inner), (label, body)
        assert inner.count(".native_start_") == 1, (label, body)


def test_each_start_wrapper_is_one_call():
    """Start: each `.native_start_*` wrapper is one bare .Call of a `wrap__*_start`."""
    src = Source(WRAPPERS_R)
    defined = re.findall(r"(\.native_start_\w+) <- function\(", src.code)
    assert len(defined) == 5
    for name in defined:
        at = src.find(f"{name} <- function(")
        params_end = src.closing(src.mask.index("(", at + len(name)))
        body_at = params_end + 1
        end = src.closing(body_at) if src.code[body_at] == "{" else src.closing(src.mask.index("(", body_at))
        body = src.code[body_at:end]
        inner = body[2:-2] if body.startswith("{ ") else body
        assert re.fullmatch(r"\.Call\(wrap__\w+_start, .*\)", inner), (name, body)


@pytest.mark.parametrize("verb", ["wait", "collect", "cancel"])
def test_handle_wrappers_reach_job_rs(verb):
    """Wait, Collect, Exit: `.native_<verb>` -> `wrap__job_<verb>` -> `job::<verb>`."""
    arg = "handle, tick" if verb == "wait" else "handle"
    assert Source(WRAPPERS_R).find(f".native_{verb} <- function({arg}) .Call(wrap__job_{verb}, {arg})") >= 0
    rust_arg = "handle: Robj, tick: f64" if verb == "wait" else "handle: Robj"
    call = "&handle, tick" if verb == "wait" else "&handle"
    _in_order(Source(R_LIB_RS).block(f"fn job_{verb}({rust_arg}) -> Robj"), f"finish(job::{verb}({call}))")


def test_wait_service_then_break_then_report_then_collect():
    """Wait/WaitWake, Service, Report, Collect: events are serviced before the break."""
    _in_order(
        _r_fn(".pg_watch"),
        "repeat {",
        ".native_wait(handle, .pg_tick())",
        ".pg_service_events()",
        "if (state$finished) break",
        "report(state)",
        ".native_collect(handle)",
    )


def test_service_events_is_a_sleep():
    """Service: R raises pending events in Sys.sleep()."""
    assert Source(PROGRESS_R).find(".pg_service_events <- function() Sys.sleep(.pg_event_sleep)") >= 0


def test_work_store_notify():
    """Work, Store, Notify: the outcome lands Ready under the lock, then notify_all."""
    _in_order(
        Source(JOB_RS).block("pub fn spawn<T: Send + 'static>(", within="impl Task {"),
        "Slot::Running",
        "pool.spawn(move || {",
        "panic::catch_unwind(",
        "*worker.slot() = Slot::Ready(outcome);",
        "worker.ready.notify_all();",
    )


def test_wait_reports_finished_once_not_running():
    """Wait, WaitWake: a timed Condvar wait, then finished unless Running."""
    src = Source(JOB_RS)
    _in_order(
        src.block("pub fn wait(&self, tick: Duration)"),
        ".wait_timeout_while(self.shared.slot(), tick, |slot| { matches!(slot, Slot::Running) })",
        "if !matches!(*slot, Slot::Running) { return None; }",
        "self.shared.progress.snapshot()",
    )
    _in_order(src.block("pub fn wait(handle: &Robj, tick: f64)"), "of(handle)?.task.wait(tick)")


def test_until_ready_has_no_timeout():
    """UntilReadyWake: collect and cancel wait on the Condvar with no tick."""
    _in_order(
        Source(JOB_RS).block("fn until_ready(&self)"),
        ".wait_while(self.slot(), |slot| matches!(slot, Slot::Running))",
    )


def test_collect_takes_the_task():
    """Collect, TakeNow: `Job::collect` -> `Task::take`: until_ready, then Collected."""
    src = Source(JOB_RS)
    _in_order(src.block("pub fn collect(handle: &Robj)"), "of(handle)?.collect()")
    _in_order(
        src.block("pub fn collect(&self) -> HostResult<Robj>"),
        "let finish = self.finish.take();",
        "match self.task.take() {",
        "Ok(Ok(out)) => finish",
    )
    _in_order(
        src.block("pub fn take(&self)"),
        "std::mem::replace(&mut *self.shared.until_ready(), Slot::Collected)",
    )


def test_cancel_sets_the_flag_then_waits():
    """CancelFlag, CancelLock: the flag, then until_ready, then Collected."""
    _in_order(
        Source(JOB_RS).block("pub fn cancel(&self)", within="impl Task {"),
        "self.shared.progress.cancel();",
        "std::mem::replace(&mut *self.shared.until_ready(), Slot::Collected)",
    )


def test_job_cancel_and_drop_cancel_the_task():
    """CancelFlag (the guard's wrap__job_cancel) and GcFlag (the finalizer) both run Task::cancel."""
    src = Source(JOB_RS)
    _in_order(src.block("pub fn cancel(handle: &Robj)"), "of(handle)?.cancel();")
    _in_order(src.block("pub fn cancel(&self)", within="impl Job {"), "self.task.cancel();")
    _in_order(src.block("impl Drop for Job"), "self.task.cancel();")


def test_progress_cancel_is_one_store_and_checks_read_it():
    """CancelFlag is one atomic store; Work's Error::Cancelled comes from the check."""
    src = Source(CORE_PROGRESS_RS)
    assert src.block("pub fn cancel(&self)") == "{ self.cancelled.store(true, Ordering::Relaxed); }"
    _in_order(
        src.block("fn check(&self) -> Result<(), Error>"),
        "if self.cancelled.load(Ordering::Relaxed) { Err(Error::Cancelled) }",
    )


# --- PyWatched.tla -----------------------------------------------------------


def test_done_stores_finished_then_unparks():
    """Store, Unpark: the drop guard stores `finished`, then unparks the caller."""
    _in_order(
        Source(PYTHON_LIB_RS).block("impl Drop for Done<'_>"),
        "self.finished.store(true, Ordering::Release);",
        "self.caller.unpark();",
    )


def _run_watched() -> str:
    return Source(PYTHON_LIB_RS).block("fn run_watched<T: Send>(")


def test_the_guard_covers_the_work():
    """Work: the guard exists before the work runs, so a panic still drops it."""
    _in_order(
        _run_watched(),
        "scope.spawn(|_| {",
        "let _done = Done {",
        "slot = Some(work(&progress));",
    )


def test_watch_loop_load_park_tick():
    """Load, Loop, Park, Tick: the loop, its park, and keeping the first error."""
    _in_order(
        _run_watched(),
        "pool.in_place_scope(|scope| {",
        "while !finished.load(Ordering::Acquire) {",
        "std::thread::park_timeout(next - now);",
        "continue;",
        "if kept.is_some() { continue; }",
        "py.check_signals()?;",
        "callback.bind(py).call1(",
        "if let Err(err) = observed { progress.cancel(); kept = Some(err); }",
    )


def test_join_raises_kept_before_the_slot():
    """Join: after the scope, a kept exception wins over the slot's outcome."""
    _in_order(
        _run_watched(),
        "if let Some(err) = kept { return Err(err); }",
        'slot.expect("the watched job ran")',
    )
