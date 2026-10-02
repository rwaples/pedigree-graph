"""Progress reporting for the long relationship calls (ADR 0017).

The engine never calls Python.  While it runs, the native binding wakes the
calling thread every ``TICK_S`` seconds; that thread checks for signals, so
Ctrl-C cancels the call, and passes the engine's phase and row counts to the
callback :func:`resolve` builds from the public ``progress=`` keyword.  An
exception from the callback cancels the call too and is raised in place of
its result.

A call moves through three phases: ``preparing`` (the ancestry compaction of
a view and engine setup, before the total is known), ``walking`` (row
visits; a ``"memory"`` pair execution visits every row twice) and
``finishing`` (whatever runs after the last row, such as assembling pair
blocks or merging moment lanes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

#: Seconds between two looks at a running call.  Tests make it small.
TICK_S = 1.0
#: Seconds between two default progress log lines, and before the first.
LOG_INTERVAL_S = 30.0

Phase = Literal["preparing", "walking", "finishing"]


@dataclass(frozen=True, slots=True)
class RelationshipProgress:
    """How far a running relationship call has got.

    Attributes:
        phase: ``"preparing"`` before the walk, ``"walking"`` during it, and
            ``"finishing"`` once every row is visited and the result is
            being assembled.
        rows_done: Row visits done; ``0`` while preparing, ``rows_total``
            while finishing.
        rows_total: Row visits in the call, ``None`` while preparing.  A
            view's call walks only the rows its ancestry needs, and a
            ``"memory"`` pair execution walks every row twice.
        elapsed: Seconds since the call started.
    """

    phase: Phase
    rows_done: int
    rows_total: int | None
    elapsed: float


#: The public ``progress=`` keyword: ``None`` to log, ``False`` for nothing,
#: or a callable that receives a :class:`RelationshipProgress`.
type ProgressArg = Callable[[RelationshipProgress], object] | bool | None
#: What the native binding calls every tick: ``(phase, rows_done, rows_total)``.
type NativeCallback = Callable[[Phase, int, int | None], None]


def resolve(progress: ProgressArg, logger: logging.Logger, label: str) -> NativeCallback | None:
    """Turn the public ``progress=`` keyword into the native binding's callback.

    The clock starts here, so ``elapsed`` counts from the call's own start.

    Args:
        progress: ``None`` to log a line through *logger* at INFO every
            ``LOG_INTERVAL_S``, ``False`` for no progress, or a callable that
            receives a :class:`RelationshipProgress` about every ``TICK_S``
            and logs nothing.
        logger: The calling method's module logger.
        label: The method name the lines start with.

    Returns:
        The ``(phase, rows_done, rows_total)`` callback, or ``None`` for no
        progress.

    Raises:
        TypeError: When *progress* is not ``None``, ``False`` or a callable;
            ``True`` included.
    """
    start = time.perf_counter()
    if progress is None:
        last = start

        def log(phase: Phase, done: int, total: int | None) -> None:
            nonlocal last
            now = time.perf_counter()
            if now - last < LOG_INTERVAL_S:
                return
            last = now
            logger.info("%s", format_progress(label, RelationshipProgress(phase, done, total, now - start)))

        return log
    if progress is False:
        return None
    if not callable(progress):
        raise TypeError(f"progress must be None, False or a callable, got {progress!r}")
    observer = progress

    def call(phase: Phase, done: int, total: int | None) -> None:
        observer(RelationshipProgress(phase, done, total, time.perf_counter() - start))

    return call


def format_progress(label: str, p: RelationshipProgress) -> str:
    """One log line for *p*, in the form of its phase.

    >>> format_progress("relationship_counts", RelationshipProgress("walking", 195_758, 783_029, 192.4))
    'relationship_counts: 195,758/783,029 rows (25%) after 3m12s'
    """
    after = _duration(p.elapsed)
    if p.phase == "preparing" or p.rows_total is None:
        return f"{label}: preparing after {after}"
    if p.phase == "finishing":
        return f"{label}: all {p.rows_total:,} rows walked, assembling after {after}"
    percent = p.rows_done * 100 // p.rows_total if p.rows_total else 100
    return f"{label}: {p.rows_done:,}/{p.rows_total:,} rows ({percent}%) after {after}"


def _duration(seconds: float) -> str:
    """``45s``, ``3m12s`` or ``2h05m09s``, truncated to whole seconds."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"
