# Progress and interrupts for the long relationship calls (ADR 0017).
#
# A start routine spawns the engine call on the package pool and returns a
# handle; .pg_watch() then waits on it a tick at a time from R, so progress
# lines and callbacks run as ordinary R code under the caller's handlers,
# and Ctrl-C, time limits and callback errors unwind the loop, whose on.exit
# cancels the call.

# Seconds between two looks at a running call; tests make it small.
.pg_tick <- function() 1
.pg_clock <- function() proc.time()[["elapsed"]]
# Seconds between two default progress lines, and before the first.
.pg_log_interval <- 30

# The reporter `progress` asks for: NULL for none, else function(state).
.pg_progress <- function(progress, label, call = sys.call(-1L)) {
  start <- .pg_clock()
  if (isTRUE(progress)) {
    last <- start
    return(function(state) {
      now <- .pg_clock()
      if (now - last < .pg_log_interval) return(invisible())
      last <<- now
      message(.pg_format_progress(label, state$phase, state$rows_done, state$rows_total, now - start))
    })
  }
  if (isFALSE(progress)) return(NULL)
  if (!is.function(progress)) .pg_usage("`progress` must be TRUE, FALSE or a function", call = call)
  function(state) {
    progress(list(
      phase = state$phase, rows_done = state$rows_done, rows_total = state$rows_total,
      elapsed = .pg_clock() - start
    ))
  }
}

# One progress line in the form of its phase, worded as Python's.
.pg_format_progress <- function(label, phase, rows_done, rows_total, elapsed) {
  after <- .pg_duration(elapsed)
  if (phase == "preparing" || is.na(rows_total)) {
    return(sprintf("%s: preparing after %s", label, after))
  }
  if (phase == "finishing") {
    return(sprintf("%s: all %s rows walked, assembling after %s", label, .pg_thousands(rows_total), after))
  }
  percent <- if (rows_total > 0) (rows_done * 100) %/% rows_total else 100
  sprintf("%s: %s/%s rows (%d%%) after %s", label, .pg_thousands(rows_done),
          .pg_thousands(rows_total), as.integer(percent), after)
}

.pg_thousands <- function(x) formatC(x, format = "f", digits = 0, big.mark = ",")

# "45s", "3m12s" or "2h05m09s", truncated to whole seconds.
.pg_duration <- function(seconds) {
  s <- floor(seconds)
  if (s < 60) return(sprintf("%ds", as.integer(s)))
  if (s < 3600) return(sprintf("%dm%02ds", as.integer(s %/% 60), as.integer(s %% 60)))
  sprintf("%dh%02dm%02ds", as.integer(s %/% 3600), as.integer(s %% 3600 %/% 60), as.integer(s %% 60))
}

# Wait for a started call, reporting each tick, and return its result.
.pg_watch <- function(handle, report, call = sys.call(-1L)) {
  on.exit(.native_cancel(handle))
  repeat {
    state <- .pg_call(.native_wait(handle, .pg_tick()), call = call)
    # Services pending events: R raises a pending Ctrl-C or time limit here,
    # which a loop of .Call()s alone never does.
    Sys.sleep(0)
    if (state$finished) break
    if (!is.null(report)) report(state)
  }
  .pg_call(.native_collect(handle), call = call)
}
