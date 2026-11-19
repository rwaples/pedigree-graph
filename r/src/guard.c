// The watched call's cancel guard (ADR 0017, issue #68).
//
// `.pg_run()` starts a relationship call and watches it from R. If anything
// unwinds that watch (Ctrl-C, a time limit, a callback error), the call's job
// must be cancelled before control leaves `.pg_run()`. An R `on.exit` cannot
// promise that: R evaluates the on.exit expression with eval(), which may
// raise a pending interrupt before the cancel runs, and never runs the rest
// (R 4.5.3 src/main/context.c:139, :155; eval.c:1091).
//
// `pg_run_guarded(start, watch)` runs `start()` and then `watch(handle)` under
// R_ExecWithCleanup. R runs its cleanup natively, as the context's `cend`, on
// a normal return and on every jump through it (context.c:128, :909), and no
// R code runs between `start()` returning and the handle being stored. The
// cleanup cancels the job through the binding's own `.Call` routine; after a
// collect that is a no-op.

#include <R.h>
#include <Rinternals.h>

// extendr's `.Call` entry for `job_cancel` (r/src/rust/src/lib.rs): stop the
// job, wait for it, drop its outcome. It reports failures as data, never by
// a jump.
SEXP wrap__job_cancel(SEXP handle);

struct guarded {
    SEXP start;
    SEXP watch;
    SEXP box; // VECSXP of length 1: the handle, once `start()` has returned
};

static SEXP run(void *data) {
    struct guarded *g = data;
    SEXP start = PROTECT(Rf_lang1(g->start));
    SET_VECTOR_ELT(g->box, 0, Rf_eval(start, R_GlobalEnv));
    SEXP watch = PROTECT(Rf_lang2(g->watch, VECTOR_ELT(g->box, 0)));
    SEXP result = Rf_eval(watch, R_GlobalEnv);
    UNPROTECT(2);
    return result;
}

static void cancel(void *data) {
    SEXP handle = VECTOR_ELT(((struct guarded *) data)->box, 0);
    if (TYPEOF(handle) == EXTPTRSXP) {
        wrap__job_cancel(handle);
    }
}

// .External(pg_run_guarded, start, watch)
SEXP pg_run_guarded(SEXP args) {
    struct guarded g;
    args = CDR(args);
    g.start = CAR(args);
    g.watch = CADR(args);
    // Protected by this frame, so the handle outlives `run` and survives the
    // cleanup R runs before it resets the protect stack on a jump.
    g.box = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP result = R_ExecWithCleanup(run, &g, cancel, &g);
    UNPROTECT(1);
    return result;
}
