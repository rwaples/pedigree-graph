#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// The routine registration extendr generates, under the name R looks for.
void R_init_pedigreegraph_extendr(void *dll);

// r/src/guard.c
SEXP pg_run_guarded(SEXP args);

static const R_ExternalMethodDef external_routines[] = {
    {"pg_run_guarded", (DL_FUNC) &pg_run_guarded, 2},
    {NULL, NULL, 0}
};

void R_init_pedigreegraph(void *dll) {
    R_init_pedigreegraph_extendr(dll);
    // Adds the .External table only: R_registerRoutines replaces just the
    // tables it is given (src/main/Rdynload.c), so extendr's .Call routines
    // stay. It resets dynamic lookup, which extendr had turned off.
    R_registerRoutines(dll, NULL, NULL, NULL, external_routines);
    R_useDynamicSymbols(dll, FALSE);
}
