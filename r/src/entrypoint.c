// The routine registration extendr generates, under the name R looks for.
void R_init_pedigreegraph_extendr(void *dll);

void R_init_pedigreegraph(void *dll) {
    R_init_pedigreegraph_extendr(dll);
}
