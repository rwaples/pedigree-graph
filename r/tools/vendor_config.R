# Run by src/Makevars.win from src/ when a source tarball builds on Windows:
# unpacks the vendored crates and writes the cargo config that builds from them
# into .cargo/, the CARGO_HOME Makevars.win sets. R's internal untar needs no
# tar program, and the vendor path is written in the Windows form cargo reads.
untar("rust/vendor.tar.xz", exdir = "rust", tar = "internal")
dir.create(".cargo", showWarnings = FALSE)
vendor <- normalizePath("rust/vendor", winslash = "/", mustWork = TRUE)
config <- readLines("rust/vendor-config.toml")
line <- '^directory = "vendor"$'
if (sum(grepl(line, config)) != 1) {
  stop("rust/vendor-config.toml has no single `directory = \"vendor\"` line to point at ",
       vendor, call. = FALSE)
}
writeLines(sub(line, sprintf('directory = "%s"', vendor), config), ".cargo/config.toml")
