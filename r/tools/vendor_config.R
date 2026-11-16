# Run by src/Makevars.win from src/ when a source tarball builds on Windows:
# unpacks the vendored crates and writes the cargo config that builds from them
# into .cargo/, the CARGO_HOME Makevars.win sets. R's untar needs no tar program,
# and the vendor path is written in the Windows form cargo reads.
untar("rust/vendor.tar.xz", exdir = "rust")
dir.create(".cargo", showWarnings = FALSE)
vendor <- normalizePath("rust/vendor", winslash = "/", mustWork = TRUE)
config <- readLines("rust/vendor-config.toml")
writeLines(sub('^directory = "vendor"$', sprintf('directory = "%s"', vendor), config),
           ".cargo/config.toml")
