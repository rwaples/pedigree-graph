#!/usr/bin/env bash
# R CMD check --as-cran on the R source tarball with the network disabled; fails on an
# ERROR or WARNING. ci.yml and publish.yml both run it, from the tarball's folder.
#
# ADR 0007: the tarball compiles offline. A network namespace drops every interface; the
# check itself runs as the calling user. Both sudos keep the environment, and R_LIBS_USER
# (where setup-r installed testthat and Matrix) is also passed by value: Ubuntu 26.04's
# sudo drops it even under --preserve-env. Unset, it passes empty, and R's Renviron puts
# the default user library back.
#
# Usage: tools/r_check_offline.sh pedigreegraph_X.Y.Z.tar.gz
set -euo pipefail

tarball=$1
sudo --preserve-env unshare --net -- sudo --preserve-env -u "$(id -un)" \
  env R_LIBS_USER="${R_LIBS_USER:-}" _R_CHECK_CRAN_INCOMING_REMOTE_=false \
  "$(command -v R)" CMD check --as-cran --no-manual "$tarball"
if grep -q '^Status:.*\(ERROR\|WARNING\)' pedigreegraph.Rcheck/00check.log; then
  echo "::error::R CMD check reported an ERROR or WARNING"
  exit 1
fi
