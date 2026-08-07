#!/usr/bin/env bash
#
# Regenerate the HDF5 1.8-era read fixtures in this directory.
#
# Needs the HDF5 1.8.23 install that `scripts/check-hdf5-18.sh` builds; run that
# first if it is not there. The fixtures are committed, so this is a developer
# tool rather than part of any test run — nothing in `cargo test` invokes it.
#
#   ./scripts/check-hdf5-18.sh          # once, builds 1.8.23
#   ./tests/fixtures/c_1_8/regen.sh
#
# Rerun it only to change what the fixtures contain. The committed files are the
# ground truth the tests read; regenerating them for no reason replaces measured
# bytes with freshly measured bytes and reviews as a diff nobody can check.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
H5CC="$REPO/tmp/hdf5-18-check/install/bin/h5cc"

if [ ! -x "$H5CC" ]; then
  echo "no HDF5 1.8.23 install at $H5CC" >&2
  echo "run ./scripts/check-hdf5-18.sh first — it builds one" >&2
  exit 1
fi

echo "==> using $("$REPO/tmp/hdf5-18-check/install/bin/h5dump" --version)"
cd "$HERE"
"$H5CC" -O2 -o regen regen.c
./regen
rm -f regen regen.o

for f in v1_superblock.h5 v2_superblock.h5; do
  ver=$(python3 -c "
import sys
b = open('$f','rb').read()
i = b.find(b'\x89HDF\r\n\x1a\n')
print(b[i+8] if i >= 0 else 'no signature')")
  echo "  $f superblock $ver"
done
