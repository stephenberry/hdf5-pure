#!/usr/bin/env bash
#
# Check this crate's HDF5 1.8 output format against an actual HDF5 1.8 library.
#
# Why this exists, and why it is not a CI job
# -------------------------------------------
# MATLAB linked HDF5 1.8.12 before R2021b — a different, older library than
# the 1.10.7 behind its own `h5read`/`h5disp`/`h5info` family. A version 3
# superblock is a 1.10 addition, so a file carrying one reads fine under
# `h5disp` and fails under `load`. That is the whole reason `mat::Options`
# defaults to `LibVer::V18`.
#
# Nothing in the test suite can catch a regression here. The `hdf5-metno`
# dev-dependency builds a current libhdf5, `h5py` links a current libhdf5, and
# every third-party MAT v7.3 reader (mat73, hdf5storage, pymatreader, MAT.jl,
# matio) delegates to whichever libhdf5 *it* links. All of them read a version 3
# superblock without complaint. The failure is one byte below everything they
# parse, so only an old library can see it.
#
# So this builds HDF5 1.8.23 — the last 1.8 release — and points its tools at
# both formats. It takes a couple of minutes and needs network access the first
# time, which is why it is a script you run rather than a gate that runs itself.
# Run it when changing anything about superblock or message versions.
#
#   ./scripts/check-hdf5-18.sh
#
# Measured on 1.8.23 when the 1.8 output format landed in 0.34.0:
#   superblock 3 (through 0.33.0) -> h5dump: "unable to open file", exit 1
#   superblock 2 (0.34.0 default) -> h5dump reads data, groups and attributes
#   h5repack round trip           -> preserves every attribute (it dropped all
#                                    of them before the Attribute Info fix)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO/tmp/hdf5-18-check"   # under the gitignored scratch dir
PREFIX="$WORK/install"
SRC_VERSION="1_8_23"
FIXTURES="$WORK/fixtures"

mkdir -p "$WORK"

# ---------------------------------------------------------------- build 1.8.23

if [ ! -x "$PREFIX/bin/h5dump" ]; then
  if [ ! -d "$WORK/hdf5-hdf5-$SRC_VERSION" ]; then
    echo "==> downloading HDF5 $SRC_VERSION"
    curl -fsSL --max-time 300 -o "$WORK/hdf5.tar.gz" \
      "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-$SRC_VERSION.tar.gz"
    tar xzf "$WORK/hdf5.tar.gz" -C "$WORK"
  fi

  cd "$WORK/hdf5-hdf5-$SRC_VERSION"

  # 1.8.23 predates arm64 macOS, and its bundled config.sub rejects the host
  # outright ("machine `aarch64-apple' not recognized"). Refreshing the two
  # config scripts from upstream is the standard remedy and leaves the build
  # itself untouched.
  if [ ! -f bin/config.sub.orig ]; then
    echo "==> refreshing config.guess / config.sub for this host"
    cp bin/config.sub bin/config.sub.orig
    cp bin/config.guess bin/config.guess.orig
    base='https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f='
    curl -fsSL --max-time 120 -o bin/config.sub "${base}config.sub;hb=HEAD"
    curl -fsSL --max-time 120 -o bin/config.guess "${base}config.guess;hb=HEAD"
    chmod +x bin/config.sub bin/config.guess
  fi

  echo "==> configuring (log: $WORK/configure.log)"
  ./configure --prefix="$PREFIX" \
    --disable-fortran --disable-cxx --disable-hl --disable-shared \
    --enable-tools > "$WORK/configure.log" 2>&1 ||
    { echo "configure failed:"; tail -30 "$WORK/configure.log"; exit 1; }

  echo "==> building (log: $WORK/make.log)"
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
    > "$WORK/make.log" 2>&1 ||
    { echo "make failed:"; tail -40 "$WORK/make.log"; exit 1; }

  make install > "$WORK/install.log" 2>&1 ||
    { echo "install failed:"; tail -20 "$WORK/install.log"; exit 1; }
fi

H5DUMP="$PREFIX/bin/h5dump"
H5REPACK="$PREFIX/bin/h5repack"
echo "==> using $("$H5DUMP" --version)"

# ------------------------------------------------------------------- fixtures

echo "==> writing fixtures"
rm -rf "$FIXTURES"
cd "$REPO"
cargo run --quiet --example libver_fixtures --features serde -- "$FIXTURES"

# --------------------------------------------------------------------- checks

failures=0
check() {  # check <description> <expected: open|refuse> <file>
  local desc="$1" expect="$2" file="$3" out rc
  out="$("$H5DUMP" -n "$file" 2>&1)" && rc=0 || rc=$?
  if [ "$expect" = open ] && [ "$rc" -eq 0 ]; then
    echo "  ok   $desc — 1.8 reads it"
  elif [ "$expect" = refuse ] && [ "$rc" -ne 0 ]; then
    echo "  ok   $desc — 1.8 refuses it (${out##*$'\n'})"
  else
    echo "  FAIL $desc — expected to $expect, exit $rc"
    echo "$out" | sed 's/^/       /'
    failures=$((failures + 1))
  fi
}

echo "==> the 1.10 format must be unreadable by 1.8 (or the bound buys nothing)"
check "mat_v110.mat"  refuse "$FIXTURES/mat_v110.mat"
check "plain_v110.h5" refuse "$FIXTURES/plain_v110.h5"

echo "==> the 1.8 format must be readable by 1.8"
check "mat_v18.mat"  open "$FIXTURES/mat_v18.mat"
check "plain_v18.h5" open "$FIXTURES/plain_v18.h5"

# The attribute-count fix, against the toolchain that lost the attributes: a
# header that declares no attributes makes h5repack copy the object without
# them, silently. Counting `ATTRIBUTE` blocks in the dump is enough to see it.
echo "==> a 1.8 h5repack round trip must preserve every attribute"
for f in mat_v18.mat plain_v18.h5; do
  before=$("$H5DUMP" -A "$FIXTURES/$f" 2>/dev/null | grep -c 'ATTRIBUTE' || true)
  rm -f "$FIXTURES/repacked-$f"
  if ! "$H5REPACK" "$FIXTURES/$f" "$FIXTURES/repacked-$f" > /dev/null 2>&1; then
    echo "  FAIL $f — h5repack itself failed"
    failures=$((failures + 1))
    continue
  fi
  after=$("$H5DUMP" -A "$FIXTURES/repacked-$f" 2>/dev/null | grep -c 'ATTRIBUTE' || true)
  if [ "$before" -gt 0 ] && [ "$before" -eq "$after" ]; then
    echo "  ok   $f — $before attributes in, $after out"
  else
    echo "  FAIL $f — $before attributes in, $after out"
    failures=$((failures + 1))
  fi
done

echo
if [ "$failures" -eq 0 ]; then
  echo "all checks passed against HDF5 1.8.23"
else
  echo "$failures check(s) failed"
  exit 1
fi
