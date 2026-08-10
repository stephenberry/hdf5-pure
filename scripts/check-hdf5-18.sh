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
#   committed (H5Tcommit) type    -> listed as a named datatype, and the dataset
#                                    and attribute typed through it both read
#
# And on 1.8.23 when the cell fixture was added in 0.35.0:
#   cell array                    -> 1.8 follows both object references to their
#                                    `#refs#` targets and reads each one's data
#   H5PATH on an interned object  -> reads back as the object's own path
#   empty value                   -> its dimension payload reads as `0 0`
#
# The cell is what puts object references and a `#refs#` group in front of an
# old library at all; before it the fixture was scalars, vectors, a struct and
# an empty, and nothing here could have seen a regression in the shape that
# every MATLAB cell array depends on.
#
# One thing that measurement showed about this file's shape: 1.8's h5dump gives
# up on the *whole file* when a committed datatype does not decode, so the plain
# checks above fail alongside the committed ones. The committed fixture is what
# makes any of them sensitive to it — before it there was no named type in the
# file to get wrong — and the committed checks say which thing broke rather than
# leaving "the file will not open" to be interpreted. The exception is a type
# that is written and referenced but never linked: the file then reads correctly
# end to end and 1.8 lists the type as `/#324`, its address standing in for the
# name it lost. Only `check_named_type` sees that one.
#
# The checks below compare dataset and attribute *values*, not just exit status.
# A file whose headers all decode while its data resolves to the wrong offset
# opens cleanly and lists every object, so `h5dump -n` alone would report it as
# passing — which is the defect class this format work fixed on the read side.
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
    echo "  ok   $desc — 1.8 lists its objects"
  elif [ "$expect" = refuse ] && [ "$rc" -ne 0 ]; then
    echo "  ok   $desc — 1.8 refuses it (${out##*$'\n'})"
  else
    echo "  FAIL $desc — expected to $expect, exit $rc"
    echo "$out" | sed 's/^/       /'
    failures=$((failures + 1))
  fi
}

# The values in h5dump's first `DATA` block, as one space-separated line.
#
# Only data lines carry an `(index):` prefix, and taking just the first block
# drops the dataset's own attributes, which `h5dump -d` prints after it. Done
# this way rather than with `-A 0` because that spelling is a 1.10 addition:
# 1.8's `-A` takes no argument, so passing one there dumps the wrong thing.
#
# A failing h5dump yields no values rather than killing the run: under `set -e`
# and `pipefail` its exit status would abort the script at the first bad file,
# and every check after it would go unanswered. This is run by hand and rarely,
# so one run has to report every check, not just the ones before the first
# failure.
dump_values() {  # dump_values <h5dump args...>
  { "$H5DUMP" "$@" 2>/dev/null || true; } |
    awk '
      /^[[:space:]]*DATA[[:space:]]*[{]/ { blocks++; if (blocks > 1) exit; next }
      blocks == 1 && /^[[:space:]]*[(][0-9,]+[)]:/ {
        sub(/^[[:space:]]*[(][0-9,]*[)]:[[:space:]]*/, ""); print
      }' |
    tr ',' ' ' | tr -s '[:space:]' ' ' |
    sed -e 's/^ //' -e 's/ $//'
}

# Compare what 1.8 *read* against what the fixture holds.
#
# `h5dump -n` lists object names and nothing else, so on its own it cannot see a
# file whose headers all decode while its data resolves to the wrong offset —
# the base-address defect class, among others. Every check below therefore names
# the values it expects rather than settling for exit 0.
check_data() {  # check_data <description> <expected values> <h5dump args...>
  local desc="$1" want="$2"; shift 2
  local got
  got="$(dump_values "$@")"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — read [$got]"
  else
    echo "  FAIL $desc — read [$got], expected [$want]"
    failures=$((failures + 1))
  fi
}

# `h5dump -n` lists each object with its kind first, so a committed datatype
# appears as `datatype /reading_t`. It is listed as one only if the object header
# decoded as a named type — which `H5O__dtype_isa` decides from the messages
# present — so this sees a header that opens but is not what it claims.
check_named_type() {  # check_named_type <description> <name> <file>
  local desc="$1" name="$2" file="$3" out
  # As in `dump_values`: a failing h5dump must fail this check, not the run.
  out="$("$H5DUMP" -n "$file" 2>&1)" || true
  if printf '%s\n' "$out" | grep -qE "datatype[[:space:]]+$name\$"; then
    echo "  ok   $desc — 1.8 lists $name as a named datatype"
  else
    echo "  FAIL $desc — $name is not listed as a named datatype"
    printf '%s\n' "$out" | sed 's/^/       /'
    failures=$((failures + 1))
  fi
}

# An object-reference dataset is the one shape whose data is a pointer. 1.8's
# h5dump follows each reference and prints the target as `DATASET <addr> "<path>"`
# before its values, so collecting those paths in order shows the references
# *resolving* — a dataset that opens with references that dangle would still
# dump, with no target lines.
check_ref_targets() {  # check_ref_targets <description> <expected paths> <file> <dataset>
  local desc="$1" want="$2" file="$3" ds="$4" got
  got="$({ "$H5DUMP" -d "$ds" "$file" 2>/dev/null || true; } |
    sed -n 's/^[[:space:]]*DATASET [0-9][0-9]* "\(.*\)".*$/\1/p' |
    tr '\n' ' ' | sed -e 's/ $//')"
  if [ "$got" = "$want" ]; then
    echo "  ok   $desc — resolved [$got]"
  else
    echo "  FAIL $desc — resolved [$got], expected [$want]"
    failures=$((failures + 1))
  fi
}

echo "==> the 1.10 format must be unreadable by 1.8 (or the bound buys nothing)"
check "mat_v110.mat"  refuse "$FIXTURES/mat_v110.mat"
check "plain_v110.h5" refuse "$FIXTURES/plain_v110.h5"

echo "==> the 1.8 format must be readable by 1.8"
check "mat_v18.mat"  open "$FIXTURES/mat_v18.mat"
check "plain_v18.h5" open "$FIXTURES/plain_v18.h5"

# Both fixtures are written by `examples/libver_fixtures.rs`; these are the
# values it puts in them. Reading the *data* is the half `-n` cannot do.
echo "==> 1.8 must read the right bytes, not merely open the file"
check_data "plain_v18.h5 /values"     "1 2 3" -d /values     "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /grp/inner"  "7 8"   -d /grp/inner  "$FIXTURES/plain_v18.h5"
check_data "mat_v18.mat /values"      "1 2 3" -d /values     "$FIXTURES/mat_v18.mat"
check_data "mat_v18.mat /nested/count" "7"    -d /nested/count "$FIXTURES/mat_v18.mat"
# An empty value's payload *is* its dimension vector, so this reads the `0x0`
# rule itself rather than a dataset that happens to be empty.
check_data "mat_v18.mat /empty"       "0 0"   -d /empty       "$FIXTURES/mat_v18.mat"

# A cell array is the only shape that interns objects under `#refs#`: the parent
# dataset holds object references instead of data, and each interned object
# carries an `H5PATH` attribute alongside its `MATLAB_class`. Neither had ever
# been put in front of an old library — the fixture had no cell in it — so a
# regression in either was invisible here.
echo "==> 1.8 must resolve a cell array's interned objects"
check_ref_targets "mat_v18.mat /ragged references" \
  "/#refs#/ref_0000000000000000 /#refs#/ref_0000000000000001" \
  "$FIXTURES/mat_v18.mat" /ragged
for ref in 0 1; do
  name="ref_$(printf '%016x' "$ref")"
  check_data "mat_v18.mat /#refs#/$name" \
    "$([ "$ref" = 0 ] && echo "1" || echo "2 3")" \
    -d "/#refs#/$name" "$FIXTURES/mat_v18.mat"
  check_data "mat_v18.mat /#refs#/$name H5PATH" "\"/#refs#/$name\"" \
    -a "/#refs#/$name/H5PATH" "$FIXTURES/mat_v18.mat"
done

# And the attribute values, not just the count the repack loop below compares.
echo "==> 1.8 must read attribute values on all three kinds of object"
check_data "plain_v18.h5 /values units" '"m/s"' -a /values/units "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 / root_attr"   '"r"'   -a /root_attr    "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /grp tag"      "7"     -a /grp/tag      "$FIXTURES/plain_v18.h5"

# A committed (`H5Tcommit`) datatype is an object of its own, and everything
# using it stores a reference to that object's header in place of an encoding.
# An old library that cannot follow the reference does not fail loudly: it reads
# the reference bytes as a datatype and reports the wrong element type, or drops
# the attribute list the reference hangs off. So name the object, then read
# through it — the dataset whose element type it is, and the attribute whose
# datatype it is.
echo "==> 1.8 must resolve a committed datatype, not merely list it"
check_named_type "plain_v18.h5" /reading_t "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /typed"          "3 1 4" -d /typed          "$FIXTURES/plain_v18.h5"
check_data "plain_v18.h5 /typed baseline" "9"     -a /typed/baseline "$FIXTURES/plain_v18.h5"

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
