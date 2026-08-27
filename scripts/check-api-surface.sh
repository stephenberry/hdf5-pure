#!/usr/bin/env bash
#
# Fail if a type is reachable through the public API without being nameable.
#
# Needs a nightly toolchain for rustdoc's JSON output, which is why this is a
# command you run rather than a CI job. Takes a couple of seconds.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! rustup toolchain list | grep -q '^nightly'; then
    echo "needs a nightly toolchain: rustup toolchain install nightly" >&2
    exit 2
fi

JSON=target/doc/hdf5_pure.json
rm -f "$JSON"
cargo +nightly rustdoc --all-features --lib -- \
    -Zunstable-options --output-format json >/dev/null
test -f "$JSON" || { echo "rustdoc wrote no JSON at $JSON" >&2; exit 2; }

exec python3 scripts/check-api-surface.py "$JSON"
