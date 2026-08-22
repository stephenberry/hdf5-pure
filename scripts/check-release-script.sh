#!/usr/bin/env bash
#
# Check that scripts/release.sh stops when the public-API delta went unchecked.
#
# Why this exists
# ---------------
# `release.sh` runs `cargo-semver-checks` once per cycle, and that run is the
# only full public-API check there is: CI's job derives what to check from the
# manifest, so once a breaking PR has bumped the version it reports "no semver
# update required" and skips every check for the rest of the cycle. Reading an
# empty report as "nothing broke" therefore takes for granted that the report
# happened at all, which for a while it did not — a `|| true` swallowed a tool
# that died before parsing, and the release proceeded having checked nothing
# (issue #337).
#
# The fix cannot key on the exit status, because what the status means changed
# under the script. Measured here:
#
#   cargo-semver-checks 0.48.0   findings -> 1     could not run -> 1
#   cargo-semver-checks 0.50.0   findings -> 100   could not run -> 101
#
# So `release.sh` takes its verdict from the `Summary` line both versions print
# once they reach one. That is a claim about a third-party tool's output, held
# by a script that runs a few times a year, which is exactly the kind of thing
# that rots unnoticed. Hence this.
#
# Why it is not a CI job
# ----------------------
# It exercises a maintainer script rather than the crate, and the crate's CI
# matrix is already the long pole. It costs about a second and needs no network
# and no build, though, so there is no reason not to run it whenever release.sh
# changes:
#
#   ./scripts/check-release-script.sh
#
# What it does: clones this repo into a scratch directory, puts a stub `cargo`
# on PATH that simulates each cargo-semver-checks outcome, and runs release.sh
# against them. Nothing touches the real repository, the network, or crates.io.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); printf '\033[32mok\033[0m   %s\n' "$*"; }
bad()  { FAIL=$((FAIL + 1)); printf '\033[31mFAIL\033[0m %s\n' "$*"; }
note() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# A stub `cargo`, and the same file as `cargo-semver-checks` so that
# release.sh's `command -v` finds it. SEMVER_MODE picks the outcome.
# ---------------------------------------------------------------------------
mkdir -p "$WORK/bin" "$WORK/bin-nosemver"
cat > "$WORK/bin/cargo" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  semver-checks)
    printf '    Building hdf5-pure (current)\n' >&2
    case "${SEMVER_MODE:?SEMVER_MODE unset}" in
      clean)
        printf '     Checked [   1.234s] 196 checks: 196 pass, 58 skip\n' >&2
        printf '     Summary no semver update required\n' >&2
        exit 0 ;;
      # A verdict was reached and it names breaks. Expected under a 0.x minor
      # bump, so the release must go on. Both exit-code contracts:
      breaks-048)
        printf '     Summary semver requires new minor version: 0 major and 1 minor checks failed\n' >&2
        exit 1 ;;
      breaks-050)
        printf '     Summary semver requires new minor version: 0 major and 1 minor checks failed\n' >&2
        exit 100 ;;
      # No verdict: the run died before checking anything. Both contracts, and
      # note that `cannot-run-048` is indistinguishable from `breaks-048` by
      # exit status alone -- the whole reason the verdict is read from output.
      cannot-run-048)
        printf 'error: unsupported rustdoc format v60 for file: target/doc/hdf5_pure.json\n' >&2
        exit 1 ;;
      cannot-run-050)
        printf 'error: failed to retrieve crate data from registry\n' >&2
        exit 101 ;;
      *) printf 'stub: unknown SEMVER_MODE %s\n' "$SEMVER_MODE" >&2; exit 127 ;;
    esac ;;
  publish) exit 0 ;;
  *) printf 'stub: unexpected cargo invocation: %s\n' "$*" >&2; exit 127 ;;
esac
STUB
cp "$WORK/bin/cargo" "$WORK/bin/cargo-semver-checks"
cp "$WORK/bin/cargo" "$WORK/bin-nosemver/cargo"   # cargo, but no cargo-semver-checks
chmod +x "$WORK/bin/cargo" "$WORK/bin/cargo-semver-checks" "$WORK/bin-nosemver/cargo"

# ---------------------------------------------------------------------------
# A scratch clone to release from, and a version that comes after its last tag.
# Derived rather than written down, so this does not go stale at the next
# release.
# ---------------------------------------------------------------------------
note "Cloning into $WORK/repo"
git clone -q "$REPO_ROOT" "$WORK/repo"
# release.sh refuses to release from anywhere but main, and the clone inherits
# whatever branch this checkout is on.
git -C "$WORK/repo" checkout -q main
LAST_TAG="$(git -C "$WORK/repo" tag --list 'v*' \
  | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed 's/^v//' | sort -V | tail -1)"
[ -n "$LAST_TAG" ] || { printf 'no vX.Y.Z tag in the clone; run `git fetch --tags`\n' >&2; exit 1; }
NEXT="$(printf '%s' "$LAST_TAG" | awk -F. '{printf "%d.%d.0", $1, $2 + 1}')"
note "Releasing $LAST_TAG -> $NEXT in the clone"

# The clone tracks the working tree's release.sh, not HEAD's, so an uncommitted
# change to it is what gets checked. It has to be committed, because release.sh
# refuses to release from a dirty tree — and it has to survive being identical to
# HEAD, which is the case whenever this runs on a clean checkout: `git commit`
# exits non-zero with nothing staged, and under `set -e` that ends the run before
# a single check.
cp "$REPO_ROOT/scripts/release.sh" "$WORK/repo/scripts/release.sh"
if ! git -C "$WORK/repo" diff --quiet; then
  git -C "$WORK/repo" commit -qam "the release.sh under test"
fi
BASELINE="$(git -C "$WORK/repo" rev-parse HEAD)"

# Back to the committed baseline, tag included: a case that runs --commit leaves
# `v$NEXT` behind, and `reset --hard` does not remove tags. Left in place it
# becomes the *last* release, and every later case fails on "does not come after".
reset_clone() {
  git -C "$WORK/repo" reset -q --hard "$BASELINE"
  git -C "$WORK/repo" clean -qfd
  git -C "$WORK/repo" tag -d "v$NEXT" >/dev/null 2>&1 || true
}

# Run release.sh in the clone and report what happened. Every case resets the
# clone first, so a case that mutates the tree cannot colour the next one.
#
# Every case names the text it expects to see, because release.sh has a dozen
# other reasons to stop and each of them would satisfy "it stopped". Left
# unpinned, all three stop cases below pass on a clone still sitting on the
# wrong branch, having never reached the guard under test.
#
#   $1 label   $2 expected outcome: "proceeds" | "stops"   $3 expected text
#   $4 PATH    $5 SEMVER_MODE      rest: extra release.sh arguments
check() {
  local label="$1" expect="$2" want="$3" path="$4" mode="$5"; shift 5
  reset_clone

  local out status dirty
  set +e
  out="$(cd "$WORK/repo" && env PATH="$path" SEMVER_MODE="$mode" \
    scripts/release.sh "$NEXT" "$@" 2>&1)"
  status=$?
  set -e
  dirty="$(git -C "$WORK/repo" status --porcelain | wc -l | tr -d ' ')"
  LAST_OUT="$out"

  if [ "$expect" = "proceeds" ] && [ "$status" -ne 0 ]; then
    bad "$label: expected the release to proceed, exited $status"
  elif [ "$expect" = "stops" ] && [ "$status" -eq 0 ]; then
    bad "$label: expected the release to stop, it proceeded"
  # A stop has to leave the tree clean as well as fail, or re-running after the
  # fix would hit "CHANGELOG already has a [X.Y.Z] section". That property is
  # why the report runs before the bump and the promotion.
  elif [ "$expect" = "stops" ] && [ "$dirty" -ne 0 ]; then
    bad "$label: stopped, but left $dirty modified file(s) behind"
  else
    case "$out" in
      *"$want"*) ok "$label"; return ;;
      *) bad "$label: right outcome, but nothing said \"$want\"" ;;
    esac
  fi
  printf '%s\n' "$out" | tail -5
}

note "Checking release.sh"
check "a clean verdict releases" \
  proceeds "no semver update required" "$WORK/bin:$PATH" clean
check "findings release (0.48.0 exits 1)" \
  proceeds "semver requires new minor version" "$WORK/bin:$PATH" breaks-048
check "findings release (0.50.0 exits 100)" \
  proceeds "semver requires new minor version" "$WORK/bin:$PATH" breaks-050
check "no verdict stops (0.48.0 exits 1)" \
  stops "printed no verdict" "$WORK/bin:$PATH" cannot-run-048
check "no verdict stops (0.50.0 exits 101)" \
  stops "printed no verdict" "$WORK/bin:$PATH" cannot-run-050
check "a missing cargo-semver-checks stops" \
  stops "is not installed" "$WORK/bin-nosemver:/usr/bin:/bin" clean
check "--skip-api-delta releases anyway" \
  proceeds "Skipping the public-API delta report" "$WORK/bin:$PATH" cannot-run-050 --skip-api-delta

# The skip is announced where it happens, which the case above pinned. It also
# has to be repeated near the end, because from there the release runs a build
# and a package check and the first warning scrolls away — the silence this
# script exists to prevent, with a flag in front of it.
#
# Each reminder is matched by its own wording rather than by the flag name. Both
# say "--skip-api-delta", and the prepared-tree run is short enough that a window
# on its tail reaches back to the first warning: matched loosely, deleting the
# reminder at the end of a run passes.
tail_says() {
  case "$(printf '%s\n' "$LAST_OUT" | tail -12)" in
    *"$2"*) ok "$1" ;;
    *) bad "$1: nothing near the end said \"$2\""; printf '%s\n' "$LAST_OUT" | tail -6 ;;
  esac
}
tail_says "the prepared-tree run ends by repeating the skip" \
  "This release's public-API delta was never checked"

# And again on the path that commits and tags, which is a different exit with a
# different reminder. Without a case that gets here, deleting that one passes
# every check above.
check "--skip-api-delta with --commit releases anyway" \
  proceeds "Done:" "$WORK/bin:$PATH" cannot-run-050 \
  --skip-api-delta --commit --summary "A summary, so --commit does not refuse the TODO."
tail_says "the commit-and-tag run ends by repeating the skip" \
  "went out without its public-API delta being checked"

# The sibling guard, which the delta report shares a block with: a feature named
# here that the manifest no longer defines must stop the release rather than
# reach cargo as an empty report that reads like a clean one. Renamed inside
# [features] only -- `ndarray` is a dependency by the same name further down.
reset_clone
awk '/^\[features\]/{f=1} /^\[dependencies\]/{f=0} f{sub(/^ndarray = /, "ndarray-renamed = ")} {print}' \
  "$WORK/repo/Cargo.toml" > "$WORK/toml" && cat "$WORK/toml" > "$WORK/repo/Cargo.toml"
grep -q '^ndarray-renamed = ' "$WORK/repo/Cargo.toml" \
  || { printf 'the drift case did not rename anything; has [features] changed?\n' >&2; exit 1; }
git -C "$WORK/repo" commit -qam "rename a feature release.sh names"
set +e
DRIFT_OUT="$(cd "$WORK/repo" && env PATH="$WORK/bin:$PATH" SEMVER_MODE=clean \
  scripts/release.sh "$NEXT" 2>&1)"
DRIFT_STATUS=$?
set -e
case "$DRIFT_STATUS:$DRIFT_OUT" in
  0:*) bad "a feature release.sh names but the manifest does not released anyway" ;;
  *SEMVER_FEATURES*) ok "a drifted feature name stops the release" ;;
  *) bad "stopped, but not on the feature drift"; printf '%s\n' "$DRIFT_OUT" | tail -5 ;;
esac

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
