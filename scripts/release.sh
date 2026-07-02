#!/usr/bin/env bash
#
# release.sh — cut a new hdf5-pure release.
#
# Automates the mechanical, easy-to-botch parts of a release so they come out
# identical every time:
#   * bump the version in Cargo.toml and Cargo.lock
#   * promote the CHANGELOG's [Unreleased] section into a dated `## [X.Y.Z]`
#     section and refresh the two compare links at the bottom
#   * verify the crate still packages (`cargo publish --dry-run`)
#   * commit "Release vX.Y.Z" and create the annotated `vX.Y.Z` tag
#
# The one-paragraph editorial summary that leads each release's changelog
# section is yours to write: pass it with --summary / --summary-file, or the
# script inserts a TODO placeholder and reminds you to fill it in before the
# release is published.
#
# Public, irreversible steps are opt-in and never run by default. Without any
# of the flags below the script only touches your working tree and local git,
# then prints the remaining commands:
#   --commit       commit the release and create the tag (implied by the below)
#   --push         push main and the tag to origin      (implies --commit)
#   --gh-release   create the GitHub release from the changelog section
#                                                        (implies --push)
#   --publish      `cargo publish` to crates.io          (implies --commit)
#
# Usage:
#   scripts/release.sh 0.21.0 --summary-file notes.md
#   scripts/release.sh 0.21.0 --summary "One-paragraph summary." --push --gh-release
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
NEW_VERSION=""
SUMMARY=""
SUMMARY_FILE=""
DO_COMMIT=0
DO_PUSH=0
DO_GH=0
DO_PUBLISH=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '\033[1m==>\033[0m %s\n' "$*"; }

usage() {
  sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)     usage 0 ;;
    --summary)     SUMMARY="${2:?--summary needs a value}"; shift 2 ;;
    --summary-file) SUMMARY_FILE="${2:?--summary-file needs a path}"; shift 2 ;;
    --commit)      DO_COMMIT=1; shift ;;
    --push)        DO_PUSH=1; shift ;;
    --gh-release)  DO_GH=1; shift ;;
    --publish)     DO_PUBLISH=1; shift ;;
    -*)            die "unknown option: $1 (see --help)" ;;
    *)
      [ -z "$NEW_VERSION" ] || die "unexpected extra argument: $1"
      NEW_VERSION="$1"; shift ;;
  esac
done

[ -n "$NEW_VERSION" ] || usage 1

# Resolve step dependencies: a later step implies every earlier one.
[ "$DO_GH" -eq 1 ] && DO_PUSH=1
{ [ "$DO_PUSH" -eq 1 ] || [ "$DO_PUBLISH" -eq 1 ]; } && DO_COMMIT=1

# ---------------------------------------------------------------------------
# Locate the repo and load current state
# ---------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CARGO_TOML="Cargo.toml"
CARGO_LOCK="Cargo.lock"
CHANGELOG="CHANGELOG.md"
TAG="v${NEW_VERSION}"

# Extract the [package] version and repository URL from Cargo.toml. The version
# lives on the first `version =` line after `name = "hdf5-pure"`.
CUR_VERSION="$(awk '/^name = "hdf5-pure"/{p=1} p&&/^version = /{gsub(/[",]/,"",$3); print $3; exit}' "$CARGO_TOML")"
REPO_URL="$(awk -F'"' '/^repository = /{print $2; exit}' "$CARGO_TOML")"
[ -n "$CUR_VERSION" ] || die "could not read current version from $CARGO_TOML"
[ -n "$REPO_URL" ] || die "could not read repository URL from $CARGO_TOML"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
note "Releasing ${CUR_VERSION} -> ${NEW_VERSION}"

printf '%s\n' "$NEW_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || die "version must be X.Y.Z (got '$NEW_VERSION')"
[ "$NEW_VERSION" != "$CUR_VERSION" ] || die "version is already $NEW_VERSION"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "not on main (on '$BRANCH'); release from main"

git diff --quiet && git diff --cached --quiet \
  || die "working tree is dirty; commit or stash first"

git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null \
  && die "tag ${TAG} already exists"

# The [Unreleased] section must contain at least one entry to release.
UNRELEASED_BODY="$(awk '
  /^## \[Unreleased\]/{grab=1; next}
  grab && /^## \[/{exit}
  grab && /[^[:space:]]/{print}
' "$CHANGELOG")"
[ -n "$UNRELEASED_BODY" ] || die "CHANGELOG [Unreleased] is empty; nothing to release"

# Resolve the summary paragraph.
if [ -n "$SUMMARY_FILE" ]; then
  [ -f "$SUMMARY_FILE" ] || die "summary file not found: $SUMMARY_FILE"
  SUMMARY="$(cat "$SUMMARY_FILE")"
fi
if [ -z "$SUMMARY" ]; then
  SUMMARY="TODO: one-paragraph summary of ${NEW_VERSION} (see prior releases for the house style)."
  SUMMARY_IS_TODO=1
else
  SUMMARY_IS_TODO=0
fi

TODAY="$(date +%Y-%m-%d)"

# ---------------------------------------------------------------------------
# 1. Bump the version in Cargo.toml and Cargo.lock
# ---------------------------------------------------------------------------
# Rewrite the first `version =` line that follows the hdf5-pure package header.
# The same state machine works for Cargo.toml ([package]) and Cargo.lock
# ([[package]] hdf5-pure).
bump_version() {
  local file="$1" tmp
  tmp="$(mktemp)"
  awk -v new="$NEW_VERSION" '
    /^name = "hdf5-pure"/ { pkg=1 }
    pkg && /^version = / { sub(/"[^"]*"/, "\"" new "\""); pkg=0 }
    { print }
  ' "$file" > "$tmp"
  mv "$tmp" "$file"
}
note "Bumping version in $CARGO_TOML and $CARGO_LOCK"
bump_version "$CARGO_TOML"
bump_version "$CARGO_LOCK"

# ---------------------------------------------------------------------------
# 2. Promote [Unreleased] into a dated section and refresh the compare links
# ---------------------------------------------------------------------------
note "Updating $CHANGELOG"
CL_TMP="$(mktemp)"
SUMMARY="$SUMMARY" awk \
  -v new="$NEW_VERSION" -v prev="$CUR_VERSION" -v date="$TODAY" -v repo="$REPO_URL" '
  # Insert the new version header + summary as the first content under
  # [Unreleased] (i.e. before the first non-blank line that follows it).
  /^## \[Unreleased\]/ { print; seen=1; next }
  seen && !done && /[^[:space:]]/ {
    print "## [" new "] - " date
    print ""
    print ENVIRON["SUMMARY"]
    print ""
    done=1
  }
  # Rewrite the [Unreleased] compare link and add the [X.Y.Z] link beneath it.
  /^\[Unreleased\]:/ {
    print "[Unreleased]: " repo "/compare/v" new "...HEAD"
    print "[" new "]: " repo "/compare/v" prev "...v" new
    next
  }
  { print }
' "$CHANGELOG" > "$CL_TMP"
mv "$CL_TMP" "$CHANGELOG"

# ---------------------------------------------------------------------------
# 3. Verify the crate still packages cleanly with the new version
# ---------------------------------------------------------------------------
note "Verifying with cargo publish --dry-run"
cargo publish --dry-run --allow-dirty

# ---------------------------------------------------------------------------
# 4. Commit + tag (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_COMMIT" -eq 0 ]; then
  note "Prepared release files (not committed)."
  git --no-pager diff --stat -- "$CARGO_TOML" "$CARGO_LOCK" "$CHANGELOG"
  [ "$SUMMARY_IS_TODO" -eq 1 ] && \
    printf '\n\033[33m!\033[0m Fill in the TODO summary in %s before committing.\n' "$CHANGELOG"
  cat <<EOF

Next steps:
  git add $CARGO_TOML $CARGO_LOCK $CHANGELOG
  git commit -m "Release $TAG"
  git tag -a "$TAG" -m "Release $TAG"
  git push origin main && git push origin "$TAG"
  # GitHub release notes = the changelog section for this version
  cargo publish
EOF
  exit 0
fi

if [ "$SUMMARY_IS_TODO" -eq 1 ]; then
  die "refusing to commit with a TODO summary; pass --summary/--summary-file"
fi

note "Committing and tagging $TAG"
git add "$CARGO_TOML" "$CARGO_LOCK" "$CHANGELOG"
git commit -m "Release $TAG"
git tag -a "$TAG" -m "Release $TAG"

# ---------------------------------------------------------------------------
# 5. Push (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_PUSH" -eq 1 ]; then
  note "Pushing main and $TAG"
  git push origin main
  git push origin "$TAG"
fi

# ---------------------------------------------------------------------------
# 6. GitHub release (opt-in) — notes are the changelog section for this version
# ---------------------------------------------------------------------------
if [ "$DO_GH" -eq 1 ]; then
  note "Creating GitHub release $TAG"
  NOTES_TMP="$(mktemp)"
  awk -v ver="$NEW_VERSION" '
    index($0, "## [" ver "]") == 1 { grab=1; next }  # start after this version header
    grab && /^## \[/ { exit }                        # stop at the next version header
    grab { print }
  ' "$CHANGELOG" > "$NOTES_TMP"
  printf '\n**Full changelog:** %s/compare/v%s...v%s\n' "$REPO_URL" "$CUR_VERSION" "$NEW_VERSION" >> "$NOTES_TMP"
  gh release create "$TAG" --title "$TAG" --notes-file "$NOTES_TMP"
  rm -f "$NOTES_TMP"
fi

# ---------------------------------------------------------------------------
# 7. crates.io publish (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_PUBLISH" -eq 1 ]; then
  note "Publishing to crates.io"
  cargo publish
fi

note "Done: $TAG"
