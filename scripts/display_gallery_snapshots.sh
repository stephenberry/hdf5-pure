#!/usr/bin/env bash
# Render the display gallery before and after the `Display` impls, for review.
#
#   scripts/display_gallery_snapshots.sh [base-commit] [output-directory]
#
# Writes `before.md`, `after.md` and `before-after.md` (the two side by side) to
# the output directory, which defaults to a temporary one it reports at the end.
# `base-commit` defaults to the commit the branch's display work starts from.
#
# The gallery (`src/display_gallery.rs`) compiles against both versions, so the
# base run needs only the file copied into a worktree of that commit and its
# `mod` line appended.
#
# Set `RUSTUP_TOOLCHAIN` to pick the toolchain (the crate needs 1.89 or newer).
set -euo pipefail

base=${1:-96b6739}
out=${2:-$(mktemp -d)}

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

worktree=$(mktemp -d)/base
trap 'git worktree remove --force "$worktree" 2>/dev/null || true' EXIT

mkdir -p "$out"

echo "after:  $(git rev-parse --short HEAD)"
DISPLAY_GALLERY_OUT="$out/after.md" cargo test --lib display_gallery >/dev/null

echo "before: $(git rev-parse --short "$base")"
git worktree add --detach "$worktree" "$base" >/dev/null
cp src/display_gallery.rs "$worktree/src/display_gallery.rs"
cat >>"$worktree/src/lib.rs" <<'EOF'

#[cfg(all(test, feature = "std", feature = "deflate", feature = "checksum"))]
mod display_gallery;
EOF
DISPLAY_GALLERY_OUT="$out/before.md" cargo test --manifest-path "$worktree/Cargo.toml" \
    --lib display_gallery >/dev/null

python3 scripts/display_gallery_join.py "$out/before.md" "$out/after.md" \
    >"$out/before-after.md"

echo
echo "$out/before.md"
echo "$out/after.md"
echo "$out/before-after.md"
