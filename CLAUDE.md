# Project guidance

## Changelog

Keep `CHANGELOG.md` entries concise and reader-facing. Each entry is one or two sentences: lead with the user-facing capability and the public API name, keep at most one short caveat clause naming what is still refused or limited, and end with the issue/PR link in `([#NN](url))` form. Use a **Breaking:** prefix for breaking changes.

Drop the things that bloat it: bit/byte-level internals and root-cause mechanics, "validated/verified against the reference C library" narration, "byte-for-byte" boilerplate, exhaustive enumerations of every refused case, and development cross-references like "addressed below". Commit `d5a966b` (#61, "Trim the changelog to concise, reader-facing entries") is the canonical example of the before/after; match its `before -> after` for any new entry, and don't let the `[Unreleased]` section regrow into PR-description-length paragraphs.

`docs/reference/changelog.md` is a generated include of `CHANGELOG.md`, so edit only the root file.

## Versioning

The version in `Cargo.toml` is the version being *developed*, not the one last published. **A PR that breaks the public API bumps it to the next minor (`0.24.0` -> `0.25.0`) in that same PR**, alongside its `CHANGELOG.md` entry. If it already reads the next minor, leave it: one bump covers every breaking change in the cycle. Additive-only PRs need no bump, and the release takes care of theirs.

This is what keeps the `SemVer (cargo-semver-checks)` job green. That job compares the branch against the latest crates.io release and fails a breaking change unless the manifest already declares a bump large enough to carry it. Deferring the bump to release time makes every breaking PR red by construction, and a check that is expected to be red stops being read at all, including when it flags an *unintended* break.

`scripts/release.sh` reads the previous version from the latest `v*` tag, so it accepts an already-bumped manifest and skips the bump.
