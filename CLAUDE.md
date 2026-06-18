# Project guidance

## Changelog

Keep `CHANGELOG.md` entries concise and reader-facing. Each entry is one or two sentences: lead with the user-facing capability and the public API name, keep at most one short caveat clause naming what is still refused or limited, and end with the issue/PR link in `([#NN](url))` form. Use a **Breaking:** prefix for breaking changes.

Drop the things that bloat it: bit/byte-level internals and root-cause mechanics, "validated/verified against the reference C library" narration, "byte-for-byte" boilerplate, exhaustive enumerations of every refused case, and development cross-references like "addressed below". Commit `d5a966b` (#61, "Trim the changelog to concise, reader-facing entries") is the canonical example of the before/after; match its `before -> after` for any new entry, and don't let the `[Unreleased]` section regrow into PR-description-length paragraphs.

`docs/reference/changelog.md` is a generated include of `CHANGELOG.md`, so edit only the root file.
