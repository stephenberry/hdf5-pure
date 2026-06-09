# Workspace split design (issue #33 follow-up)

Status: **implemented** (issue #34). The crate is now a Cargo workspace of six internal sub-crates under the `hdf5-pure` facade. This document records the dependency analysis, the crate boundaries, the publish model, the phased migration plan, and (at the end) the friction points discovered while executing it that deviate from the original proposal.

> Naming note: the original proposal named the sub-crates `hdf5-core`, `hdf5-format`, etc. They ship as **`hdf5-pure-core`, `hdf5-pure-format`, `hdf5-pure-filters`, `hdf5-pure-engine`, `hdf5-pure-api`, `hdf5-pure-mat`** — `hdf5-format` was already taken on crates.io, and the `hdf5-pure-*` scheme namespaces all of them consistently as internal to the facade. The crate names in the tables below use the original short forms for readability; read them with the `hdf5-pure-` prefix.

## Motivation

`hdf5-pure` is a single ~40.8k-LoC crate with ~50 top-level modules. Consequences:

- Every edit recompiles the whole crate; there is no incremental boundary.
- Optional functionality is expressed with `#[cfg(feature = "...")]` scattered across modules instead of crate-level separation.
- Nearly every module is `pub mod` at the crate root, so the public API surface is enormous and accidental (issue #33, a dead `pub fn` that shipped in 0.8.0, is a direct symptom).

Splitting into a Cargo workspace addresses the first two structurally and gives us a natural place to tighten the third.

## Hard constraint: read and write are one strongly-connected component

Rust forbids circular dependencies *between crates*. The data engine has genuine mutual recursion that a naive "read crate vs write crate" split cannot honor. The verified edges:

```
chunk_cache      -> chunked_read
chunked_read     -> chunk_cache, extensible_array, fixed_array, filters, ...
extensible_array -> chunked_read, chunked_write          <-- depends on BOTH directions
fixed_array      -> chunked_read
chunked_write    -> chunked_read, chunk_cache, extensible_array, file_writer
file_writer      -> chunked_write, data_read, attribute, type_builders, metadata_index, ...
data_read        -> chunked_read
metadata_index   -> chunked_write, type_builders
type_builders    -> chunked_write, attribute
attribute        -> data_read
```

`extensible_array` depends on both `chunked_read` and `chunked_write`; `file_writer` and `chunked_write` are mutually dependent; `data_read`/`attribute`/`type_builders`/`metadata_index` close further loops (the edges above are an illustrative subset, not the full set). Collapsed, these modules form a single strongly-connected component (SCC) that must live in **one** crate.

The SCC is larger than the short list above: a Tarjan run over the extracted edge set (see "Methodology") puts **14 of the 15 engine modules in one SCC** — `chunk_cache`, `chunked_read`, `chunked_write`, `extensible_array`, `fixed_array`, `data_read`, `file_writer`, `metadata_index`, `type_builders`, `attribute`, `group_v1`, `group_v2`, `provenance`, and `parallel_read`. They are all mutually reachable through the core loop (for example `file_writer -> group_v2 -> data_read -> chunked_read -> extensible_array -> chunked_write -> file_writer`, and `file_writer <-> provenance` is a direct two-cycle). The **only** engine module outside the SCC is `lane_partition`: it has zero outgoing `crate::` edges, so it is a pure leaf, not part of the cycle. (`lane_partition` could therefore be pushed down into a lower crate, but it is only used by the engine's parallel path and gains nothing from moving, so it stays.) This makes the central finding stronger, not weaker: the engine is essentially irreducible.

That central finding: **you cannot get separate read and write crates without first inverting some of these dependencies through traits.** That refactor is possible (see "Breaking the SCC" below) but is a larger, behavior-touching change and is out of scope for the first split.

Everything *below* the SCC (primitives, format structures, filters) and *above* it (the thin high-level API, the MAT subsystem) separates cleanly.

### Methodology

The dependency graph in this document was derived mechanically: extract every `use crate::<module>` / `crate::<module>::` reference from each source file (matching module names that include digits, e.g. `group_v1`), collapse the `mat/**` subtree to a single node, build the directed module graph, and run Tarjan's algorithm for the SCCs plus a layer-assignment check for upward edges. It is reproducible from the current tree, not hand-curated.

## Proposed crates

Six library crates plus a facade. Validated acyclic: every cross-crate `use crate::` edge points to the same or a lower layer (0 upward edges).

| Crate          | LoC    | Depends on                  | Contents |
|----------------|--------|-----------------------------|----------|
| `hdf5-core`    | 1,919  | (none)                      | `error`, `convert`, `message_type`, `nosync`, `checksum`, `signature`, `source` |
| `hdf5-format`  | 9,254  | core                        | `dataspace`, `datatype`, `data_layout`, `link_message`, `link_info`, `group_info`, `attribute_info`, `btree_v1`, `btree_v2`, `local_heap`, `global_heap`, `symbol_table`, `fractal_heap`, `shared_message`, `object_header`, `object_header_writer`, `vl_data`, `superblock` |
| `hdf5-filters` | 5,518  | core, format                | `filter_pipeline`, `filters`, `scaleoffset`, `zfp` (`zfp` feature) |
| `hdf5-engine`  | 14,177 | core, format, filters       | the 14-module SCC (`chunk_cache`, `chunked_read`, `chunked_write`, `extensible_array`, `fixed_array`, `parallel_read`, `data_read`, `file_writer`, `metadata_index`, `type_builders`, `attribute`, `group_v1`, `group_v2`, `provenance`) plus the `lane_partition` leaf |
| `hdf5-api`     | 2,642  | engine                      | `reader`, `writer`, `types`, `swmr_writer`, `ndarray_support` (the std/ndarray-gated high-level surface) |
| `hdf5-mat`     | 7,155  | api **and** engine          | `mat/**` (MATLAB v7.3 serde; `serde` feature) — reaches `file_writer`/`type_builders` (engine) directly, not only `reader`/`writer`/`types` (api) |
| `hdf5-pure`    | facade | api, mat                    | re-exports only; owns the feature flags and the published API |

LoC figures are measured from the current tree and will drift as code changes.

### Why the engine stays large

`hdf5-engine` (~14k LoC) is the irreducible SCC plus the format-orchestration around it. It is the one crate that does not subdivide under the no-cycle rule. The high-level API (`hdf5-api`) peels off above it only because nothing in the SCC depends back on `reader`/`writer`/`types`/`swmr_writer`/`ndarray_support` (verified).

## Feature flags

Features stay defined on the facade `hdf5-pure` and forward to the sub-crates:

- `std` -> enables `hdf5-api` (and the std paths in core/source).
- `parallel` -> enables `lane_partition`/`parallel_read` inside `hdf5-engine` (via an `hdf5-engine/parallel` feature).
- `zfp` -> `hdf5-filters/zfp`.
- `deflate` / `fast-deflate` / `checksum` / `fast-checksum` -> core/filters as appropriate.
- `provenance` -> `hdf5-engine/provenance`.
- `ndarray` -> `hdf5-api/ndarray`.
- `serde` -> enables `hdf5-mat`.
- `matio-crosscheck` -> stays a test-only feature on the facade (or on `hdf5-mat`).

Each sub-crate keeps only the `#[cfg]`s relevant to its own contents, which removes most of the cross-cutting `cfg` scatter.

## Publish model: facade over published sub-crates

Decision: **`hdf5-pure` is the only crate users are expected to depend on, but all seven crates must be published to crates.io.** This is a correction to an earlier draft of this section, which claimed only the facade would be published with the sub-crates referenced by path alone. That does not work: **crates.io rejects any published crate that has a path-only dependency.** When you run `cargo publish`, cargo discards the `path` and resolves each dependency by its `version` requirement against the registry, so every sub-crate the facade pulls in must already exist on crates.io. There is no "publish the facade only" option for a real multi-crate split — this is exactly the model gitoxide uses (the `gix-*` crates are all published, with `gix` as the facade).

What this means concretely:

- **All seven crates are published, version-locked, in dependency order.** A release publishes bottom-up: `hdf5-core` → `hdf5-format` → `hdf5-filters` → `hdf5-engine` → `hdf5-api` → `hdf5-mat` → `hdf5-pure`. Each sub-crate dependency is declared as `{ path = "...", version = "=x.y.z" }` (or a compatible range) so local builds use the workspace path and published builds use the registry version. A `[workspace.package]` `version` field plus `version.workspace = true` in each member keeps them in lockstep from a single source of truth.
- **One version number, but N publish steps.** The changelog can stay single (on the facade), but the release process is no longer one `cargo publish` — it is a scripted, ordered, all-or-nothing sequence. `cargo release` or `cargo-workspaces` handles this; doing it by hand is error-prone because a failure partway leaves a partial release on the registry (crates.io publishes are immutable; you can only yank).
- **Each sub-crate now has its own public API on crates.io**, which `cargo-semver-checks` will track per-crate. This makes the "shrink the public surface" follow-up below more urgent, not less: every `pub` item in a sub-crate is a registry-visible commitment unless demoted to `pub(crate)` or the crate is explicitly documented as an internal implementation detail not covered by semver (a `README`/doc note, the convention gitoxide's `gix-*` crates use).

What survives from the original rationale:

- **Users see no change:** `hdf5_pure::...` paths and the crate name stay identical; `cargo add hdf5-pure` still pulls one dependency line (cargo resolves the rest transitively).
- **The facade is still the stable public entry point**, and internal boundaries can still be re-drawn later — but only behind the facade's re-exports, since the sub-crates are now themselves on the registry and a careless change to a sub-crate's `pub` surface is a public break that `cargo-semver-checks` will flag.

Consequence for the facade: because the current crate exposes almost every module as `pub mod`, the facade must re-export all of those paths (`pub use hdf5_engine::chunked_read;` etc.) to keep the public API byte-for-byte compatible. `cargo-semver-checks` (now in CI) will enforce that during the migration.

A subtlety the per-crate manifests must account for: **internal crates have internal consumers, not just the facade.** A module's dependency edges do not all point at the layer directly below it. `hdf5-mat`, for instance, uses `file_writer` and `type_builders` (engine, layer 3) directly, not only the `reader`/`writer`/`types` wrappers in `hdf5-api` (layer 4) — so `hdf5-mat` must depend on both `hdf5-api` and `hdf5-engine` (or `hdf5-api` must re-export those engine items for it). Either way the graph stays acyclic (engine is below mat), but whoever wires `Cargo.toml` cannot assume each crate depends solely on the one beneath it.

### Recommended follow-up (separate, breaking): shrink the public surface

Exposing every internal module is the root cause of issue #33. Once the workspace exists, a *separate* 0.x-minor (breaking) release should demote internal modules from `pub` to `pub(crate)`/crate-private and keep only the curated surface (`File`, `Dataset`, `Group`, `FileBuilder`, `SwmrWriter`, the `type_builders` makers, `ScaleOffset`, `Error`/`FormatError`, `H5Element`). That is intentionally not part of the split itself, so the split can land as a non-breaking internal refactor.

## Phased migration

Each phase compiles and passes the full test suite on its own; land them as separate PRs.

1. **Scaffold the workspace.** Add `[workspace]` to the root manifest, create `crates/hdf5-core` and move the 7 core modules. Repoint `crate::` -> `hdf5_core::` in movers; the main crate depends on it by path. Smallest, safest first move.
2. **`hdf5-format`.** Move the 18 structure modules. Largest single mechanical move; no logic changes.
3. **`hdf5-filters`.** Move the 4 filter modules; wire the `zfp`/`deflate` features through.
4. **`hdf5-engine`.** Move the SCC as one unit (it cannot be subdivided). Wire `parallel`/`provenance`.
5. **`hdf5-api`.** Move the 5 high-level modules; wire `std`/`ndarray`.
6. **`hdf5-mat`.** Move `mat/**`; wire `serde`.
7. **Reduce the facade to re-exports.** `src/lib.rs` becomes `pub use` lines only. Confirm `cargo-semver-checks` reports no public-API change versus the pre-split release.
8. **Wire the multi-crate release.** Add `[workspace.package]` with a single `version`, set `version.workspace = true` on every member, and convert each sub-crate dependency to `{ path = "...", version = "=x.y.z" }`. Adopt a release tool (`cargo release` or `cargo-workspaces`) to publish bottom-up in one ordered, all-or-nothing step (see "Publish model"). This phase is a prerequisite for the *next* published release, not for the split landing on `main`; the path-only workspace compiles and tests fine without it.

Throughout: `cargo fmt --check`, `cargo clippy -D warnings`, the full feature matrix, the no_std build, and the 32-bit jobs must stay green (CI already covers these). The cast-ratchet baseline in CI is measured on the whole crate and may need re-pointing per sub-crate.

## Breaking the SCC (optional, later)

If separate read/write crates are ever wanted, the SCC must be cut with dependency inversion. The two load-bearing back-edges are:

- `extensible_array` / `fixed_array` -> `chunked_write` (the index structures call into the writer). Invert by having the writer pass in a trait the index implements, rather than the index reaching up to the writer.
- `file_writer` <-> `chunked_write`. Define a `ChunkSink`/`DatasetWriter` trait in a lower crate that `file_writer` implements and `chunked_write` consumes.

This is real refactoring with test surface, tracked separately from the mechanical split. The mechanical split above delivers most of the compile-time and separation benefit without it.

## Expected benefits and their limits

- **Separation of concerns / smaller public surface:** the biggest, most durable win. Internals stop being addressable from outside; `#[cfg]` scatter collapses to crate boundaries.
- **Incremental compile:** editing `hdf5-filters` (which holds the 2.9k-LoC `zfp.rs`) or `hdf5-format` no longer forces a full-crate rebuild of unrelated code. The win is bounded by the linear `core -> format -> filters -> engine` chain and by the engine remaining one ~14k-LoC unit; `hdf5-mat` and `hdf5-api` are the most independently rebuildable.
- **Testing:** each layer gets its own test target and can be exercised in isolation.

The split is feasible and low-risk as a mechanical refactor under the published-facade model (all sub-crates published, `hdf5-pure` as the curated entry point). It does not, by itself, shrink the engine or break the read/write coupling; those are separate, deliberately-scoped follow-ups.

## Implementation notes (deviations discovered during execution)

The mechanical plan held, but four practical issues surfaced that the proposal did not anticipate. They are recorded here because they shape the workspace as built.

1. **The facade re-exports use `#[doc(inline)]`.** Each `pub use hdf5_pure_engine::...;` in `src/lib.rs` carries `#[doc(inline)]` so rustdoc renders the inlined items under `hdf5_pure::*` (matching the pre-split docs). The compile-level path compatibility (`hdf5_pure::chunked_write::...` etc.) holds either way; the attribute is for documentation fidelity.

2. **`cargo-semver-checks` cannot verify the facade.** rustdoc omits cross-crate re-exported item *bodies* from its JSON output, so `cargo-semver-checks` (0.48, the latest) reports every re-exported item under `hdf5_pure::*` as "removed" — a false positive. `#[doc(inline)]` does not change this. The CI `semver-checks` job was therefore re-pointed to check **each sub-crate** (where the real definitions live, introspectable, with `--all-features`) and to **skip the facade**, whose compatibility is instead guaranteed structurally (it is pure re-exports) and exercised by the test suite. Sub-crates not yet on crates.io are skipped until their first publish establishes a baseline. This supersedes the proposal's claim that semver-checks would "enforce that during the migration."

3. **The `ndarray` integration required dependency inversion.** `DatasetBuilder::with_ndarray` is an inherent method on a type that lives in `hdf5-pure-engine`, so the inherent impl had to move into the engine (Rust forbids inherent impls on a foreign type). But the public `H5Element` bound also drives `Dataset::read_array`, and `Dataset` lives one layer up in `hdf5-pure-api`. To keep `H5Element` a single sealed trait without forming a cycle, the engine defines a `ScalarSource` trait that `hdf5-pure-api`'s `Dataset` implements; `H5Element::read_from` dispatches through it. The public surface (`H5Element`, `with_ndarray`, `read_array`/`read_array_dyn`) is unchanged.

4. **A few engine internals were promoted from `pub(crate)` to `pub`** (marked `#[doc(hidden)]`) because the SWMR append writer and the MAT builder, which now live in higher crates, reach into them: `chunked_write::{ea_compute_stats, build_aesb, write_ea_addr, EaStats}`, `extensible_array::EaGeometry`, and `type_builders::GroupBuilder::new`. These are additive (non-breaking) and hidden from docs; the "shrink the public surface" follow-up should re-evaluate them.

### Per-crate testing wiring

Two ergonomic adjustments let each layer be tested in isolation (`cargo test --workspace`):

- The `no_std`-capable sub-crates (`core`/`format`/`filters`/`engine`) carry a `default = ["std", ...]` so a bare `cargo test -p <crate>` gets a std test harness; the facade sets `default-features = false` on its path deps and drives features explicitly, so this does not leak into `hdf5-pure`'s `no_std` builds.
- Doc examples in the sub-crates are written from the end user's perspective (`use hdf5_pure::...`). Each such sub-crate declares `hdf5-pure` as a **path-only** (no version) dev-dependency; cargo strips path-only dev-deps on publish, so the bottom-up publish order is preserved while the doctests resolve the facade locally.
- The 32-bit `check-32bit` and `cast-ratchet-32bit` CI jobs were re-pointed from `--lib` (the now-empty facade) to `--workspace` so they lint the cast-bearing sub-crate code; the cast baseline was re-measured (269).
