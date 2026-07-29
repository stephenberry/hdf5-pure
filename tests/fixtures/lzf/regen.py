#!/usr/bin/env python3
# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "h5py==3.16.0",
#     "numpy==2.4.4",
# ]
# ///
"""Regenerate LZF crosscheck fixtures from h5py's built-in LZF filter.

Produces a `manifest.json` plus, per fixture:
  <name>.raw.bin          raw uncompressed values (little-endian native width)
  <name>.compressed.bin   stored chunk bytes, only for single-chunk cases
                          whose pipeline is exactly [lzf] — the only streams
                          src/lzf_crosscheck.rs compares at the codec level
                          (raw bytes when h5py skipped the optional filter —
                          see `filter_mask` in the manifest)
  <name>.h5               the h5py-written file, read end-to-end by
                          tests/lzf_roundtrip.rs and handy for h5dump

LZF ships with h5py itself (no hdf5plugin needed). h5py registers it as an
*optional* filter: when liblzf cannot shrink a chunk the chunk is stored raw
with its filter-mask bit set — the `u8_noise` case exists to capture exactly
that path, which hdf5-pure's writer (mandatory filters only) can never emit.

Run from the repo root with uv (resolves the pins above automatically):
    uv run tests/fixtures/lzf/regen.py

Or in a venv pinned from the sibling requirements file:
    python -m venv tests/fixtures/lzf/.venv
    tests/fixtures/lzf/.venv/bin/pip install -r tests/fixtures/lzf/requirements.txt
    tests/fixtures/lzf/.venv/bin/python tests/fixtures/lzf/regen.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
MANIFEST = FIXTURE_DIR / "manifest.json"

NP_DTYPE = {
    "u8": np.uint8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f64": np.float64,
}


@dataclass
class Case:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: Any  # numpy array (any dtype; cast on write)
    chunks: tuple[int, ...] | None = None  # None = one chunk covering shape
    shuffle: bool = False
    notes: str = ""

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=NP_DTYPE[self.dtype]).reshape(self.shape)


def xorshift_noise(n: int) -> np.ndarray:
    """Deterministic incompressible bytes.

    Same xorshift constants and seed as the `round_trips` unit test in
    src/lzf.rs; the Rust integration tests consume this fixture's bytes
    directly rather than regenerating them.
    """
    out = np.empty(n, dtype=np.uint8)
    x = 0x2545_F491_4F6C_DD1D
    mask = (1 << 64) - 1
    for i in range(n):
        x = (x ^ (x << 13)) & mask
        x ^= x >> 7
        x = (x ^ (x << 17)) & mask
        out[i] = x & 0xFF
    return out


def build_cases() -> list[Case]:
    return [
        Case(
            name="i32_ramp",
            dtype="i32",
            shape=(1024,),
            data=np.arange(1024),
            notes="compressible monotonic ramp, lzf only",
        ),
        Case(
            name="u8_noise",
            dtype="u8",
            shape=(4096,),
            data=xorshift_noise(4096),
            notes="incompressible; h5py's optional LZF stores the chunk raw "
            "with the filter-mask bit set",
        ),
        Case(
            name="i16_rle",
            dtype="i16",
            shape=(2048,),
            data=np.repeat(np.arange(8), 256),
            notes="long constant runs (overlapping back-references)",
        ),
        Case(
            name="f64_shuffle_lzf",
            dtype="f64",
            shape=(512,),
            data=np.sin(np.arange(512) * 0.05),
            shuffle=True,
            notes="shuffle+lzf chain; end-to-end .h5 coverage only",
        ),
        Case(
            name="i64_multichunk",
            dtype="i64",
            shape=(300,),
            data=np.arange(300) * 7 - 500,
            chunks=(128,),
            notes="multi-chunk with partial edge chunk; end-to-end .h5 only",
        ),
    ]


def write_fixture(case: Case) -> dict[str, Any]:
    arr = case.to_numpy()
    h5_path = FIXTURE_DIR / f"{case.name}.h5"
    chunks = case.chunks or arr.shape

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "v", data=arr, chunks=chunks, compression="lzf", shuffle=case.shuffle
        )

    with h5py.File(h5_path, "r") as f:
        d = f["v"]
        dcpl = d.id.get_create_plist()
        filters = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
        lzf = next(flt for flt in filters if flt[0] == 32000)
        _, lzf_flags, lzf_cd_values, _ = lzf
        # The chunk mask is a measured, per-chunk fact: record it only for
        # single-chunk cases rather than fabricating a whole-dataset value.
        if chunks == arr.shape:
            filter_mask, chunk_bytes = d.id.read_direct_chunk((0,) * arr.ndim)
        else:
            filter_mask, chunk_bytes = None, None

    raw_bytes = arr.tobytes(order="C")
    (FIXTURE_DIR / f"{case.name}.raw.bin").write_bytes(raw_bytes)
    # Only a bare-lzf stream is codec-comparable; chained pipelines are
    # exercised end-to-end through the .h5 instead.
    if chunk_bytes is not None and [int(f[0]) for f in filters] == [32000]:
        (FIXTURE_DIR / f"{case.name}.compressed.bin").write_bytes(chunk_bytes)
    else:
        chunk_bytes = None

    return {
        "name": case.name,
        "dtype": case.dtype,
        "shape": list(case.shape),
        "chunk_shape": list(chunks),
        "filters": [int(flt[0]) for flt in filters],
        "lzf_flags": int(lzf_flags),
        "cd_values_u32": [int(v) & 0xFFFFFFFF for v in lzf_cd_values],
        "filter_mask": None if filter_mask is None else int(filter_mask),
        "raw_bytes_len": len(raw_bytes),
        "compressed_bytes_len": len(chunk_bytes) if chunk_bytes is not None else None,
        "notes": case.notes,
    }


def main() -> None:
    entries = [write_fixture(case) for case in build_cases()]
    manifest = {
        "generator": f"h5py {h5py.__version__}, numpy {np.__version__}",
        "fixtures": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(entries)} fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
