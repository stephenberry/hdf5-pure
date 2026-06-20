# Vendored real-MATLAB `.mat` v7.3 fixtures

The `.mat` files in this directory are **genuine real-MATLAB output** (the
userblock header records `Platform: PCWIN64`), vendored as ground-truth read
fixtures for `tests/mat_opaque_real_read.rs`. They let the MCOS opaque-class
decoder be validated against MATLAB itself rather than only round-tripping
against this crate's own writer.

## Source

Vendored from the **`foreverallama/matio`** project (the `mat-io` Python
library), directory `tests/data/`:

- Repository: <https://github.com/foreverallama/matio>
- Commit: `77a578b6605f90035a4030b4dbce0cbe69e0a3d5`
- Files: `test_string_v73.mat`, `test_time_v73.mat`, `test_tables_v73.mat`,
  `test_corrupted_subsystem.mat`, `test_corrupted_mcos_object_metadata.mat`

The expected decoded values asserted in `tests/mat_opaque_real_read.rs` are
transcribed from that project's MATLAB generator scripts
(`tests/data/generators/*.m`) and its independent pytest oracle
(`tests/test_datetime.py`, `test_duration.py`, `test_categorical.py`,
`test_matstring.py`).

## License

`foreverallama/matio` is distributed under the BSD 3-Clause License. Its
copyright notice is retained below as required.

```
BSD 3-Clause License

Copyright (c) 2025, foreverallama

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
