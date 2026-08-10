% Do the shapes 0.35.0 changed load the way MATLAB means them?
%
% Run:
%   >> cd('matlab_fixtures')
%   >> check_cell_shapes
%
% Run this in **MATLAB**. Octave 11's `load` does not follow the object
% references a v7.3 cell array is built from — it warns "unknown datatype" and
% hands back something that is not a cell — so it cannot answer any question
% here. This script detects that and reports INCONCLUSIVE rather than guessing,
% the same way `check_format.m` does for the format question.
%
% ---------------------------------------------------------------------------
% What is being tested
%
% Three changes in 0.35.0, all of them about a value's shape or its metadata
% rather than its contents, and all measured against MATLAB's own files before
% they were made:
%
%   1. An empty cell array is written `0x0`, MATLAB's own `{}`. Through 0.34.0
%      it was `0x1`. Both are empty, so `isempty` held either way; what changed
%      is `size`. Of the 122 empty cells in the genuine MATLAB files vendored
%      under tests/fixtures/mat_real, every one is `0x0` and none is `0x1`.
%
%   2. A cell array follows the writer's 1-D orientation option, as every other
%      1-D value already did. Through 0.34.0 a cell was a column whatever the
%      option said, so a numeric vector and a cell in the same file could
%      disagree. `cell_shapes.mat` asks for columns and `cell_shapes_rows.mat`
%      for rows; the two hold identical content.
%
%   3. Every object interned under `#refs#` now carries an `H5PATH` attribute
%      holding its own path, which is what MATLAB writes on all but one of its
%      own (the `canonical empty` placeholder in the MCOS subsystem carries
%      none, and neither do we).
%
% The third has no assertion of its own, and cannot have one: `load` does not
% surface an attribute. What it has instead is everything below succeeding —
% the cells here resolve through references that now carry the annotation, the
% `string` value comes back through an MCOS subsystem whose helper objects
% carry it too, and every other fixture in this directory that `verify` reads
% carries it as well. If `H5PATH` broke MATLAB's reference following, nothing
% in this directory would load.
%
% Measured on R2023a Update 1 (9.14.0.2239454), HDF5 1.10.8: CONFIRMED. The
% empty cell loads as `0x0`, both files come back in the orientation asked for
% with the row file the transpose of the column one, `name` arrives as a
% `string` object through the MCOS subsystem, and `struct([])` still reads as
% an empty struct. `verify` passed on the same release with every fixture
% carrying `H5PATH`, which is the whole of the evidence for change 3.
%
% One release is one data point. Run this on whatever MATLAB is to hand and
% extend the record — the same release answers `check_format.m`, and what that
% script measured is the reason the published library-version table cannot be
% read as a list of what `load` accepts.
% ---------------------------------------------------------------------------

fprintf('=== check_cell_shapes ===\n');

is_octave = exist('OCTAVE_VERSION', 'builtin') == 5;
if is_octave
  fprintf('Host: GNU Octave %s\n', OCTAVE_VERSION);
else
  fprintf('Host: MATLAB %s\n', version);
  % Three separate outputs, not a vector. Asking for one and indexing it gets
  % the major version and then an error, which lands in the catch and reports
  % the library as unavailable on a release that would have told you.
  try
    [hdf5_major, hdf5_minor, hdf5_patch] = H5.get_libversion();
    fprintf('HDF5 library: %d.%d.%d\n', hdf5_major, hdf5_minor, hdf5_patch);
  catch
    fprintf('HDF5 library: (H5.get_libversion unavailable)\n');
  end
end

% MATLAB's own answer for the shape of an empty cell, printed rather than
% asserted: it is the thing change 1 above was made to match, and it costs
% nothing to record from the release actually running this.
fprintf('This release writes {} as %dx%d\n', size({}));

% Loaded into a struct rather than the workspace so the loader's own failure is
% inspectable: Octave 11 warns "unknown datatype" and then *omits the variable*
% rather than returning it as something else, so asking `iscell` about it first
% is an error about a missing field, not a verdict. Both shapes of failure —
% missing, and present but not a cell — report INCONCLUSIVE.
s = load('cell_shapes.mat');
if ~isfield(s, 'ragged') || ~iscell(s.ragged)
  fprintf('\nINCONCLUSIVE: this loader did not decode a v7.3 cell array.\n');
  if ~isfield(s, 'ragged')
    fprintf('  `ragged` did not load at all; the loader skipped the variable.\n');
  else
    fprintf('  `ragged` came back as %s, not a cell.\n', class(s.ragged));
  end
  fprintf('  Octave 11 does not follow v7.3 object references; run this in MATLAB.\n');
  return
end

fprintf('\n-- cell_shapes.mat (columns) --\n');
ok(iscell(s.empty_cell), 'empty_cell is a cell');
ok(isempty(s.empty_cell), 'empty_cell is empty');
ok(isequal(size(s.empty_cell), [0 0]), 'empty_cell is 0x0, as MATLAB writes {}');

ok(isequal(size(s.ragged), [2 1]), 'ragged is 2x1, the column the options asked for');
ok(isequal(s.ragged{1}(:), [1; 2; 3]), 'ragged{1} contents');
ok(isequal(s.ragged{2}(:), [4; 5]), 'ragged{2} contents');

ok(iscell(s.records), 'records is a cell');
ok(isequal(size(s.records), [2 1]), 'records is 2x1');
ok(s.records{1}.x == 1.0 && s.records{1}.y == 2.0, 'records{1} fields');
ok(s.records{2}.x == 3.0 && s.records{2}.y == 4.0, 'records{2} fields');

ok(iscell(s.optionals), 'optionals is a cell');
ok(s.optionals{1} == 10.0, 'optionals{1} value');
ok(mat_is_empty_struct(s.optionals{2}), 'optionals{2} is struct([])');

% The `string` class goes through the MCOS subsystem, whose helper objects the
% same release stamps -- all but the canonical empty. A subsystem MATLAB could
% not walk would surface here rather than as a missing attribute.
if exist('isstring', 'builtin') == 5 || exist('isstring', 'file') == 2
  ok(isstring(s.name), 'name came back as a string object');
  ok(s.name == "sensor-1", 'name value');
else
  ok(strcmp(char(s.name), 'sensor-1'), 'name value (no string class on this release)');
end

fprintf('\n-- cell_shapes_rows.mat (rows) --\n');
r = load('cell_shapes_rows.mat');
ok(isequal(size(r.ragged), [1 2]), 'ragged is 1x2, transposed against the column file');
ok(isequal(size(r.records), [1 2]), 'records is 1x2');
ok(isequal(size(r.empty_cell), [0 0]), 'empty_cell stays 0x0 under either orientation');
ok(isequal(r.ragged{1}(:), [1; 2; 3]), 'ragged{1} contents survive the transpose');
ok(r.records{2}.x == 3.0, 'records{2} contents survive the transpose');

fprintf('\nCONFIRMED: MATLAB reads the 0.35.0 shapes as intended.\n');
fprintf('  An empty cell is 0x0, a cell takes the orientation asked for, and\n');
fprintf('  every reference resolved through objects carrying the new H5PATH.\n');
