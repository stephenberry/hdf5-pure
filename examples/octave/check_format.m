% Does MATLAB's `load` accept the on-disk format this crate writes?
%
% Run:
%   >> cd('matlab_fixtures')
%   >> check_format
%
% Run this in **MATLAB**, not Octave. Octave links a modern HDF5 that reads
% both formats happily, so it cannot tell them apart and will report both as
% loading. That is exactly the blind spot this script exists to cover.
%
% ---------------------------------------------------------------------------
% What is being tested
%
% Which HDF5 library MATLAB links has changed across releases, and it decides
% whether a file can be opened at all. MathWorks documents it:
%
%   R2021a and earlier  1.8.12     R2023b   1.10.10
%   R2021b              1.10.7     R2024a   1.10.11
%   R2022a - R2023a     1.10.8     R2024b+  1.14.4.3
%
% A version 3 superblock is an HDF5 1.10 addition, so a file carrying one cannot
% be opened by MATLAB before R2021b. Every release of this crate through 0.33.0
% wrote one; 0.34.0 writes the HDF5 1.8 format by default. The two files here
% hold identical content and differ only in that format:
%
%   format_v18.mat    version 2 superblock -- the new default, must load
%   format_v110.mat   version 3 superblock -- what 0.33.0 wrote
%
% What the outcome means depends on the release, which is why this reports the
% library version alongside the result:
%
%   1.8.x   the v1.10 file must fail. If it loads, the diagnosis is wrong.
%   1.10+   the v1.10 file is expected to load. That says nothing against the
%           1.8 default -- it only means this release was never affected.
%
% Either way the v1.8 file must load and decode correctly. That is the claim the
% default rests on, and it holds for every release in the table.
%
% There is one wrinkle the version number will not show. Around R2021b MathWorks
% shipped two libraries at once, keeping 1.8.12 on the MAT v7.3 path while
% `h5read`/`h5disp` used 1.10.7. On such a release `H5.get_libversion` reports
% 1.10.7 while `load` still cannot open a v1.10 file -- so a 1.10 report with a
% failing control is a meaningful result, not a contradiction.
% ---------------------------------------------------------------------------

fprintf('\n');
is_octave = exist('OCTAVE_VERSION', 'builtin') == 5;
if is_octave
    fprintf('!! Running under Octave %s, which CANNOT answer the question\n', ...
        OCTAVE_VERSION);
    fprintf('!! this script exists for. Octave links a modern HDF5 and reads\n');
    fprintf('!! both formats, so it will report both files as loading no matter\n');
    fprintf('!! what. Run this in MATLAB.\n');
    fprintf('\n');
else
    fprintf('MATLAB version: %s\n', version);
end
hdf5_major = NaN;
try
    [hdf5_major, hdf5_minor, hdf5_patch] = H5.get_libversion();
    fprintf('HDF5 library:   %d.%d.%d\n', hdf5_major, hdf5_minor, hdf5_patch);
catch
    fprintf('HDF5 library:   <H5.get_libversion unavailable>\n');
end
% Before R2021b MATLAB was wholly on 1.8.12, where a version 3 superblock cannot
% be opened. From 1.10 on it can, so the control file loading is expected there.
%
% `hdf5_known` is tracked separately from `expects_v110_to_fail` because "not
% expected to fail" and "we never found out" are different answers, and only one
% of them licenses a claim about which library this release links. Without it the
% verdict below reported an unknown-version run as "expected on HDF5 1.10 or
% newer" -- and, worse, reported the genuinely anomalous 1.8 outcome as expected.
hdf5_known = ~isnan(hdf5_major);
expects_v110_to_fail = hdf5_known && hdf5_major == 1 && hdf5_minor < 10;
fprintf('\n');

% --- the file that must load ------------------------------------------------
fprintf('--- format_v18.mat (version 2 superblock, the 0.34.0 default) ---\n');
v18_loaded = false;
try
    s18 = load('format_v18.mat');
    v18_loaded = true;
    fprintf('  LOADED\n');
catch err
    fprintf('  FAILED TO LOAD: %s\n', err.message);
end

if v18_loaded
    fprintf('  contents:\n');
    fprintf('    values      %s\n', mat2str(double(s18.values(:))'));
    fprintf('    label       "%s"\n', char(s18.label));
    fprintf('    count       %s\n', mat2str(double(s18.count)));
    fprintf('    flag        %s\n', mat2str(double(s18.flag)));
    fprintf('    empty       size %s, isempty=%d\n', ...
        mat2str(mat_empty_dims(s18.empty)), mat_isempty(s18.empty));
    fprintf('    nested.a    %s\n', mat2str(double(s18.nested.a)));

    good = isequal(double(s18.values(:)), [1;2;3]) ...
        && isequal(double(s18.count), 7) ...
        && double(s18.flag) == 1 ...
        && mat_isempty(s18.empty) ...
        && isequal(mat_empty_dims(s18.empty), [0 0]) ...
        && isequal(double(s18.nested.a), 5);
    if good
        fprintf('  VALUES CORRECT\n');
    else
        fprintf('  VALUES WRONG -- the file loaded but decoded incorrectly\n');
    end
else
    good = false;
end

% --- the control that is expected to fail -----------------------------------
fprintf('\n');
fprintf('--- format_v110.mat (version 3 superblock, what 0.33.0 wrote) ---\n');
v110_loaded = false;
try
    s110 = load('format_v110.mat');  %#ok<NASGU>
    v110_loaded = true;
    fprintf('  LOADED (unexpected)\n');
catch err
    fprintf('  FAILED TO LOAD (expected): %s\n', err.message);
end

% --- verdict ----------------------------------------------------------------
fprintf('\n');
fprintf('=================== VERDICT ===================\n');
if is_octave
    fprintf('INCONCLUSIVE. Octave cannot distinguish the two formats, so this\n');
    fprintf('run says nothing about whether MATLAB can `load` the file. The\n');
    fprintf('content checks above did run, and are worth reading. Re-run under\n');
    fprintf('MATLAB for the answer.\n');
elseif good && ~v110_loaded
    fprintf('CONFIRMED. The 1.8-format file loads and the 1.10-format one does\n');
    fprintf('not, so the superblock version is exactly what broke `load` and the\n');
    fprintf('0.34.0 default fixes it.\n');
elseif good && v110_loaded && ~hdf5_known
    fprintf('PARTIALLY CONFIRMED. The 1.8-format file loads and decodes\n');
    fprintf('correctly, which is what the default rests on, so the claim this\n');
    fprintf('script exists to check holds. The 1.10-format file also loads, and\n');
    fprintf('this run could not read the HDF5 library version, so it cannot say\n');
    fprintf('whether that is the expected 1.10-or-newer outcome or the anomaly\n');
    fprintf('the branch below describes. Please report the MATLAB version above.\n');
elseif good && v110_loaded && ~expects_v110_to_fail
    fprintf('CONFIRMED for this release. The 1.8-format file loads and decodes\n');
    fprintf('correctly, which is what the default rests on. The 1.10-format file\n');
    fprintf('also loads, which is expected on HDF5 1.10 or newer -- it means this\n');
    fprintf('release was never affected, not that the 1.8 default is unnecessary:\n');
    fprintf('MATLAB before R2021b cannot open that file at all.\n');
elseif good && v110_loaded
    fprintf('UNEXPECTED. This MATLAB reports HDF5 1.8, which should not be able\n');
    fprintf('to open a version 3 superblock, yet the 1.10-format file loaded.\n');
    fprintf('The 1.8 default is still harmless, but the reasoning behind it does\n');
    fprintf('not hold here -- please report both versions printed above.\n');
elseif ~v18_loaded
    fprintf('FAILED. The 1.8-format file does not load, so something beyond the\n');
    fprintf('superblock version is wrong. Please report the error text above.\n');
else
    fprintf('FAILED. The 1.8-format file loads but decodes incorrectly. Please\n');
    fprintf('report the contents printed above.\n');
end
fprintf('===============================================\n');
fprintf('\n');
