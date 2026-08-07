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
% MATLAB does not read `.mat` files with the HDF5 library it exposes through
% `h5read`/`h5disp`/`h5info`. Those go through HDF5 1.10.7; `load` for a MAT
% v7.3 file goes through a separate HDF5 1.8.12. A version 3 superblock is a
% 1.10 addition, so a file carrying one inspects fine under `h5disp` and fails
% under `load`.
%
% Every release of this crate through 0.33.0 wrote a version 3 superblock.
% 0.34.0 writes the HDF5 1.8 format by default. The two files here hold
% identical content and differ only in that format:
%
%   format_v18.mat    version 2 superblock -- the new default, must load
%   format_v110.mat   version 3 superblock -- what 0.33.0 wrote, expected to fail
%
% The second file is the control. A run where *both* load says the superblock
% version was never what broke `load`, and the diagnosis behind the 0.34.0
% change is wrong -- which is worth knowing, and is why the file is here rather
% than only the one that should work.
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
    fprintf('\n');
end

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
    fprintf('not, so the superblock version was the cause and the 0.34.0 default\n');
    fprintf('fixes it.\n');
elseif good && v110_loaded
    fprintf('PARTIAL. Both files load, so this MATLAB reads a version 3\n');
    fprintf('superblock and the superblock was NOT what broke `load` here.\n');
    fprintf('The 1.8 default is still harmless, but the diagnosis behind it does\n');
    fprintf('not hold for this MATLAB version -- please report the version above.\n');
elseif ~v18_loaded
    fprintf('FAILED. The 1.8-format file does not load, so something beyond the\n');
    fprintf('superblock version is wrong. Please report the error text above.\n');
else
    fprintf('FAILED. The 1.8-format file loads but decodes incorrectly. Please\n');
    fprintf('report the contents printed above.\n');
end
fprintf('===============================================\n');
fprintf('\n');
