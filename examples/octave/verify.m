% Verify every fixture file loads correctly in MATLAB/Octave.
%
% Run:
%   cd matlab_fixtures
%   octave --no-gui --eval verify        % or just `verify` in MATLAB
%
% Note on logicals: MATLAB's `load` decodes MATLAB_class="logical" into a
% `logical`. Octave 11's `load` for v7.3 keeps them as `uint8` (the
% underlying storage class). The checks below accept either.

is_truey = @(x) (islogical(x) && logical(x)) || (isnumeric(x) && x == 1);
is_falsy = @(x) (islogical(x) && ~logical(x)) || (isnumeric(x) && x == 0);

fprintf('=== scalars.mat ===\n');
load scalars.mat
ok(x_f64 == 3.14159, 'x_f64');
ok(y_f32 == single(2.718), 'y_f32');
ok(n_i32 == int32(-42), 'n_i32');
ok(m_i64 == int64(9999999999), 'm_i64');
ok(u_u32 == uint32(2147483648), 'u_u32');
ok(v_u8 == uint8(255), 'v_u8');
ok(is_truey(b_true), 'b_true == 1');
ok(is_falsy(b_false), 'b_false == 0');
clearvars -except is_truey is_falsy

fprintf('=== vectors.mat ===\n');
load vectors.mat
ok(isequal(xs, [1.0; 2.0; 3.0; 4.0; 5.0]), 'xs');
ok(isequal(ns, int32([-1; 0; 1])), 'ns');
ok(isequal(double(flags(:)), [1; 0; 1; 1; 0]), 'flags values');
ok(isempty(empty), 'empty');
clearvars -except is_truey is_falsy

fprintf('=== matrix.mat ===\n');
load matrix.mat
expected = [1 2 3 4; 5 6 7 8; 9 10 11 12];
ok(isequal(a, expected), 'a is 3x4 with expected values');
ok(isequal(id, eye(2)), 'id is 2x2 identity');
clearvars -except is_truey is_falsy

fprintf('=== strings.mat ===\n');
load strings.mat
% In MATLAB this is `strcmp(ascii, 'hello MATLAB')`; Octave returns the raw
% char row so compare as char strings directly.
ok(numel(ascii) == 12 && all(double(ascii(:)') == double('hello MATLAB')), 'ascii');
ok(ischar(ascii) || isa(ascii, 'uint16'), 'ascii is char-like');
ok(isempty(empty), 'empty string');
clearvars -except is_truey is_falsy

fprintf('=== nested.mat ===\n');
load nested.mat
ok(isstruct(e), 'e is struct');
name_str = char(double(e.name(:))');
ok(strcmp(name_str, 'run_alpha'), 'e.name');
ok(e.trial == uint32(7), 'e.trial');
ok(abs(e.timestamp - 1.7e9) < 1, 'e.timestamp');
ok(isstruct(e.config), 'e.config is struct');
ok(strcmp(char(double(e.config.tag(:))'), 'prod'), 'e.config.tag');
ok(e.config.threshold == 0.85, 'e.config.threshold');
ok(e.config.max_iter == uint32(1000), 'e.config.max_iter');
ok(isequal(e.samples, [10.0; 20.0; 30.0; 40.0]), 'e.samples');
clearvars -except is_truey is_falsy

fprintf('=== options.mat ===\n');
load options.mat
vars = who;
ok(ismember('required', vars), 'required field present');
ok(ismember('present', vars), 'present field present');
ok(~ismember('absent', vars), 'absent field correctly missing');
ok(required == 1.5, 'required value');
ok(strcmp(char(double(present(:))'), 'yes'), 'present value');
clearvars -except is_truey is_falsy

fprintf('=== complex.mat ===\n');
load complex.mat
ok(iscomplex(z), 'z is complex');
ok(z == complex(1.0, -2.0), 'z value');
expected_signal = [complex(1,0); complex(0,1); complex(-1,0); complex(0,-1)];
ok(isequal(signal, expected_signal), 'signal');
clearvars -except is_truey is_falsy

fprintf('=== enum.mat ===\n');
load enum.mat
ok(strcmp(char(double(phase(:))'), 'Running'), 'phase');
clearvars -except is_truey is_falsy

fprintf('=== experiment.mat ===\n');
load experiment.mat
ok(strcmp(char(double(name(:))'), 'full_run'), 'name');
ok(abs(pi - 3.14159265358979) < 1e-10, 'pi');
ok(trial == uint32(42), 'trial');
ok(is_truey(active), 'active == 1');
ok(numel(samples) == 8, 'samples length');
ok(isequal(size(result), [2 3]), 'result size 2x3');
ok(iscomplex(signal) && numel(signal) == 3, 'signal complex 3-vec');
ok(strcmp(char(double(phase(:))'), 'Done'), 'phase');
ok(isstruct(config) && strcmp(char(double(config.tag(:))'), 'ship_it'), 'config.tag');
ok(strcmp(char(double(note(:))'), 'looks good'), 'note');
ok(~exist('skipped', 'var'), 'skipped absent');

fprintf('\nAll fixtures verified.\n');
