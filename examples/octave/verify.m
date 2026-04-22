% Verify every fixture file loads correctly in MATLAB/Octave.
%
% Run:
%   cd matlab_fixtures
%   octave --no-gui --eval verify        % or just `verify` in MATLAB
%
% Note on logicals: MATLAB's `load` decodes MATLAB_class="logical" into a
% `logical`. Octave 11's `load` for v7.3 keeps them as `uint8` (the
% underlying storage class). The checks below accept either.
%
% Note on char: Octave 11's `load` keeps MATLAB_class="char" as `uint16`
% (the underlying storage class). MATLAB itself returns `char`. The helper
% `eq_text` normalizes both by flattening to double code units.

is_truey = @(x) (islogical(x) && logical(x)) || (isnumeric(x) && x == 1);
is_falsy = @(x) (islogical(x) && ~logical(x)) || (isnumeric(x) && x == 0);
% Convert a char/uint16 MATLAB loaded value into a flat row of code-unit doubles.
as_codes = @(x) double(x(:))';
% Compare two strings (possibly char or uint16) for equality.
eq_text  = @(a, b) isequal(as_codes(a), as_codes(b));

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
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== vectors.mat ===\n');
load vectors.mat
ok(isequal(xs, [1.0; 2.0; 3.0; 4.0; 5.0]), 'xs');
ok(isequal(ns, int32([-1; 0; 1])), 'ns');
ok(isequal(double(flags(:)), [1; 0; 1; 1; 0]), 'flags values');
ok(isempty(empty), 'empty');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== matrix.mat ===\n');
load matrix.mat
expected = [1 2 3 4; 5 6 7 8; 9 10 11 12];
ok(isequal(a, expected), 'a is 3x4 with expected values');
ok(isequal(id, eye(2)), 'id is 2x2 identity');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== strings.mat ===\n');
load strings.mat
ok(numel(ascii) == 12 && eq_text(ascii, 'hello MATLAB'), 'ascii');
ok(ischar(ascii) || isa(ascii, 'uint16'), 'ascii is char-like');
ok(isempty(empty), 'empty string');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== nested.mat ===\n');
load nested.mat
ok(isstruct(e), 'e is struct');
ok(eq_text(e.name, 'run_alpha'), 'e.name');
ok(e.trial == uint32(7), 'e.trial');
ok(abs(e.timestamp - 1.7e9) < 1, 'e.timestamp');
ok(isstruct(e.config), 'e.config is struct');
ok(eq_text(e.config.tag, 'prod'), 'e.config.tag');
ok(e.config.threshold == 0.85, 'e.config.threshold');
ok(e.config.max_iter == uint32(1000), 'e.config.max_iter');
ok(isequal(e.samples, [10.0; 20.0; 30.0; 40.0]), 'e.samples');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== options.mat ===\n');
load options.mat
vars = who;
ok(ismember('required', vars), 'required field present');
ok(ismember('present', vars), 'present field present');
ok(~ismember('absent', vars), 'absent field correctly missing');
ok(required == 1.5, 'required value');
ok(eq_text(present, 'yes'), 'present value');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== complex.mat ===\n');
load complex.mat
ok(iscomplex(z), 'z is complex');
ok(z == complex(1.0, -2.0), 'z value');
expected_signal = [complex(1,0); complex(0,1); complex(-1,0); complex(0,-1)];
ok(isequal(signal, expected_signal), 'signal');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== enum.mat ===\n');
load enum.mat
ok(eq_text(phase, 'Running'), 'phase');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== experiment.mat ===\n');
load experiment.mat
ok(eq_text(name, 'full_run'), 'name');
ok(abs(pi - 3.14159265358979) < 1e-10, 'pi');
ok(trial == uint32(42), 'trial');
ok(is_truey(active), 'active == 1');
ok(numel(samples) == 8, 'samples length');
ok(isequal(size(result), [2 3]), 'result size 2x3');
ok(iscomplex(signal) && numel(signal) == 3, 'signal complex 3-vec');
ok(eq_text(phase, 'Done'), 'phase');
ok(isstruct(config) && eq_text(config.tag, 'ship_it'), 'config.tag');
ok(eq_text(note, 'looks good'), 'note');
ok(~exist('skipped', 'var'), 'skipped absent');
clearvars -except is_truey is_falsy as_codes eq_text

% ==========================================================================
% Edge-case fixtures
% ==========================================================================

fprintf('=== extremes.mat ===\n');
load extremes.mat
ok(isnan(nan64), 'nan64 is NaN');
ok(isinf(pos_inf) && pos_inf > 0, 'pos_inf');
ok(isinf(neg_inf) && neg_inf < 0, 'neg_inf');
% -0.0 compares equal to 0.0, so inspect the IEEE 754 bit pattern (sign bit set).
ok(neg_zero == 0 && typecast(double(neg_zero), 'uint64') == uint64(hex2dec('8000000000000000')), 'neg_zero sign bit preserved');
ok(subnormal > 0 && subnormal < 1e-300, 'subnormal > 0 and tiny');
% Octave 11's v7.3 loader does not preserve the `single` class (loads as
% double). MATLAB_class is correctly written as "single" in our file — real
% MATLAB does preserve it. Just verify the value.
ok(isnan(nan32), 'nan32 is NaN');
ok(isinf(pos_inf32), 'pos_inf32 is Inf');
ok(isa(i64_min, 'int64') && i64_min == intmin('int64'), 'i64_min');
ok(isa(i64_max, 'int64') && i64_max == intmax('int64'), 'i64_max');
ok(isa(i32_min, 'int32') && i32_min == intmin('int32'), 'i32_min');
ok(isa(i32_max, 'int32') && i32_max == intmax('int32'), 'i32_max');
ok(isa(u64_max, 'uint64') && u64_max == intmax('uint64'), 'u64_max');
ok(isa(i8_min, 'int8') && i8_min == intmin('int8'), 'i8_min');
ok(isa(i8_max, 'int8') && i8_max == intmax('int8'), 'i8_max');
ok(isa(u8_max, 'uint8') && u8_max == uint8(255), 'u8_max');
% NaN-vector check: use isnan()/isinf() per position, since x == NaN is false.
ok(numel(nan_vec) == 5, 'nan_vec length');
ok(nan_vec(1) == 1.0, 'nan_vec(1) == 1');
ok(isnan(nan_vec(2)), 'nan_vec(2) is NaN');
ok(nan_vec(3) == Inf, 'nan_vec(3) is +Inf');
ok(nan_vec(4) == -Inf, 'nan_vec(4) is -Inf');
ok(nan_vec(5) == 0 && typecast(double(nan_vec(5)), 'uint64') == uint64(hex2dec('8000000000000000')), 'nan_vec(5) -0 sign bit');
ok(numel(i64_extremes) == 5, 'i64_extremes length');
ok(i64_extremes(1) == intmin('int64'), 'i64_extremes(1) == i64min');
ok(i64_extremes(5) == intmax('int64'), 'i64_extremes(5) == i64max');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== shapes.mat ===\n');
load shapes.mat
% 1x1 matrix is a scalar in MATLAB but size is [1,1].
ok(isequal(size(m_1x1), [1 1]), 'm_1x1 size [1,1]');
ok(m_1x1 == 42.0, 'm_1x1 value');
% 1xN is a row vector.
ok(isequal(size(m_1x5), [1 5]), 'm_1x5 size [1,5]');
ok(isequal(m_1x5, [10 20 30 40 50]), 'm_1x5 values');
% Nx1 is a column vector.
ok(isequal(size(m_5x1), [5 1]), 'm_5x1 size [5,1]');
ok(isequal(m_5x1, [10; 20; 30; 40; 50]), 'm_5x1 values');
% 2x3 matrix (rows < cols)
ok(isequal(size(m_2x3), [2 3]), 'm_2x3 size');
ok(isequal(m_2x3, [1 2 3; 4 5 6]), 'm_2x3 values');
% 3x2 matrix (rows > cols)
ok(isequal(size(m_3x2), [3 2]), 'm_3x2 size');
ok(isequal(m_3x2, [1 2; 3 4; 5 6]), 'm_3x2 values');
% 3x3 with distinct per-cell values: m_3x3(r,c) = 10*r + c.
ok(isequal(size(m_3x3), [3 3]), 'm_3x3 size');
ok(m_3x3(1,1) == 11 && m_3x3(1,2) == 12 && m_3x3(1,3) == 13, 'm_3x3 row 1');
ok(m_3x3(2,1) == 21 && m_3x3(2,2) == 22 && m_3x3(2,3) == 23, 'm_3x3 row 2');
ok(m_3x3(3,1) == 31 && m_3x3(3,2) == 32 && m_3x3(3,3) == 33, 'm_3x3 row 3');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== int_matrices.mat ===\n');
load int_matrices.mat
% Every Matrix<T> should preserve both its integer class and its values.
% Octave may store logical as uint8, so accept either class for bools.
ok(isa(m_i8, 'int8') && isequal(m_i8, int8([-128 127; 0 -1])), 'm_i8');
ok(isa(m_i16, 'int16') && isequal(m_i16, int16([-32768 32767; 0 -1])), 'm_i16');
ok(isa(m_i32, 'int32') && isequal(m_i32, int32([-1 2; 3 4])), 'm_i32');
ok(isa(m_i64, 'int64') && isequal(m_i64, int64([-1 2; 3 intmax('int64')])), 'm_i64');
ok(isa(m_u8, 'uint8') && isequal(m_u8, uint8([0 1 2; 253 254 255])), 'm_u8');
ok(isa(m_u16, 'uint16') && isequal(m_u16, uint16([0 1; 65534 65535])), 'm_u16');
ok(isa(m_u32, 'uint32') && isequal(m_u32, uint32([0 1; 4294967294 4294967295])), 'm_u32');
ok(isa(m_u64, 'uint64') && isequal(m_u64, uint64([0 1; 2 intmax('uint64')])), 'm_u64');
% Logical matrix: class is either 'logical' (MATLAB) or 'uint8' (Octave v7.3).
ok(isequal(size(m_bool), [2 2]), 'm_bool size');
ok(isequal(double(m_bool), [1 0; 0 1]), 'm_bool values');
ok(islogical(m_bool) || isa(m_bool, 'uint8'), 'm_bool class is logical-like');
% f32 matrix (Octave 11 loads as double; check values only)
ok(isequal(double(m_f32), [1.5 2.5; 3.5 4.5]), 'm_f32 values');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== unicode.mat ===\n');
load unicode.mat
% Verify code units explicitly — Octave interprets UTF-8 source literals as
% raw bytes, so we can't compare against `'café'` directly.
% latin1 = "café — naïve — résumé" — 21 UTF-16 code units.
latin1_expected = [99 97 102 233 32 8212 32 110 97 239 118 101 32 8212 32 114 233 115 117 109 233];
ok(isequal(as_codes(latin1), latin1_expected), 'latin1 code units');
% cjk = "日本語テスト" — 6 UTF-16 code units (all in BMP).
cjk_expected = [26085 26412 35486 12486 12473 12488];
ok(isequal(as_codes(cjk), cjk_expected), 'cjk code units');
% Emoji 🎉 (U+1F389) must survive as a surrogate pair: 0xD83C (55356), 0xDF89 (57225).
ok(numel(emoji) == 2, 'emoji is 2 code units (surrogate pair)');
ok(isequal(as_codes(emoji), [55356 57225]), 'emoji surrogate pair values');
% Mixed: é (1 unit) + 日 (1 unit) + 🎉 (2 units) + A (1 unit) = 5 units
ok(numel(mixed) == 5, 'mixed length 5 code units');
ok(isequal(as_codes(mixed), [233 26085 55356 57225 65]), 'mixed code units');
% Multi-line: ensure \n (10) and \t (9) preserved.
mcodes = as_codes(multiline);
ok(ismember(10, mcodes), 'multiline contains \\n');
ok(ismember(9, mcodes), 'multiline contains \\t');
% "line1\nline2\tindented\nend" = l i n e 1 \n l i n e 2 \t i n d e n t e d \n e n d
ok(numel(multiline) == 24, 'multiline length 24');
ok(numel(one_char) == 1 && as_codes(one_char) == 88, 'one_char is "X" (88)');
ok(numel(long_ascii) == 5000, 'long_ascii length 5000');
long_codes = as_codes(long_ascii);
ok(long_codes(1) == 97, 'long_ascii first char (a=97)');
ok(long_codes(5000) == 106, 'long_ascii last char (j=106)');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== complex_edges.mat ===\n');
load complex_edges.mat
ok(iscomplex(z_nan), 'z_nan is complex');
ok(isnan(real(z_nan)) && imag(z_nan) == 0, 'z_nan real is NaN');
ok(iscomplex(z_inf), 'z_inf is complex');
ok(isinf(real(z_inf)) && real(z_inf) > 0, 'z_inf real is +Inf');
ok(isinf(imag(z_inf)) && imag(z_inf) < 0, 'z_inf imag is -Inf');
ok(z_zero == 0, 'z_zero == 0');
ok(real(z_pure_imag) == 0 && imag(z_pure_imag) == 2.5, 'z_pure_imag');
ok(real(z_pure_real) == 3.5 && imag(z_pure_real) == 0, 'z_pure_real');
% Octave 11 v7.3 loader loses the single class (real MATLAB preserves it).
ok(iscomplex(z32), 'z32 is complex');
ok(double(real(z32)) == 1.25 && double(imag(z32)) == -0.5, 'z32 values');
ok(numel(z32_vec) == 3, 'z32_vec length');
ok(iscomplex(z32_vec), 'z32_vec is complex');
ok(isequal(size(cmat), [2 2]), 'cmat size');
ok(isequal(cmat, [1 2; 3 4]), 'cmat values');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== deep_nested.mat ===\n');
load deep_nested.mat
ok(isstruct(root), 'root is struct');
ok(eq_text(root.label, 'top'), 'root.label');
ok(isstruct(root.inner), 'root.inner is struct');
ok(root.inner.depth == uint32(2), 'root.inner.depth');
ok(isstruct(root.inner.sub), 'root.inner.sub is struct');
ok(eq_text(root.inner.sub.tag, 'middle'), 'root.inner.sub.tag');
ok(isstruct(root.inner.sub.leaf), 'root.inner.sub.leaf is struct');
ok(root.inner.sub.leaf.id == uint64(12345), 'leaf.id');
ok(isequal(root.inner.sub.leaf.values, [1.5; 2.5; 3.5]), 'leaf.values');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== bool_ext.mat ===\n');
load bool_ext.mat
% Single-element bool vec: should still be a vector (1x1 or 1-element).
ok(numel(single) == 1, 'single-bool vec has 1 element');
ok(double(single(1)) == 1, 'single is true');
% 3x3 logical matrix with expected pattern.
ok(isequal(size(mat), [3 3]), 'mat size 3x3');
ok(isequal(double(mat), [1 0 1; 0 1 0; 1 0 1]), 'mat values');
ok(islogical(mat) || isa(mat, 'uint8'), 'mat logical-like');
% flags vector
ok(numel(flags) == 7, 'flags length 7');
ok(isequal(double(flags(:))', [1 1 0 1 0 0 1]), 'flags contents');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== empty_variants.mat ===\n');
load empty_variants.mat
ok(isempty(e_f64), 'e_f64 empty');
ok(isempty(e_f32), 'e_f32 empty');
ok(isempty(e_i32), 'e_i32 empty');
ok(isempty(e_u8), 'e_u8 empty');
ok(isempty(e_bool), 'e_bool empty');
ok(isempty(e_str), 'e_str empty string');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('=== large_matrix.mat ===\n');
load large_matrix.mat
ok(isequal(size(m), [100 50]), 'large m size 100x50');
ok(m(1,1) == 0, 'm(1,1) == 0');
ok(m(1,50) == 49, 'm(1,50) == 49');
ok(m(100,1) == 99000, 'm(100,1) == 99000');
ok(m(100,50) == 99000 + 49, 'm(100,50) == 99049');
ok(m(7,13) == 6 * 1000 + 12, 'm(7,13) interior');
ok(rows == uint32(100) && cols == uint32(50), 'rows/cols scalar fields');
clearvars -except is_truey is_falsy as_codes eq_text

fprintf('\nAll fixtures verified.\n');
