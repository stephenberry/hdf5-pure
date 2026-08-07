function d = mat_empty_dims(x)
% MATLAB dimensions of a value that is supposed to be empty.
%
% MAT v7.3 stores an empty array as a two-element uint64 dataset whose payload
% is the dimension vector, marked with the `MATLAB_empty` attribute. MATLAB's
% `load` honors that attribute and hands back a genuine empty array, so
% `size(x)` is the answer there.
%
% Octave 11's v7.3 `load` ignores the attribute and returns the raw payload: a
% 1x2 uint64 holding the dimensions. This is not a property of the files this
% crate writes. MATLAB's own empty markers use the identical encoding — of the
% 352 empty datasets in the genuine MATLAB files under `tests/fixtures/mat_real`,
% every one is this shape — and a dataset copied byte-for-byte out of one of
% those files reads back from Octave the same undecoded way. Under Octave, the
% payload *is* the dimension vector, so return it directly.
%
% The practical consequence: Octave cannot check that an empty value round-trips
% as empty. Only MATLAB can, which is what `check_format.m` is for.
  if exist('OCTAVE_VERSION', 'builtin') == 5 && isa(x, 'uint64') && numel(x) == 2
    d = double(x(:))';
  else
    d = size(x);
  end
end
