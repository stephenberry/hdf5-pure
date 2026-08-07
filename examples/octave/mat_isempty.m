function t = mat_isempty(x)
% Is `x` an empty MATLAB value, or Octave's undecoded marker for one?
%
% See [mat_empty_dims] for why the second case exists: Octave 11's v7.3 `load`
% does not honor `MATLAB_empty`, so it returns the marker's raw dimension
% payload instead of an empty array — for MATLAB's own files as much as for
% this crate's.
  t = isempty(x) || any(mat_empty_dims(x) == 0);
end
