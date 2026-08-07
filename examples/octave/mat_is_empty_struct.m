function t = mat_is_empty_struct(x)
% Is `x` MATLAB's `struct([])`, an empty struct array with no fields?
%
% This is what the default `NullPolicy::EmptyStructArray` writes for a Rust
% `None` or `()`, so `isfield` reports true and MATLAB code can reference the
% field unconditionally.
%
% Under MATLAB the check is exact. Under Octave it cannot be: Octave 11's v7.3
% `load` ignores `MATLAB_empty` and hands back the marker's raw dimension
% payload as a uint64 pair, losing the `MATLAB_class` distinction along with it
% — so an empty struct and an empty double are indistinguishable there. See
% [mat_empty_dims] for why that is Octave's behavior rather than a property of
% these files. The weaker Octave check is stated here rather than hidden in
% each call site, so it is clear which assertion is being relaxed and why.
  if isstruct(x)
    t = isempty(fieldnames(x));
  else
    t = mat_isempty(x);
  end
end
