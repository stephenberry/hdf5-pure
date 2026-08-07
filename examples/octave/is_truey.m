function t = is_truey(x)
% Is `x` a MATLAB `true`, however this loader decoded it?
%
% MATLAB's `load` decodes MATLAB_class="logical" into a `logical`. Octave 11's
% v7.3 loader keeps it as `uint8`, the underlying storage class. Accept either
% rather than asserting one, since the file is the same in both cases.
  t = (islogical(x) && logical(x)) || (isnumeric(x) && x == 1);
end
