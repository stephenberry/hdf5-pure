function t = is_falsy(x)
% Is `x` a MATLAB `false`? The counterpart of [is_truey]; see it for why both
% `logical` and `uint8` are accepted.
  t = (islogical(x) && ~logical(x)) || (isnumeric(x) && x == 0);
end
