function c = as_codes(x)
% Flatten a loaded string to a row of UTF-16 code units as doubles.
%
% Octave 11's v7.3 loader keeps MATLAB_class="char" as `uint16`, the underlying
% storage class, where MATLAB returns `char`. Comparing code units compares what
% the file actually holds and works under both.
  c = double(x(:))';
end
