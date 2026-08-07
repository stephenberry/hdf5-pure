function t = has_sign_bit(x)
% Is `x` a float whose IEEE 754 sign bit is set? The way to tell -0.0 from 0.0,
% which compare equal.
%
% `bitshift(uint64(1), 63)` rather than the more obvious
% `hex2dec('8000000000000000')`: hex2dec returns a *double*, and 2^63 is past
% flintmax, so MATLAB warns on every call that the value may not survive the
% conversion. This one does survive, being a power of two, but the warning is
% right about the technique, and a check that cries wolf each run gets ignored.
  t = typecast(double(x), 'uint64') == bitshift(uint64(1), 63);
end
