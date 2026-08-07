function t = eq_text(a, b)
% Compare two strings for equality whether either arrived as `char` or `uint16`.
% See [as_codes] for why that varies by loader.
  t = isequal(as_codes(a), as_codes(b));
end
