function ok(cond, msg)
  if logical(cond)
    fprintf('  ok: %s\n', msg);
  else
    error('FAIL: %s', msg);
  end
end
