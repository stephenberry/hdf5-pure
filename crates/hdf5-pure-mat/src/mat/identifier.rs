//! MATLAB identifier validation, sanitization, and deduplication.
//!
//! MATLAB workspace variable names and struct field names must:
//! - Be 1 to 2048 ASCII characters long.
//! - Begin with an ASCII alphabetic character.
//! - Contain only ASCII alphanumerics and `_` after the first character.
//! - Not collide with a MATLAB keyword.
//!
//! Names that fail these rules either error out or get rewritten, depending on
//! the policy passed to the writer. Duplicate names within the same scope
//! always get a numeric suffix.

use std::collections::HashSet;

/// Maximum length of a MATLAB identifier.
pub const MATLAB_NAME_MAX: usize = 2048;

/// Reserved MATLAB keywords. Names equal to any of these are not valid
/// identifiers. Sorted for binary search.
pub const MATLAB_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "classdef",
    "continue",
    "else",
    "elseif",
    "end",
    "events",
    "for",
    "function",
    "global",
    "if",
    "methods",
    "otherwise",
    "parfor",
    "persistent",
    "properties",
    "return",
    "spmd",
    "switch",
    "try",
    "while",
];

/// Returns `true` if `name` is a valid MATLAB identifier (length, alphabet,
/// and not a keyword).
pub fn is_valid_name(name: &str) -> bool {
    if name.is_empty() || name.len() > MATLAB_NAME_MAX || is_keyword(name) {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

/// Returns `true` if `name` is a reserved MATLAB keyword.
pub fn is_keyword(name: &str) -> bool {
    MATLAB_KEYWORDS.binary_search(&name).is_ok()
}

/// Rewrite a non-identifier `name` into a valid MATLAB identifier.
///
/// Strategy:
/// - Replace an invalid first character with `'x'` followed by the original
///   char (or `'_'` if the original wasn't alphanumeric or `_`).
/// - Replace other invalid chars with `'_'`.
/// - If the result is empty, return `"x"`.
/// - If the result is a MATLAB keyword, append `'_'`.
/// - Truncate to [`MATLAB_NAME_MAX`] characters.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in name.chars().enumerate() {
        let valid = if idx == 0 {
            ch.is_ascii_alphabetic()
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if valid {
            out.push(ch);
        } else if idx == 0 {
            out.push('x');
            if ch.is_ascii_alphanumeric() || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    if is_keyword(&out) {
        out.push('_');
    }
    if out.len() > MATLAB_NAME_MAX {
        out.truncate(MATLAB_NAME_MAX);
    }
    out
}

/// Maximum number of dedupe iterations before we give up. With suffix
/// `_<usize>` and a 2048-char cap, you have to truly try to hit this; it
/// exists to fail loudly instead of looping in pathological inputs.
const MAX_DEDUPE_ATTEMPTS: usize = 1_000_000;

/// Append a numeric suffix to `candidate` until it is not present in `used`.
/// The suffix is `_1`, `_2`, ...; the base is truncated as needed to keep the
/// total length under [`MATLAB_NAME_MAX`]. Panics after
/// [`MAX_DEDUPE_ATTEMPTS`] collisions.
pub fn dedupe_name(mut candidate: String, used: &HashSet<String>) -> String {
    if !used.contains(&candidate) {
        return candidate;
    }
    let base = candidate.clone();
    for suffix in 1..=MAX_DEDUPE_ATTEMPTS {
        let suffix_str = format!("_{suffix}");
        let max_base_len = MATLAB_NAME_MAX.saturating_sub(suffix_str.len());
        if max_base_len == 0 {
            let tail_len = MATLAB_NAME_MAX.saturating_sub(1);
            let start = suffix_str.len().saturating_sub(tail_len);
            candidate.clear();
            candidate.push('x');
            candidate.push_str(&suffix_str[start..]);
        } else {
            let prefix = if base.len() > max_base_len {
                &base[..max_base_len]
            } else {
                &base
            };
            candidate.clear();
            candidate.push_str(prefix);
            candidate.push_str(&suffix_str);
        }
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    panic!("dedupe_name: exceeded {MAX_DEDUPE_ATTEMPTS} attempts for base name {base:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names_pass() {
        assert!(is_valid_name("foo"));
        assert!(is_valid_name("foo_bar"));
        assert!(is_valid_name("a1"));
        assert!(is_valid_name("ABC"));
    }

    #[test]
    fn invalid_names_rejected() {
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("1foo"));
        assert!(!is_valid_name("foo bar"));
        assert!(!is_valid_name("foo-bar"));
        assert!(!is_valid_name("end"));
        assert!(!is_valid_name("for"));
    }

    #[test]
    fn sanitize_inserts_x_prefix() {
        assert_eq!(sanitize_name("1 bad"), "x1_bad");
        assert_eq!(sanitize_name(""), "x");
        assert_eq!(sanitize_name("foo bar"), "foo_bar");
        assert_eq!(sanitize_name("end"), "end_");
    }

    #[test]
    fn dedupe_appends_numeric_suffix() {
        let mut used = HashSet::new();
        used.insert("foo".to_owned());
        assert_eq!(dedupe_name("foo".to_owned(), &used), "foo_1");
        used.insert("foo_1".to_owned());
        assert_eq!(dedupe_name("foo".to_owned(), &used), "foo_2");
    }
}
