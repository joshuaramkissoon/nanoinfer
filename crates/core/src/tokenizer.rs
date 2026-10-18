//! GPT-2's byte-level BPE tokenizer, from scratch.
//!
//! Turning text into token ids happens in three steps:
//!
//! 1. **Pre-tokenize.** Split the text into chunks ("words") with a fixed rule:
//!    contractions (`'s`, `'ll`, ...), runs of letters, runs of digits, runs of
//!    other symbols (each optionally led by one space), and whitespace. BPE never
//!    merges across chunk boundaries. GPT-2 does this with a regex; we hand-code
//!    it because the regex uses a look-ahead (`\s+(?!\S)`) and because it's
//!    clearer to read as code.
//! 2. **Bytes to symbols.** Each chunk becomes its UTF-8 bytes, so *any* string can
//!    be encoded with no "unknown" token. Each of the 256 bytes has its own token.
//! 3. **Merge.** Repeatedly merge the adjacent pair with the best (lowest) rank in
//!    `merges.txt` until no mergeable pair is left. Rank = how early in training
//!    that merge was learned, so common pairs like `t`+`h` merge first.
//!
//! Decoding is the reverse: concatenate each token's bytes and read them as UTF-8.
//!
//! ### The byte-to-unicode detour
//! `vocab.json` and `merges.txt` are text files, but tokens are byte strings,
//! and some bytes (control chars, space) are unprintable or would confuse the
//! file format. GPT-2 therefore maps each byte to a printable unicode character
//! ([`bytes_to_unicode`]): printable bytes map to themselves and the rest are
//! shifted to U+0100 and up. That's why a leading space shows up as `Ġ` in the
//! vocab. We undo the mapping at load time and work with raw bytes throughout.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// GPT-2's `<|endoftext|>` token.
pub const END_OF_TEXT: u32 = 50256;
const END_OF_TEXT_STR: &str = "<|endoftext|>";

/// The reversible byte -> printable char table used by GPT-2's vocab files.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut next = 256;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    table
}

pub struct Tokenizer {
    /// Bytes of each token id.
    id_to_bytes: Vec<Vec<u8>>,
    /// Token id of each single byte (the starting symbols).
    byte_to_id: [u32; 256],
    /// (left id, right id) -> (rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
}

impl Tokenizer {
    /// Build from the contents of HF's `vocab.json` (a.k.a. `encoder.json`) and `merges.txt`.
    pub fn from_vocab_and_merges(vocab_json: &[u8], merges_txt: &[u8]) -> Result<Tokenizer> {
        let err = |m: String| Error::Format(format!("tokenizer: {m}"));
        let vocab: HashMap<String, u32> = serde_json::from_slice(vocab_json).map_err(|e| err(e.to_string()))?;

        // Invert the byte->char table so we can turn vocab strings back into bytes.
        let b2u = bytes_to_unicode();
        let u2b: HashMap<char, u8> = b2u.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let to_bytes = |s: &str| -> Result<Vec<u8>> {
            s.chars().map(|c| u2b.get(&c).copied().ok_or_else(|| err(format!("unmapped char {c:?}")))).collect()
        };

        let n = vocab.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut id_to_bytes = vec![Vec::new(); n];
        let mut str_to_id: HashMap<&str, u32> = HashMap::with_capacity(vocab.len());
        for (s, &id) in &vocab {
            // Special tokens like <|endoftext|> are stored literally.
            id_to_bytes[id as usize] = if s == END_OF_TEXT_STR { s.as_bytes().to_vec() } else { to_bytes(s)? };
            str_to_id.insert(s.as_str(), id);
        }

        let mut byte_to_id = [0u32; 256];
        for (b, c) in b2u.iter().enumerate() {
            let s = c.to_string();
            byte_to_id[b] = *str_to_id.get(s.as_str()).ok_or_else(|| err(format!("no token for byte {b}")))?;
        }

        let text = std::str::from_utf8(merges_txt).map_err(|e| err(e.to_string()))?;
        let mut merges = HashMap::new();
        let lines = text.lines().filter(|l| !l.starts_with("#version") && !l.trim().is_empty());
        for (rank, line) in lines.enumerate() {
            let (a, b) = line.split_once(' ').ok_or_else(|| err(format!("bad merge line {line:?}")))?;
            let id = |s: &str| str_to_id.get(s).copied().ok_or_else(|| err(format!("merge token {s:?} not in vocab")));
            let merged = format!("{a}{b}");
            merges.insert((id(a)?, id(b)?), (rank as u32, id(&merged)?));
        }

        Ok(Tokenizer { id_to_bytes, byte_to_id, merges })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Encode text to token ids. `<|endoftext|>` in the text becomes token 50256,
    /// matching Hugging Face's tokenizer.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for (i, part) in text.split(END_OF_TEXT_STR).enumerate() {
            if i > 0 {
                ids.push(END_OF_TEXT);
            }
            for word in pre_tokenize(part) {
                self.bpe(word.as_bytes(), &mut ids);
            }
        }
        ids
    }

    /// Byte-pair-encode one pre-tokenized chunk, appending ids to `out`.
    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = bytes.iter().map(|&b| self.byte_to_id[b as usize]).collect();
        loop {
            // Find the adjacent pair that was learned earliest (lowest rank).
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id)))
                .min();
            match best {
                Some((_, i, id)) => {
                    syms[i] = id;
                    syms.remove(i + 1);
                }
                None => break,
            }
        }
        out.extend_from_slice(&syms);
    }

    /// The raw bytes of one token. A token can be part of a multi-byte UTF-8
    /// character, so streaming output should buffer bytes until they're valid UTF-8.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.id_to_bytes.get(id as usize).map_or(&[], |b| b.as_slice())
    }

    /// Decode ids to text (invalid UTF-8 is replaced with U+FFFD, like HF).
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id).iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's pre-tokenization, equivalent to the regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// applied left to right.
///
/// Approximation: we use Rust's `char::is_alphabetic` / `is_numeric` for `\p{L}` /
/// `\p{N}`. They agree for essentially all real text; `is_alphabetic` also
/// accepts a few combining marks that `\p{L}` doesn't.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    #[derive(PartialEq, Clone, Copy)]
    enum Class {
        Letter,
        Number,
        Space,
        Other,
    }
    let class = |c: char| {
        if c.is_whitespace() {
            Class::Space
        } else if c.is_alphabetic() {
            Class::Letter
        } else if c.is_numeric() {
            Class::Number
        } else {
            Class::Other
        }
    };

    // (byte offset, char) pairs, plus a sentinel offset for the end.
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let offset = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let rest = &text[offset(i)..];
        let c = chars[i].1;

        // 1. Contractions.
        if c == '\'' {
            if let Some(suf) = ["s", "t", "re", "ve", "m", "ll", "d"].iter().find(|s| rest[1..].starts_with(*s)) {
                let len = 1 + suf.len(); // all ASCII, so chars == bytes
                out.push(&rest[..len]);
                i += len;
                continue;
            }
        }

        // 2-4. An optional single space, then a run of letters, digits, or "other".
        let start = if c == ' ' { i + 1 } else { i };
        if start < n {
            let cls = class(chars[start].1);
            if cls != Class::Space {
                let mut end = start + 1;
                while end < n && class(chars[end].1) == cls {
                    end += 1;
                }
                out.push(&text[offset(i)..offset(end)]);
                i = end;
                continue;
            }
        }

        // 5-6. Whitespace. `\s+(?!\S)` means: take the whitespace run, but if it's
        // followed by a non-space, leave the last whitespace char behind so it can
        // become the leading space of the next word (" world").
        let mut end = i + 1;
        while end < n && class(chars[end].1) == Class::Space {
            end += 1;
        }
        if end < n && end - i >= 2 {
            end -= 1;
        }
        out.push(&text[offset(i)..offset(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_tokenize_matches_gpt2_regex() {
        assert_eq!(pre_tokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pre_tokenize("I'll don't"), vec!["I", "'ll", " don", "'t"]);
        assert_eq!(pre_tokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pre_tokenize("x\n\ny "), vec!["x", "\n", "\n", "y", " "]);
        assert_eq!(pre_tokenize("pi=3.14!"), vec!["pi", "=", "3", ".", "14", "!"]);
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let t = bytes_to_unicode();
        let mut seen: Vec<char> = t.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 256);
        assert_eq!(t[b' ' as usize], 'Ġ');
    }
}
