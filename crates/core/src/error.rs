//! One small error type for the whole crate. We avoid external error crates to
//! keep the dependency list (and the wasm binary) small.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The weights / tokenizer bytes were not what we expected.
    Format(String),
    /// A tensor we need is missing from the checkpoint.
    MissingTensor(String),
    /// The sequence would exceed the model's (or session's) context window.
    ContextFull { len: usize, max: usize },
    /// A token id outside the vocabulary.
    BadToken(u32),
    /// Unknown stage name (e.g. from a command line flag).
    UnknownStage(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(m) => write!(f, "format error: {m}"),
            Error::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Error::ContextFull { len, max } => {
                write!(f, "context full: {len} tokens exceeds max {max}")
            }
            Error::BadToken(t) => write!(f, "token id {t} is out of vocabulary"),
            Error::UnknownStage(s) => write!(f, "unknown stage '{s}'"),
        }
    }
}

impl std::error::Error for Error {}
