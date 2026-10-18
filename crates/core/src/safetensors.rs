//! A minimal reader for the `.safetensors` format.
//!
//! The format is refreshingly simple:
//!
//! ```text
//! [ u64 little-endian: N ][ N bytes of JSON header ][ raw tensor bytes ... ]
//! ```
//!
//! The JSON maps tensor name -> `{ "dtype": "F32", "shape": [..], "data_offsets": [start, end] }`
//! where offsets are relative to the end of the header. We only need F32.

use crate::error::{Error, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range within the data section.
    pub start: usize,
    pub end: usize,
}

/// A parsed view over safetensors bytes. Borrows the bytes (e.g. an mmap) and
/// copies nothing until you ask for a tensor.
pub struct SafeTensors<'a> {
    data: &'a [u8],
    tensors: HashMap<String, TensorInfo>,
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let fmt = |m: &str| Error::Format(format!("safetensors: {m}"));
        if bytes.len() < 8 {
            return Err(fmt("file too small"));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = bytes.get(8..8 + n).ok_or_else(|| fmt("truncated header"))?;
        let json: serde_json::Value = serde_json::from_slice(header).map_err(|e| fmt(&e.to_string()))?;
        let obj = json.as_object().ok_or_else(|| fmt("header is not an object"))?;

        let mut tensors = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or_else(|| fmt("missing dtype"))?;
            let shape = v["shape"]
                .as_array()
                .ok_or_else(|| fmt("missing shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| fmt("bad shape"))?;
            let offs = v["data_offsets"].as_array().ok_or_else(|| fmt("missing offsets"))?;
            let (start, end) = match (offs.first().and_then(|x| x.as_u64()), offs.get(1).and_then(|x| x.as_u64())) {
                (Some(s), Some(e)) => (s as usize, e as usize),
                _ => return Err(fmt("bad offsets")),
            };
            tensors.insert(name.clone(), TensorInfo { dtype: dtype.to_string(), shape, start, end });
        }
        Ok(SafeTensors { data: &bytes[8 + n..], tensors })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Copy an F32 tensor out as a `Vec<f32>`, checking its shape.
    ///
    /// We copy (rather than reinterpret the bytes in place) because the data is
    /// not guaranteed to be 4-byte aligned, and because we often want to
    /// re-layout weights at load time anyway.
    pub fn f32(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.info(name).ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if info.dtype != "F32" {
            return Err(Error::Format(format!("{name}: expected F32, got {}", info.dtype)));
        }
        if info.shape != expected_shape {
            return Err(Error::Format(format!("{name}: expected shape {expected_shape:?}, got {:?}", info.shape)));
        }
        let raw =
            self.data.get(info.start..info.end).ok_or_else(|| Error::Format(format!("{name}: data out of bounds")))?;
        if raw.len() != expected_shape.iter().product::<usize>() * 4 {
            return Err(Error::Format(format!("{name}: byte length mismatch")));
        }
        let (chunks, _) = raw.as_chunks::<4>();
        Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }
}
