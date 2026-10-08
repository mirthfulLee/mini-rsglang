//! Read tensor ranges without materializing multi-gigabyte checkpoint shards.
use crate::invalid;
use rsglang_core::Result;
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub(super) struct TensorSource {
    file: PathBuf,
    offset: u64,
    bytes: usize,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
}
impl TensorSource {
    pub fn read(&self) -> Result<Vec<u8>> {
        self.read_range(0, self.bytes)
    }
    pub fn read_range(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        if start.checked_add(len).is_none_or(|n| n > self.bytes) {
            return Err(invalid("tensor byte range out of bounds"));
        }
        let mut file = File::open(&self.file)?;
        file.seek(SeekFrom::Start(self.offset + start as u64))?;
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}
#[derive(Deserialize)]
struct HeaderTensor {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
pub(super) struct Checkpoint {
    tensors: HashMap<String, TensorSource>,
}
impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let mut files = BTreeSet::new();
        let index = path.join("model.safetensors.index.json");
        if index.exists() {
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)
                .map_err(|e| invalid(e.to_string()))?;
            let map = value
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| invalid("missing safetensors weight_map"))?;
            for v in map.values() {
                let name = v
                    .as_str()
                    .ok_or_else(|| invalid("invalid shard filename"))?;
                if Path::new(name).components().count() != 1 || !name.ends_with(".safetensors") {
                    return Err(invalid(
                        "checkpoint shard must be a local safetensors filename",
                    ));
                }
                files.insert(name.to_owned());
            }
        } else {
            files.insert("model.safetensors".to_owned());
        }
        let mut tensors = HashMap::new();
        for name in files {
            let p = path.join(name);
            let mut file = File::open(&p)?;
            let length = file.metadata()?.len();
            let mut bytes = [0; 8];
            file.read_exact(&mut bytes)?;
            let header_len = u64::from_le_bytes(bytes);
            if header_len > 100_000_000 || header_len.checked_add(8).is_none_or(|n| n > length) {
                return Err(invalid("invalid safetensors header length"));
            }
            let mut header = vec![0; header_len as usize];
            file.read_exact(&mut header)?;
            let values: HashMap<String, serde_json::Value> =
                serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
            let base = 8 + header_len;
            let mut ranges = vec![];
            for (name, value) in values {
                if name == "__metadata__" {
                    continue;
                }
                let h: HeaderTensor =
                    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))?;
                let [start, end] = h.data_offsets;
                if !h.dtype.bitsize().is_multiple_of(8) {
                    return Err(invalid("sub-byte safetensors dtype is unsupported"));
                }
                let expected = h
                    .shape
                    .iter()
                    .try_fold(h.dtype.bitsize() / 8, |a, &b| a.checked_mul(b))
                    .ok_or_else(|| invalid("tensor size overflow"))?;
                if end < start || end > length - base || end - start != expected as u64 {
                    return Err(invalid(format!("invalid tensor byte range for {name}")));
                }
                ranges.push((start, end));
                let source = TensorSource {
                    file: p.clone(),
                    offset: base + start,
                    bytes: expected,
                    shape: h.shape,
                    dtype: h.dtype,
                };
                if tensors.insert(name.clone(), source).is_some() {
                    return Err(invalid(format!("duplicate tensor {name}")));
                }
            }
            ranges.sort_unstable();
            let mut next = 0;
            for (start, end) in ranges {
                if start != next {
                    return Err(invalid("overlapping or noncontiguous safetensors ranges"));
                }
                next = end;
            }
            if next != length - base {
                return Err(invalid("safetensors contains unreferenced bytes"));
            }
        }
        Ok(Self { tensors })
    }
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<TensorSource> {
        let source = self
            .tensors
            .remove(name)
            .ok_or_else(|| invalid(format!("missing weight {name}")))?;
        if source.shape != shape {
            return Err(invalid(format!(
                "{name}: shape {:?}, expected {shape:?}",
                source.shape
            )));
        }
        Ok(source)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn finish(self) -> Result<()> {
        if !self.tensors.is_empty() {
            return Err(invalid(format!(
                "unexpected weights: {:?}",
                self.tensors.keys().collect::<Vec<_>>()
            )));
        }
        Ok(())
    }
}
