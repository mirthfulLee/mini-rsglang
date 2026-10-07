//! CPU-only tensor-parallel geometry and rank coordination.
#![forbid(unsafe_code)]
mod group;
pub use group::{RankGroup, RankRunner};
use rsglang_core::{Error, Result};
use std::{collections::HashSet, ops::Range};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorParallel {
    rank: usize,
    size: usize,
}
impl Default for TensorParallel {
    fn default() -> Self {
        Self { rank: 0, size: 1 }
    }
}
impl TensorParallel {
    pub fn new(rank: usize, size: usize) -> Result<Self> {
        if size == 0 || size > i32::MAX as usize || rank >= size {
            return Err(Error::Invalid("invalid tensor-parallel rank/size".into()));
        }
        Ok(Self { rank, size })
    }
    pub fn rank(self) -> usize {
        self.rank
    }
    pub fn size(self) -> usize {
        self.size
    }
    pub fn partition(self, width: usize) -> Result<Range<usize>> {
        if width == 0 || !width.is_multiple_of(self.size) {
            return Err(Error::Invalid(format!(
                "dimension {width} must be divisible by TP size {}",
                self.size
            )));
        }
        let local = width / self.size;
        Ok(self.rank * local..(self.rank + 1) * local)
    }
    /// KV heads are replicated across adjacent ranks when TP exceeds their count.
    pub fn kv_heads(self, heads: usize) -> Result<Range<usize>> {
        if heads >= self.size {
            return self.partition(heads);
        }
        if heads == 0 || !self.size.is_multiple_of(heads) {
            return Err(Error::Invalid(
                "TP size and KV head count must divide one another".into(),
            ));
        }
        let head = self.rank / (self.size / heads);
        Ok(head..head + 1)
    }
    pub fn vocab(self, width: usize) -> Result<VocabShard> {
        if width == 0 {
            return Err(Error::Invalid("empty vocabulary".into()));
        }
        let padded_rows = width.div_ceil(self.size);
        let start = (self.rank * padded_rows).min(width);
        Ok(VocabShard {
            start,
            end: (start + padded_rows).min(width),
            padded_rows,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VocabShard {
    pub start: usize,
    pub end: usize,
    pub padded_rows: usize,
}

/// Describes checkpoint slicing before decoding/uploading the selected values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shard {
    Replicated,
    Rows {
        start: usize,
        len: usize,
        padded: usize,
    },
    Columns {
        start: usize,
        len: usize,
    },
}
impl Shard {
    pub fn rows(range: Range<usize>) -> Self {
        Self::Rows {
            start: range.start,
            len: range.len(),
            padded: range.len(),
        }
    }
    pub fn shape(&self, full: &[usize]) -> Result<Vec<usize>> {
        if full.is_empty() || full.contains(&0) {
            return Err(Error::Invalid("empty weight shape".into()));
        }
        let mut local = full.to_vec();
        match *self {
            Self::Replicated => {}
            Self::Rows { start, len, padded } => {
                if full.len() != 2
                    || start.checked_add(len).is_none_or(|n| n > full[0])
                    || padded < len
                    || padded == 0
                {
                    return Err(Error::Invalid("invalid row shard".into()));
                }
                local[0] = padded;
            }
            Self::Columns { start, len } => {
                if full.len() != 2 || len == 0 || start.checked_add(len).is_none_or(|n| n > full[1])
                {
                    return Err(Error::Invalid("invalid column shard".into()));
                }
                local[1] = len;
            }
        }
        Ok(local)
    }
    pub fn source_index(&self, full: &[usize], local_index: usize) -> Option<usize> {
        match *self {
            Self::Replicated => Some(local_index),
            Self::Rows { start, len, .. } => {
                (local_index / full[1] < len).then_some(start * full[1] + local_index)
            }
            Self::Columns { start, len } => {
                Some((local_index / len) * full[1] + start + local_index % len)
            }
        }
    }
}

pub fn validate_devices(devices: &[usize]) -> Result<()> {
    if devices.is_empty()
        || devices.len() > i32::MAX as usize
        || devices.iter().collect::<HashSet<_>>().len() != devices.len()
    {
        return Err(Error::Invalid(
            "TP requires a nonempty list of distinct CUDA devices".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partitions_and_replicated_kv_preserve_gqa() {
        for rank in 0..8 {
            let tp = TensorParallel::new(rank, 8).unwrap();
            assert_eq!(tp.partition(32).unwrap(), rank * 4..rank * 4 + 4);
            assert_eq!(tp.kv_heads(4).unwrap(), rank / 2..rank / 2 + 1);
            assert_eq!(tp.kv_heads(16).unwrap(), rank * 2..rank * 2 + 2);
        }
        assert!(TensorParallel::new(0, 3).unwrap().kv_heads(2).is_err());
        assert!(TensorParallel::new(0, 3).unwrap().partition(8).is_err());
        assert!(validate_devices(&[0, 0]).is_err());
    }
    #[test]
    fn odd_vocab_padding_and_column_indices() {
        let v = TensorParallel::new(3, 4).unwrap().vocab(17).unwrap();
        assert_eq!((v.start, v.end, v.padded_rows), (15, 17, 5));
        let shard = Shard::Rows {
            start: v.start,
            len: v.end - v.start,
            padded: v.padded_rows,
        };
        assert_eq!(shard.shape(&[17, 3]).unwrap(), [5, 3]);
        assert_eq!(shard.source_index(&[17, 3], 5), Some(50));
        assert_eq!(shard.source_index(&[17, 3], 6), None);
        let cols = Shard::Columns { start: 2, len: 2 };
        assert_eq!(cols.source_index(&[3, 8], 3), Some(11));
    }
}
