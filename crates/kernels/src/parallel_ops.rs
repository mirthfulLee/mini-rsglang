use super::*;

impl CudaBackend {
    pub(crate) fn concat_checked(&self, tensors: &[Tensor]) -> Result<Tensor> {
        if tensors.is_empty() || tensors[0].shape.len() != 2 {
            return Err(invalid("concat requires matrices"));
        }
        let cols = tensors[0].shape[1];
        let mut rows = 0usize;
        for t in tensors {
            self.check(t)?;
            if t.shape.len() != 2 || t.shape[1] != cols {
                return Err(invalid("concat matrix widths differ"));
            }
            rows = rows
                .checked_add(t.shape[0])
                .ok_or_else(|| invalid("concat overflow"))?;
        }
        let mut out = self.zeros(&[rows, cols])?;
        let mut offset = 0;
        for t in tensors {
            self.stream
                .memcpy_dtod(
                    &t.data,
                    &mut out.data.slice_mut(offset..offset + t.data.len()),
                )
                .map_err(backend)?;
            offset += t.data.len();
        }
        Ok(out)
    }
    pub(crate) fn split_checked(&self, x: &Tensor, widths: &[usize]) -> Result<Vec<Tensor>> {
        self.check(x)?;
        if x.shape.len() != 2
            || widths.is_empty()
            || widths.contains(&0)
            || widths.iter().try_fold(0usize, |a, &b| a.checked_add(b)) != Some(x.shape[1])
        {
            return Err(invalid("invalid column splits"));
        }
        let (n, full) = (int(x.shape[0])?, int(x.shape[1])?);
        let fun = self.function("split_columns")?;
        let mut result = vec![];
        let mut offset = 0usize;
        for &width in widths {
            let mut out = self.zeros(&[x.shape[0], width])?;
            let (start, w) = (int(offset)?, int(width)?);
            let cfg = Self::flat(out.data.len())?;
            // SAFETY: positive partitions cover the input columns exactly; each output owns
            // its buffer and each thread writes one in-range element.
            unsafe {
                self.stream
                    .launch_builder(&fun)
                    .arg(&x.data)
                    .arg(&mut out.data)
                    .arg(&n)
                    .arg(&full)
                    .arg(&start)
                    .arg(&w)
                    .launch(cfg)
            }
            .map_err(backend)?;
            result.push(out);
            offset += width;
        }
        Ok(result)
    }
    pub(crate) fn embed_shard_checked(
        &self,
        w: &Tensor,
        m: &Metadata,
        start: usize,
    ) -> Result<Tensor> {
        self.check(w)?;
        self.check_meta(m)?;
        let shard = self.tp.vocab(m.vocab)?;
        if start != shard.start || w.shape.len() != 2 || w.shape[0] != shard.padded_rows {
            return Err(invalid("embedding shard shape/range mismatch"));
        }
        let mut out = self.zeros(&[m.tokens, w.shape[1]])?;
        let (n, d) = (int(m.tokens)?, int(w.shape[1])?);
        let start = u32::try_from(start).map_err(backend)?;
        let rows = u32::try_from(shard.end - shard.start).map_err(backend)?;
        let fun = self.function("embedding_shard")?;
        let cfg = Self::flat(out.data.len())?;
        // SAFETY: validated IDs/range/shard dimensions; masked IDs never index local weights.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&w.data)
                .arg(&m.ids)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&d)
                .arg(&start)
                .arg(&rows)
                .launch(cfg)
        }
        .map_err(backend)?;
        self.reduce_checked(out)
    }
    /// Fixed rank ordering for 3+ ranks: NCCL ring owner/algorithm selection otherwise
    /// changes FP32 summation order with token count or row placement. A BF16 rounding tie
    /// can amplify that difference through later layers. Two-term sums commute exactly.
    pub(crate) fn reduce_f32(&self, data: &mut CudaSlice<f32>) -> Result<()> {
        let Some(comm) = &self.communicator else {
            return Ok(());
        };
        if self.tp.size() <= 2 {
            return comm.all_reduce(data);
        }
        let count = data.len();
        let mut ranked = self
            .stream
            .alloc_zeros::<f32>(elems(&[self.tp.size(), count])?)
            .map_err(backend)?;
        comm.all_gather(data, &mut ranked)?;
        self.sum_ranked(&ranked, data, self.tp.size())
    }
    fn sum_ranked(
        &self,
        ranked: &CudaSlice<f32>,
        data: &mut CudaSlice<f32>,
        ranks: usize,
    ) -> Result<()> {
        if ranks == 0
            || ranked.context().as_ref() != self.ctx.as_ref()
            || data.context().as_ref() != self.ctx.as_ref()
            || data.len().checked_mul(ranks) != Some(ranked.len())
        {
            return Err(invalid("ranked sum shape/context mismatch"));
        }
        let count = data.len();
        let (n, ranks) = (int(count)?, int(ranks)?);
        let fun = self.function("sum_ranks")?;
        // SAFETY: gathered [rank,element] is fully initialized; each thread owns one
        // in-range output element, summing ranks in the same order for every token row.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(ranked)
                .arg(data)
                .arg(&n)
                .arg(&ranks)
                .launch(Self::flat(count)?)
        }
        .map_err(backend)?;
        Ok(())
    }
    pub(crate) fn cast_bf16(&self, data: &CudaSlice<f32>, shape: &[usize]) -> Result<Tensor> {
        if data.context().as_ref() != self.ctx.as_ref() || elems(shape)? != data.len() {
            return Err(invalid("FP32 cast size/context mismatch"));
        }
        let mut out = self.zeros(shape)?;
        let n = int(data.len())?;
        let fun = self.function("to_bf16")?;
        // SAFETY: equal validated lengths, output is exclusive, stream guards order buffers.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(data)
                .arg(&mut out.data)
                .arg(&n)
                .launch(Self::flat(data.len())?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    pub(crate) fn reduce_checked(&self, x: Tensor) -> Result<Tensor> {
        self.check(&x)?;
        if self.communicator.is_none() {
            return Ok(x);
        }
        let mut data = self
            .stream
            .alloc_zeros::<f32>(x.data.len())
            .map_err(backend)?;
        let n = int(x.data.len())?;
        let fun = self.function("to_float")?;
        // SAFETY: equal lengths; one exclusive FP32 output element per thread.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&mut data)
                .arg(&n)
                .launch(Self::flat(x.data.len())?)
        }
        .map_err(backend)?;
        self.reduce_f32(&mut data)?;
        self.cast_bf16(&data, &x.shape)
    }
    pub(crate) fn gather_logits_checked(&self, x: Logits, vocab: usize) -> Result<Logits> {
        if x.data.context().as_ref() != self.ctx.as_ref()
            || x.cols != self.tp.vocab(vocab)?.padded_rows
        {
            return Err(invalid("logits shard shape/context mismatch"));
        }
        let Some(comm) = &self.communicator else {
            return Ok(x);
        };
        let mut ranked = self
            .stream
            .alloc_zeros::<f32>(elems(&[self.tp.size(), x.rows, x.cols])?)
            .map_err(backend)?;
        comm.all_gather(&x.data, &mut ranked)?;
        let mut data = self
            .stream
            .alloc_zeros::<f32>(elems(&[x.rows, vocab])?)
            .map_err(backend)?;
        let (rows, v, local) = (int(x.rows)?, int(vocab)?, int(x.cols)?);
        let fun = self.function("gather_logits")?;
        let cfg = Self::flat(data.len())?;
        // SAFETY: all-gather produced rank-major [rank,row,local_vocab]; kernel transposes
        // to [row,vocab], excludes padded tokens, and writes each output exactly once.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&ranked)
                .arg(&mut data)
                .arg(&rows)
                .arg(&v)
                .arg(&local)
                .launch(cfg)
        }
        .map_err(backend)?;
        Ok(Logits {
            data,
            rows: x.rows,
            cols: vocab,
        })
    }
    pub(crate) fn route_checked(
        &self,
        x: &Tensor,
        k: usize,
        norm: bool,
    ) -> Result<Vec<ExpertAssignment>> {
        self.check(x)?;
        if x.shape.len() != 2 || k == 0 || k > x.shape[1] {
            return Err(invalid("invalid MoE router shape/top-k"));
        }
        let (rows, experts) = (x.shape[0], x.shape[1]);
        let count = elems(&[rows, k])?;
        let mut ids = self.stream.alloc_zeros::<u32>(count).map_err(backend)?;
        let mut weights = self.stream.alloc_zeros::<bf16>(count).map_err(backend)?;
        let (e, topk, norm) = (int(experts)?, int(k)?, i32::from(norm));
        let fun = self.function("route")?;
        // SAFETY: one CTA owns each router row; expert reads and top-k outputs are bounded.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&mut ids)
                .arg(&mut weights)
                .arg(&e)
                .arg(&topk)
                .arg(&norm)
                .launch(Self::rows(rows, 32)?)
        }
        .map_err(backend)?;
        let ids = self.stream.clone_dtoh(&ids).map_err(backend)?;
        let weights = self.stream.clone_dtoh(&weights).map_err(backend)?;
        let mut assignments: Vec<_> = (0..experts)
            .map(|expert| ExpertAssignment {
                expert,
                rows: vec![],
                weights: vec![],
            })
            .collect();
        for row in 0..rows {
            let mut seen = HashSet::new();
            for j in 0..k {
                let expert = ids[row * k + j] as usize;
                let weight = weights[row * k + j].to_f32();
                if expert >= experts || !seen.insert(expert) || !weight.is_finite() || weight < 0. {
                    return Err(backend("invalid/nonfinite MoE routing result"));
                }
                assignments[expert].rows.push(row as u32);
                assignments[expert].weights.push(weight);
            }
        }
        Ok(assignments
            .into_iter()
            .filter(|a| !a.rows.is_empty())
            .collect())
    }
    pub(crate) fn gather_rows_checked(&self, x: &Tensor, rows: &[u32]) -> Result<Tensor> {
        self.check(x)?;
        if x.shape.len() != 2 || rows.is_empty() || rows.iter().any(|&r| r as usize >= x.shape[0]) {
            return Err(invalid("gather row out of range"));
        }
        let indices = self.stream.clone_htod(rows).map_err(backend)?;
        let mut out = self.zeros(&[rows.len(), x.shape[1]])?;
        let (n, d) = (int(rows.len())?, int(x.shape[1])?);
        let fun = self.function("gather")?;
        let cfg = Self::flat(out.data.len())?;
        // SAFETY: each row index checked; repeated read indices are valid; output is disjoint.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&indices)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&d)
                .launch(cfg)
        }
        .map_err(backend)?;
        Ok(out)
    }
    pub(crate) fn scatter_checked(
        &self,
        out: &mut Accumulator,
        x: &Tensor,
        a: &ExpertAssignment,
    ) -> Result<()> {
        self.check(x)?;
        if out.data.context().as_ref() != self.ctx.as_ref()
            || x.shape != [a.rows.len(), out.cols]
            || a.weights.len() != a.rows.len()
            || a.rows.iter().any(|&r| r as usize >= out.rows)
            || a.rows.iter().collect::<HashSet<_>>().len() != a.rows.len()
            || a.weights.iter().any(|&w| !w.is_finite() || w < 0.)
        {
            return Err(invalid("weighted scatter shape/rows mismatch"));
        }
        let rows = self.stream.clone_htod(&a.rows).map_err(backend)?;
        let values: Vec<_> = a.weights.iter().copied().map(bf16::from_f32).collect();
        let weights = self.stream.clone_htod(&values).map_err(backend)?;
        let (n, d) = (int(a.rows.len())?, int(out.cols)?);
        let fun = self.function("weighted_scatter")?;
        // SAFETY: unique in-range rows within this launch; exclusive accumulator borrow;
        // launches for different experts are ordered on the same stream, so no atomics needed.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&rows)
                .arg(&weights)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&d)
                .launch(Self::flat(x.data.len())?)
        }
        .map_err(backend)?;
        Ok(())
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    #[test]
    #[ignore = "requires NVIDIA CUDA 12.6; run explicitly and under Compute Sanitizer"]
    fn rank_ordered_sum_cancellation_and_bounds() -> Result<()> {
        let b = CudaBackend::new(0)?;
        let terms: Vec<f32> = (0..4)
            .flat_map(|rank| {
                (0..33).map(move |i| match rank {
                    0 => 1e8,
                    1 => (i + 1) as f32,
                    2 => -1e8,
                    _ => 1.,
                })
            })
            .collect();
        let input = b.stream.clone_htod(&terms).map_err(backend)?;
        let mut output = b.stream.alloc_zeros::<f32>(33).map_err(backend)?;
        b.sum_ranked(&input, &mut output, 4)?;
        let got = b.stream.clone_dtoh(&output).map_err(backend)?;
        for i in 0..33 {
            let expected = (0..4).fold(0f32, |v, rank| v + terms[rank * 33 + i]);
            assert_eq!(got[i].to_bits(), expected.to_bits(), "row {i}");
        }
        assert!(b.sum_ranked(&input, &mut output, 3).is_err());
        b.synchronize()?;
        Ok(())
    }
}
