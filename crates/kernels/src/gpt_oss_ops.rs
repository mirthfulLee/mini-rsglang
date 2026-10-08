use super::*;

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_checked(
        &self,
        q: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        m: &Metadata,
        qheads: usize,
        kvheads: usize,
        dim: usize,
        sinks: Option<&Tensor>,
        window: usize,
    ) -> Result<Tensor> {
        for x in [q, kc, vc] {
            self.check(x)?;
        }
        self.check_meta(m)?;
        if !(1..=256).contains(&dim)
            || kvheads == 0
            || !qheads.is_multiple_of(kvheads)
            || q.shape != [m.tokens, qheads * dim]
            || kc.shape != [m.pages, m.page_size, kvheads, dim]
            || vc.shape != kc.shape
        {
            return Err(invalid("paged GQA shape mismatch"));
        }
        if let Some(sink) = sinks {
            self.check(sink)?;
            if sink.shape != [qheads] {
                return Err(invalid("attention sink shape mismatch"));
            }
        }
        let sink_data = &sinks.unwrap_or(q).data;
        let window = int(window)?;
        let has_sink = i32::from(sinks.is_some());
        let mut out = self.zeros(&q.shape)?;
        let qh = int(qheads)?;
        let kh = int(kvheads)?;
        let d = int(dim)?;
        let p = int(m.page_size)?;
        let tw = int(m.table_width)?;
        let scale_q = (dim as f64).powf(-0.25) as f32;
        let fun = self.function(if sinks.is_some() {
            "attention_gpt"
        } else {
            "attention"
        })?;
        let cfg = LaunchConfig {
            grid_dim: (m.tokens as u32, qheads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: CTA(row,head) exclusively owns its output. Metadata guarantees page coverage
        // through each causal position. Shared query <=256 and every CTA uses 4 full warps.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&q.data)
                .arg(&kc.data)
                .arg(&vc.data)
                .arg(&m.positions)
                .arg(&m.sequences)
                .arg(&m.table)
                .arg(&mut out.data)
                .arg(&qh)
                .arg(&kh)
                .arg(&d)
                .arg(&p)
                .arg(&tw)
                .arg(&scale_q)
                .arg(sink_data)
                .arg(&window)
                .arg(&has_sink)
                .launch(cfg)
        }
        .map_err(backend)?;
        Ok(out)
    }
}

impl CudaBackend {
    pub(crate) fn gpt_swiglu_checked(&self, x: &Tensor, limit: f32) -> Result<Tensor> {
        self.check(x)?;
        if x.shape.len() != 2 || !x.shape[1].is_multiple_of(2) || !limit.is_finite() || limit <= 0.0
        {
            return Err(invalid("GPT-OSS SwiGLU shape/limit mismatch"));
        }
        let mut out = self.zeros(&[x.shape[0], x.shape[1] / 2])?;
        let n = int(out.data.len())?;
        let fun = self.function("gpt_swiglu")?;
        // SAFETY: interleaved pairs cover exactly twice the output element count.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&limit)
                .launch(Self::flat(n as usize)?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_scaled_checked(
        &self,
        x: &mut Tensor,
        m: &Metadata,
        heads: usize,
        dim: usize,
        frequencies: &[f32],
        magnitude: f32,
    ) -> Result<()> {
        self.check(x)?;
        self.check_meta(m)?;
        if dim == 0
            || !dim.is_multiple_of(2)
            || x.shape != [m.tokens, heads * dim]
            || frequencies.len() != dim / 2
            || frequencies.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !magnitude.is_finite()
            || magnitude <= 0.0
        {
            return Err(invalid("scaled RoPE shape/frequencies mismatch"));
        }
        let frequencies = self.stream.clone_htod(frequencies).map_err(backend)?;
        let (n, h, d) = (int(m.tokens)?, int(heads)?, int(dim)?);
        let fun = self.function("gpt_rope")?;
        // SAFETY: checked metadata and dimension pairs; tensor is exclusively borrowed.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&mut x.data)
                .arg(&m.positions)
                .arg(&frequencies)
                .arg(&n)
                .arg(&h)
                .arg(&d)
                .arg(&magnitude)
                .launch(Self::flat(m.tokens * heads * (dim / 2))?)
        }
        .map_err(backend)?;
        Ok(())
    }
    pub(crate) fn mxfp4_checked(
        &self,
        blocks: &[u8],
        scales: &[u8],
        shape: &[usize],
    ) -> Result<Tensor> {
        let count = elems(shape)?;
        if shape.len() != 2
            || !shape[1].is_multiple_of(32)
            || blocks.len() != count / 2
            || scales.len() != count / 32
        {
            return Err(invalid("MXFP4 block/scale shape mismatch"));
        }
        let blocks = self.stream.clone_htod(blocks).map_err(backend)?;
        let scales = self.stream.clone_htod(scales).map_err(backend)?;
        let mut out = self.zeros(shape)?;
        let n = int(count)?;
        let fun = self.function("mxfp4")?;
        // SAFETY: two nibbles per byte and one scale per 32 checked output elements.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&blocks)
                .arg(&scales)
                .arg(&mut out.data)
                .arg(&n)
                .launch(Self::flat(count)?)
        }
        .map_err(backend)?;
        Ok(out)
    }
}

impl CudaBackend {
    pub(crate) fn linear_bias_checked(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        reduce: bool,
    ) -> Result<Tensor> {
        let (rows, cols, _) = self.linear_shape(x, weight)?;
        self.check(bias)?;
        if bias.shape != [cols] {
            return Err(invalid("linear bias shape mismatch"));
        }
        let mut partial = self.logits(x, weight)?;
        if reduce {
            self.reduce_f32(&mut partial.data)?;
        }
        let mut out = self.zeros(&[rows, cols])?;
        let (n, width) = (int(rows * cols)?, int(cols)?);
        let fun = self.function("linear_bias")?;
        // SAFETY: checked GEMM output and column bias; each output element is exclusive.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&partial.data)
                .arg(&bias.data)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&width)
                .launch(Self::flat(rows * cols)?)
        }
        .map_err(backend)?;
        Ok(out)
    }
}
