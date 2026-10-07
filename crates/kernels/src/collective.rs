//! NCCL resources and abort handling; no CUDA/NCCL handles escape this crate.
use crate::{backend, invalid};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
    nccl::sys,
};
use rsglang_core::{Error, Result};
use rsglang_distributed::TensorParallel;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct Registered {
    handle: usize,
    context: Arc<CudaContext>,
}
struct State {
    aborted: bool,
    ranks: Vec<Option<Registered>>,
}
/// Shared control plane only. Opaque handle addresses are guarded by the mutex:
/// NCCL enqueue/query/abort cannot race communicator destruction. Tensor pointers
/// are never stored here and remain owned by each rank's backend.
pub struct NcclTeam {
    api: crate::nccl_api::Api,
    id: sys::ncclUniqueId,
    size: usize,
    devices: Vec<usize>,
    state: Mutex<State>,
    timeout: Duration,
    version: i32,
}
impl NcclTeam {
    pub fn new(size: usize, timeout: Duration) -> Result<Arc<Self>> {
        Self::new_for_devices(&(0..size).collect::<Vec<_>>(), timeout)
    }
    pub fn new_for_devices(devices: &[usize], timeout: Duration) -> Result<Arc<Self>> {
        rsglang_distributed::validate_devices(devices)?;
        let size = devices.len();
        if size < 2 || size > i32::MAX as usize || timeout.is_zero() {
            return Err(invalid("invalid NCCL team size/timeout"));
        }
        let api = crate::nccl_api::Api::load()?;
        let mut version = 0;
        let mut id = sys::ncclUniqueId { internal: [0; 128] };
        // SAFETY: loaded NCCL ABI; valid distinct host output pointers used synchronously.
        let (vstatus, istatus) = unsafe { ((api.version)(&mut version), (api.unique_id)(&mut id)) };
        if vstatus != sys::ncclResult_t::ncclSuccess || istatus != sys::ncclResult_t::ncclSuccess {
            return Err(backend(format!("NCCL bootstrap: {vstatus:?}, {istatus:?}")));
        }
        if version < 22700 {
            return Err(invalid("tensor parallelism requires NCCL >= 2.27"));
        }
        Ok(Arc::new(Self {
            api,
            id,
            size,
            devices: devices.to_vec(),
            state: Mutex::new(State {
                aborted: false,
                ranks: (0..size).map(|_| None).collect(),
            }),
            timeout,
            version,
        }))
    }
    pub fn version(&self) -> i32 {
        self.version
    }
    pub fn abort(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.aborted = true;
        let ranks: Vec<_> = state.ranks.iter_mut().filter_map(Option::take).collect();
        // Remove all handles before aborting; enqueues/queries now return Stopped.
        // NCCL cleanup of multiple ranks in one process must progress concurrently.
        drop(state);
        std::thread::scope(|scope| {
            for rank in ranks {
                scope.spawn(move || {
                    let _ = rank.context.bind_to_thread();
                    // SAFETY: registry owns a live handle; its mutex excludes enqueue/query/abort.
                    // ncclCommAbort is the NCCL failure cleanup API and releases the communicator.
                    let status = unsafe { (self.api.abort)(rank.handle as sys::ncclComm_t) };
                    if status != sys::ncclResult_t::ncclSuccess {
                        eprintln!("NCCL abort: {status:?}");
                    }
                });
            }
        });
    }
    pub(crate) fn connect(
        self: &Arc<Self>,
        stream: Arc<CudaStream>,
        tp: TensorParallel,
    ) -> Result<Communicator> {
        if tp.size() != self.size || self.devices[tp.rank()] != stream.context().ordinal() {
            return Err(invalid("NCCL rank/device mapping mismatch"));
        }
        stream.context().bind_to_thread().map_err(backend)?;
        let mut cfg = sys::ncclConfig_t {
            size: std::mem::size_of::<sys::ncclConfig_t>(),
            magic: 0xcafebeef,
            version: 22700,
            blocking: 0,
            cgaClusterSize: i32::MIN,
            minCTAs: i32::MIN,
            maxCTAs: i32::MIN,
            netName: std::ptr::null(),
            splitShare: i32::MIN,
            trafficClass: i32::MIN,
            commName: std::ptr::null(),
            collnetEnable: i32::MIN,
            CTAPolicy: i32::MIN,
            shrinkShare: i32::MIN,
            nvlsCTAs: i32::MIN,
        };
        let status = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| backend("poisoned NCCL registry"))?;
            if state.aborted {
                return Err(Error::Stopped);
            }
            if state.ranks[tp.rank()].is_some() {
                return Err(invalid("NCCL rank initialized twice"));
            }
            let mut handle = std::ptr::null_mut();
            // SAFETY: valid output/config pointers and copied NCCL ID; rank is validated.
            // blocking=0 ensures initialization returns and permits peer failure/timeout handling.
            let status = unsafe {
                (self.api.init)(
                    &mut handle,
                    tp.size() as i32,
                    self.id,
                    tp.rank() as i32,
                    &mut cfg,
                )
            };
            if !handle.is_null() {
                state.ranks[tp.rank()] = Some(Registered {
                    handle: handle as usize,
                    context: stream.context().clone(),
                });
            }
            status
        };
        let comm = Communicator {
            team: self.clone(),
            stream,
            tp,
            scratch: Mutex::new(Scratch { ptr: 0, bytes: 0 }),
        };
        comm.complete(status)?;
        Ok(comm)
    }
}
impl Drop for NcclTeam {
    fn drop(&mut self) {
        self.abort();
    }
}

pub(crate) struct Communicator {
    team: Arc<NcclTeam>,
    stream: Arc<CudaStream>,
    tp: TensorParallel,
    scratch: Mutex<Scratch>,
}
struct Scratch {
    ptr: usize,
    bytes: usize,
}
impl Communicator {
    fn call(
        &self,
        f: impl FnOnce(sys::ncclComm_t) -> sys::ncclResult_t,
    ) -> Result<sys::ncclResult_t> {
        self.stream.context().bind_to_thread().map_err(backend)?;
        let state = self
            .team
            .state
            .lock()
            .map_err(|_| backend("poisoned NCCL registry"))?;
        let rank = state.ranks[self.tp.rank()].as_ref().ok_or(Error::Stopped)?;
        Ok(f(rank.handle as sys::ncclComm_t))
    }
    fn enqueue(
        &self,
        f: impl FnOnce(sys::ncclComm_t) -> sys::ncclResult_t,
    ) -> Result<sys::ncclResult_t> {
        self.call(|handle| {
            // SAFETY: matched thread-local group calls in the loaded NCCL ABI. Even a
            // failed enqueue closes the group. Poll the outer group status before issuing
            // subsequent stream operations when nonblocking NCCL launches asynchronously.
            let start = unsafe { (self.team.api.group_start)() };
            if start != sys::ncclResult_t::ncclSuccess {
                return start;
            }
            let operation = f(handle);
            let end = unsafe { (self.team.api.group_end)() };
            if (operation != sys::ncclResult_t::ncclSuccess
                && operation != sys::ncclResult_t::ncclInProgress)
                || end == sys::ncclResult_t::ncclSuccess
            {
                operation
            } else {
                end
            }
        })
    }
    fn complete(&self, mut status: sys::ncclResult_t) -> Result<()> {
        let deadline = Instant::now() + self.team.timeout;
        while status == sys::ncclResult_t::ncclInProgress {
            if Instant::now() >= deadline {
                self.team.abort();
                return Err(backend(format!("NCCL rank {} timed out", self.tp.rank())));
            }
            let mut asynchronous = sys::ncclResult_t::ncclSuccess;
            let checked = self.call(|handle| {
                // SAFETY: call() holds the registry lock for this live handle and valid output.
                unsafe { (self.team.api.asynchronous_error)(handle, &mut asynchronous) }
            })?;
            status = if checked == sys::ncclResult_t::ncclSuccess {
                asynchronous
            } else {
                checked
            };
            if status == sys::ncclResult_t::ncclInProgress {
                std::thread::sleep(Duration::from_micros(100));
            }
        }
        if status != sys::ncclResult_t::ncclSuccess {
            self.team.abort();
            return Err(backend(format!("NCCL rank {}: {status:?}", self.tp.rank())));
        }
        Ok(())
    }
    fn staging(&self, bytes: usize) -> Result<std::sync::MutexGuard<'_, Scratch>> {
        self.stream.context().bind_to_thread().map_err(backend)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| backend("poisoned NCCL staging"))?;
        if bytes > scratch.bytes {
            // All previous uses must finish before freeing or resizing the rank-owned staging.
            self.stream.synchronize().map_err(backend)?;
            if scratch.ptr != 0 {
                // SAFETY: no pending stream access; uniquely owned ncclMemAlloc allocation.
                let status = unsafe { (self.team.api.mem_free)(scratch.ptr as _) };
                if status != sys::ncclResult_t::ncclSuccess {
                    return Err(backend(format!("NCCL free: {status:?}")));
                }
                scratch.ptr = 0;
                scratch.bytes = 0;
            }
            let capacity = bytes
                .checked_next_power_of_two()
                .ok_or_else(|| invalid("NCCL staging overflow"))?
                .max(1 << 20);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: valid output, current rank context; NCCL allocates transport-compatible
            // memory independently of cudarc's stream-ordered tensor allocation pool.
            let status = unsafe { (self.team.api.mem_alloc)(&mut ptr, capacity) };
            if status != sys::ncclResult_t::ncclSuccess {
                return Err(backend(format!("NCCL allocation: {status:?}")));
            }
            scratch.ptr = ptr as usize;
            scratch.bytes = capacity;
        }
        Ok(scratch)
    }
    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: callers validate both allocation bounds and hold their ownership/stream
        // guards. All copies and collectives use the same stream, preserving staging reuse order.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())
        }
        .map_err(backend)
    }
    pub(crate) fn all_reduce(&self, x: &mut CudaSlice<f32>) -> Result<()> {
        if x.context().as_ref() != self.stream.context().as_ref() {
            return Err(invalid("NCCL buffer context mismatch"));
        }
        let bytes = x
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("NCCL size overflow"))?;
        let scratch = self.staging(bytes)?;
        let count = x.len();
        let (ptr, _guard) = x.device_ptr_mut(&self.stream);
        self.copy(scratch.ptr as u64, ptr, bytes)?;
        let status = self.enqueue(|handle| {
            // SAFETY: exclusive staging of sufficient size, supported in-place reduction;
            // every rank submits matching counts. Staging remains live through stream completion.
            unsafe {
                (self.team.api.all_reduce)(
                    scratch.ptr as _,
                    scratch.ptr as _,
                    count,
                    sys::ncclDataType_t::ncclFloat32,
                    sys::ncclRedOp_t::ncclSum,
                    handle,
                    self.stream.cu_stream() as _,
                )
            }
        })?;
        self.complete(status)?;
        self.copy(ptr, scratch.ptr as u64, bytes)
    }
    pub(crate) fn all_gather(&self, x: &CudaSlice<f32>, out: &mut CudaSlice<f32>) -> Result<()> {
        if x.context().as_ref() != self.stream.context().as_ref()
            || out.context().as_ref() != self.stream.context().as_ref()
            || x.len().checked_mul(self.tp.size()) != Some(out.len())
        {
            return Err(invalid("NCCL gather buffer size/context mismatch"));
        }
        let output_bytes = out
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("NCCL size overflow"))?;
        let source_bytes = x
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("NCCL size overflow"))?;
        // Separate input/output, with 256-byte alignment for the output region. This avoids
        // aliasing the send region with receives and simplifies ownership/bounds checks.
        let offset = source_bytes
            .checked_add(255)
            .ok_or_else(|| invalid("NCCL size overflow"))?
            & !255;
        let bytes = offset
            .checked_add(output_bytes)
            .ok_or_else(|| invalid("NCCL size overflow"))?;
        let scratch = self.staging(bytes)?;
        let source = scratch.ptr;
        let destination = scratch.ptr + offset;
        let (src, _sg) = x.device_ptr(&self.stream);
        let (dst, _dg) = out.device_ptr_mut(&self.stream);
        self.copy(source as u64, src, x.len() * 4)?;
        let status = self.enqueue(|handle| {
            // SAFETY: disjoint validated send/receive regions in the owned staging allocation.
            // The receive region covers every rank, guarded against concurrent reuse.
            unsafe {
                (self.team.api.all_gather)(
                    source as _,
                    destination as _,
                    x.len(),
                    sys::ncclDataType_t::ncclFloat32,
                    handle,
                    self.stream.cu_stream() as _,
                )
            }
        })?;
        self.complete(status)?;
        self.copy(dst, destination as u64, output_bytes)
    }
    pub(crate) fn abort(&self) {
        self.team.abort();
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        // Engine/RankGroup abort communicators before dropping buffers on failure. Normal
        // steps synchronize before replying. Keep the context and library alive until freeing.
        self.team.abort();
        let _ = self.stream.context().bind_to_thread();
        if self.stream.synchronize().is_ok() {
            let scratch = self.scratch.get_mut().unwrap_or_else(|e| e.into_inner());
            if scratch.ptr != 0 {
                // SAFETY: uniquely owned allocation with no outstanding GPU accesses.
                unsafe { (self.team.api.mem_free)(scratch.ptr as _) };
            }
        }
    }
}
