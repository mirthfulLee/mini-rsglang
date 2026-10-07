//! Explicit NCCL SONAME loading: cudarc's generic CUDA loader does not search libnccl.so.2.
use cudarc::nccl::sys::*;
use libloading::Library;
use rsglang_core::{Error, Result};
use std::ffi::{c_int, c_void, OsString};
type Status = ncclResult_t;
pub(super) struct Api {
    _library: Library,
    pub group_start: unsafe extern "C" fn() -> Status,
    pub group_end: unsafe extern "C" fn() -> Status,
    pub mem_alloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> Status,
    pub mem_free: unsafe extern "C" fn(*mut c_void) -> Status,
    pub version: unsafe extern "C" fn(*mut c_int) -> Status,
    pub unique_id: unsafe extern "C" fn(*mut ncclUniqueId) -> Status,
    pub init: unsafe extern "C" fn(
        *mut ncclComm_t,
        c_int,
        ncclUniqueId,
        c_int,
        *mut ncclConfig_t,
    ) -> Status,
    pub abort: unsafe extern "C" fn(ncclComm_t) -> Status,
    pub asynchronous_error: unsafe extern "C" fn(ncclComm_t, *mut Status) -> Status,
    pub all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        ncclDataType_t,
        ncclRedOp_t,
        ncclComm_t,
        cudaStream_t,
    ) -> Status,
    pub all_gather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        ncclDataType_t,
        ncclComm_t,
        cudaStream_t,
    ) -> Status,
}
fn symbol<F: Copy>(library: &Library, name: &[u8]) -> Result<F> {
    // SAFETY: private callers use the exact NCCL 2.27 ABI signatures above; the API
    // retains the library for the lifetime of all copied function pointers/communicators.
    unsafe { library.get::<F>(name).map(|s| *s) }
        .map_err(|e| Error::Backend(format!("NCCL symbol: {e}")))
}
impl Api {
    pub fn load() -> Result<Self> {
        let candidates: Vec<OsString> = std::env::var_os("RSGLANG_NCCL_LIBRARY")
            .map(|s| vec![s])
            .unwrap_or_else(|| vec!["libnccl.so.2".into(), "libnccl.so".into()]);
        let mut last = String::new();
        for path in candidates {
            // SAFETY: loads the installed NCCL runtime selected by the user or SONAME.
            let library = match unsafe { Library::new(&path) } {
                Ok(l) => l,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            return Ok(Self {
                group_start: symbol(&library, b"ncclGroupStart\0")?,
                group_end: symbol(&library, b"ncclGroupEnd\0")?,
                mem_alloc: symbol(&library, b"ncclMemAlloc\0")?,
                mem_free: symbol(&library, b"ncclMemFree\0")?,
                version: symbol(&library, b"ncclGetVersion\0")?,
                unique_id: symbol(&library, b"ncclGetUniqueId\0")?,
                init: symbol(&library, b"ncclCommInitRankConfig\0")?,
                abort: symbol(&library, b"ncclCommAbort\0")?,
                asynchronous_error: symbol(&library, b"ncclCommGetAsyncError\0")?,
                all_reduce: symbol(&library, b"ncclAllReduce\0")?,
                all_gather: symbol(&library, b"ncclAllGather\0")?,
                _library: library,
            });
        }
        Err(Error::Backend(format!(
            "cannot load NCCL: {last}; set RSGLANG_NCCL_LIBRARY or LD_LIBRARY_PATH"
        )))
    }
}
