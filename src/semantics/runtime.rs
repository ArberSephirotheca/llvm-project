//! Runtime intrinsics for compiled-oracle path (to be linked/called from LLVM lowering).
#[inline(always)]
pub fn simt_wave_all(active: u64, pred: u64) -> bool { (active & !pred) == 0 }
#[inline(always)]
pub fn simt_ballot(active: u64, pred: u64) -> u64 { active & pred }
#[inline(always)]
pub fn simt_barrier(_scope: u32, _sem: u32) {}
#[inline(always)]
pub fn simt_fence(_scope: u32, _sem: u32) {}
#[inline(always)]
pub fn simt_shuffle(val: u32, src: u32, _w: u32, lanes: &[u32]) -> u32 {
    // demo: use provided lane array as storage; real impl will differ
    let idx = src.min(lanes.len() as u32 - 1) as usize;
    lanes[idx]
}
