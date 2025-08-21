# FastSiftGPU

A performance-oriented, CUDA-accelerated SIFT implementation used as a sandbox to learn, apply, and document modern GPU optimization techniques.

> Based on and originally derived from **pitzer/SiftGPU**.  
> Repository: https://github.com/pitzer/SiftGPU

---

## Why this project exists

SIFT is a classic, widely used feature extractor. That makes it a great testbed for practical CUDA optimization: the workload includes multi-scale image processing, neighborhood operations, and non-trivial memory access patterns. The goal here is to:

- **Learn and apply** real GPU optimization methods (not just theory).
- **Measure and report** the effect of each change with clear before/after data.
- **Modernize** legacy pieces (e.g., static textures) without losing correctness.

---

## Compatibility

- **CUDA**: Tested with **CUDA 11.8**.  
  The current implementation uses **static texture references**, which are supported up to CUDA 11.x.  
---

## Benchmarking & reporting plan

To keep results credible and reproducible, I will be sharing following after each optimization step:

- **Metrics to report:**
  - Kernel time (per stage and end-to-end)
  - SM occupancy, achieved occupancy
  - DRAM throughput & L2 hit rate
  - Warp execution efficiency / branch efficiency
  - Registers per thread, shared memory per block
- **Artifacts:** For each optimization step:
  - Short write-up (what changed, why it should help)
  - Nsight Compute screenshots or metric tables
  - Before/after numbers with percent change
  - Any correctness deltas (e.g., keypoint counts, descriptor diffs)

---


## License

- Inherit and comply with the original SiftGPU license terms.  
  Any additional code herein follows the same license unless stated otherwise.

---

## Disclaimer

This repository is primarily an educational and experimental vehicle for GPU optimization.  
