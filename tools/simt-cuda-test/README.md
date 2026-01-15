# simt-cuda-test

A small AmberScript-like CUDA test runner. It parses a single-kernel script,
compiles the inline CUDA C with NVRTC, launches the kernel, and validates buffer
results with optional float tolerances.

## Build

This target requires the CUDA toolkit (NVRTC + driver API). If you want to skip
building it, configure with `-DSIMT_STEP_BUILD_CUDA_TEST=OFF`.

## Usage

```
simt-cuda-test <script.cuda> [--device N] [--arch sm_80] [--dump-ptx]
```

- `--device N`: CUDA device index (default: 0)
- `--arch sm_80`: passed to NVRTC as `--gpu-architecture=sm_80`
- `--dump-ptx`: print the generated PTX to stdout

## Script format (v0)

Comments use `#` at line start. Commands are line-based. One kernel per file.
Buffers must be declared before they are referenced by FILL/INIT/BIND/EXPECT.

Supported commands:
- `BUFFER <name> TYPE <type> SIZE <n>`
- `FILL <name> <value>`
- `INIT <name> <index> <value>`
- `KERNEL [<name>]` ... `ENDKERNEL`
- `BIND <name> ARG <n>`
- `BIND CONST <value> TYPE <type> ARG <n>`
- `LAUNCH GRID <x> <y> <z> BLOCK <x> <y> <z>`
- `EXPECT <name> <index> <value> [ABS_TOL <a>] [REL_TOL <r>]`
- `EXPECT_RANGE <name> <start> <end> <value> [ABS_TOL <a>] [REL_TOL <r>]`

Types:
- `i32`, `u32`, `f32`

Ranges are inclusive (start/end are both checked).

### Kernel rules
- The code inside `KERNEL` is passed to NVRTC verbatim (raw code only).
- If `KERNEL` has no name, the kernel name is `main`.
- The kernel signature must match the `BIND` list, ordered by `ARG` index.
  - `BUFFER` binds become pointer arguments of the declared element type.
  - `CONST` binds become scalar arguments of the declared type.

### Example

```
BUFFER buf0 TYPE i32 SIZE 16
FILL buf0 0
INIT buf0 3 42

KERNEL
extern "C" __global__ void main(int* buf0, int n) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n) buf0[tid] += 1;
}
ENDKERNEL

BIND buf0 ARG 0
BIND CONST 16 TYPE i32 ARG 1

LAUNCH GRID 1 1 1 BLOCK 16 1 1

EXPECT buf0 3 43
EXPECT_RANGE buf0 0 15 1
```
