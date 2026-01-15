# simt-cuda-test design

This document captures the initial design for an AmberScript-like CUDA test
framework that runs a single kernel per script. The focus is small, strict,
and predictable: define buffers, compile inline CUDA C, launch, and validate.

## Goals
- AmberScript-style text input (line-based commands).
- Inline CUDA C kernel source (raw code only).
- Single kernel per script (one compile + one launch).
- Deterministic validation with float tolerances.
- Simple CLI for batch testing.

## Non-goals (v0)
- Multiple kernels per script.
- Graphics pipelines or textures.
- Kernel includes or file-based code blocks.
- Complex data layouts (structs, matrices, images).

## Script shape

Commands are line oriented. Comments use `#` at line start.
Buffers must be declared before they are referenced by FILL/INIT/BIND/EXPECT.

Supported commands (v0):
- `BUFFER <name> TYPE <type> SIZE <n>`
- `FILL <name> <value>`
- `INIT <name> <index> <value>`
- `KERNEL [<name>]` ... `ENDKERNEL`
- `BIND <name> ARG <n>`
- `BIND CONST <value> TYPE <type> ARG <n>`
- `LAUNCH GRID <x> <y> <z> BLOCK <x> <y> <z>`
- `EXPECT <name> <index> <value> [ABS_TOL <a>] [REL_TOL <r>]`
- `EXPECT_RANGE <name> <start> <end> <value> [ABS_TOL <a>] [REL_TOL <r>]`

Types (v0):
- `i32`, `u32`, `f32`

Ranges are inclusive (start/end are both checked).

### Kernel rules
- If `KERNEL` has no name, the kernel name is `main`.
- If `KERNEL <name>` is provided, that name is used.
- The code block is passed to NVRTC verbatim (raw code only). No wrapper is
  injected by the runner.
- The kernel signature must match the `BIND` list in argument order.
  - `BUFFER` binds become pointer args (e.g., `i32` -> `int*`, `f32` -> `float*`).
  - `CONST` binds become scalar args (type required).

### Example
```
BUFFER buf0 TYPE i32 SIZE 16
FILL buf0 0
INIT buf0 3 42

KERNEL
extern "C" __global__ void main(int* buf0, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) buf0[tid] += 1;
}
ENDKERNEL

BIND buf0 ARG 0
BIND CONST 16 TYPE i32 ARG 1

LAUNCH GRID 1 1 1 BLOCK 16 1 1

EXPECT buf0 3 43
EXPECT_RANGE buf0 0 15 1
```

## Execution model
1. Parse script into AST.
2. Allocate buffers on device; apply FILL and INIT.
3. Compile kernel with NVRTC to PTX.
4. Load module, get kernel entry by name.
5. Bind arguments and launch.
6. Read back buffers and validate.

## Validation rules
- Integers are exact by default.
- Floats use per-expect tolerances:
  - `ABS_TOL` and `REL_TOL`, both optional for `EXPECT`.
  - `EXPECT_RANGE` is exact by default; tolerances are optional for floats.
- Mismatch reports: buffer name, index, expected, actual, tolerances.

## Parser notes
- Line-based tokenizer; keywords in uppercase.
- `KERNEL` to `ENDKERNEL` captures raw text verbatim.
- Strict error handling: unknown commands or malformed args abort parsing.

## CLI (planned)
```
simt-cuda-test <script.ambercuda> [--device N] [--arch sm_80] [--dump-ptx]
```

## File layout (planned)
- `tools/simt-cuda-test/`
  - `main.cpp` (CLI + driver)
  - `Parser.{h,cpp}`
  - `CudaBackend.{h,cpp}`
  - `Ast.{h,cpp}`
  - `DESIGN.md` (this file)
