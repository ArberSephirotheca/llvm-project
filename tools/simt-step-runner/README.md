# simt-step-runner

Run an existing SIMT-Step MLIR module with the SimpleProgramRunner.

## Build

From the repo root:

```
cmake --build build
```

Binary: `build/tools/simt-step-runner/simt-step-runner`

## Usage

```
simt-step-runner <input.mlir> [options]
```

Options:
- `--func=<name>`: entry function to run (default: `main`)
- `--lanes=<n>`: number of lanes to execute (default: `4`)
- `--print-ir`: print parsed IR before running
- `--trace-file=<path>`: write trace JSONL for the visualizer
- `--collective-cf` / `--sync-cf`: control-flow execution mode
- `--collective-mem` / `--sync-mem`: memory execution mode

## Example

```
build/tools/simt-step-runner/simt-step-runner my_module.mlir \
  --func=main --lanes=4 --print-ir \
  --collective-cf --collective-mem \
  --trace-file=tools/simt-step-viz/trace.jsonl
```
