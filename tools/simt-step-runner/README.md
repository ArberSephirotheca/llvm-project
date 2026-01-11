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
- `--subgroup-width=<n>`: subgroup width (default: `8`)
- `--print-ir`: print parsed IR before running
- `--trace-file=<path>`: write trace JSONL for the visualizer
- `--init=bufN:idx:value`: initialize buffer entry before running (repeatable)
- `--init-file=<path>`: YAML file for buffer initialization
- `--collective-cf` / `--sync-cf`: control-flow execution mode
- `--collective-mem` / `--sync-mem`: memory execution mode

## Example

```
build/tools/simt-step-runner/simt-step-runner my_module.mlir \
  --func=main --lanes=4 --print-ir \
  --init=buf0:0:42 --init=buf0:1:7 \
  --init-file=tools/simt-step-runner/init.yaml \
  --collective-cf --collective-mem \
  --trace-file=tools/simt-step-viz/trace.jsonl
```

## init.yaml format

```yaml
buffers:
  - buffer: buf1
    size: 8
    fill: 0
    entries:
      - { index: 0, value: 7 }
      - { index: 5, value: 9 }
  - buffer: buf0
    entries:
      - { index: 3, value: 42 }
```

Rules:
- `buffer` is `bufN` or `argN` (argument index in `@main`).
- `fill` requires `size` and fills indices `[0, size)`.
- `entries` override `fill`.
