# Gemma 4 Mac Sandbox

Mac-local experiment harness for the opt-in Gemma 4 paged per-layer embedding
loader.

Build:

```sh
swift build --package-path tools/gemma4-mac-sandbox -c release
```

Command-line SwiftPM cannot compile MLX's Metal shader bundle. The sandbox
locates a known host copy, verifies SHA-256
`71f2ad788d86f29486315b55835ce6ed17ceb9b0302f6b80429306340dca28d1`,
and places a symlink named `mlx.metallib` beside the executable. Pass an
explicit source with `--metallib /path/to/default.metallib` when needed.

Commands:

```sh
Gemma4MacSandbox self-test --model-dir /path/to/pinned/snapshot
Gemma4MacSandbox run --mode resident --color red --model-dir /path/to/pinned/snapshot --output /tmp/run.json
Gemma4MacSandbox compare --receipts /tmp/gemma4-paged-ple-comparison
```

`self-test` compares the five fixed quantized rows against MLX's normal loader.
`run` performs real deterministic image inference and records raw tokens,
load/prefill/generation timing, peak process memory, loader component bytes,
and pager counters. Use `run-comparison.sh` to produce three fresh-process runs
for each mode/color pair before calling `compare`.

This Mac-only experiment does not authorize phone deployment.
