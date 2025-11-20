Tests here are currently standalone MLIR snippets intended for manual runs/
future harnessing. They cover CPS interpreter mask behavior:

- `branch_reconverge.mlir` — multi-lane branch where lane0 takes true, others false; exercises per-lane reconvergence.
- `collective_expected_shrink.mlir` — branch with a barrier in the true arm; other lanes take the false path to ensure expected masks shrink so the barrier does not deadlock.

There is no automated runner wired yet; hook these into a test harness or
exercise via a custom driver as needed.
