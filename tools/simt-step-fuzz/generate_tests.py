#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic SIMT-Step fuzz tests."
    )
    parser.add_argument(
        "--fuzzer",
        default="build/tools/simt-step-fuzz/simt-step-fuzz",
        help="Path to simt-step-fuzz binary",
    )
    parser.add_argument(
        "--out-dir",
        default="fuzz-tests",
        help="Output directory for generated MLIR tests",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of tests")
    parser.add_argument("--lanes", type=int, default=4, help="Lane count")
    parser.add_argument(
        "--subgroup-width", type=int, default=8, help="Subgroup width"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Determinism trials per program (>=2)",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=1,
        help="Base schedule seed for randomized scheduling",
    )
    parser.add_argument(
        "--predicate-buffer",
        action="store_true",
        help="Emit predicate buffer and YAML per test",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0, help="First program seed"
    )
    parser.add_argument(
        "--seed-step", type=int, default=1, help="Seed increment per attempt"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Max attempts before giving up",
    )
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("--trials must be >= 2 to enforce determinism")

    fuzzer = Path(args.fuzzer)
    if not fuzzer.exists():
        sys.stderr.write(f"error: fuzzer not found at {fuzzer}\n")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    count = 0
    attempts = 0
    seed = args.seed_start
    with manifest_path.open("a", encoding="utf-8") as manifest:
        while count < args.count and attempts < args.max_attempts:
            attempts += 1
            validate_cmd = [
                str(fuzzer),
                f"--seed={seed}",
                f"--lanes={args.lanes}",
                f"--subgroup-width={args.subgroup_width}",
                "--run",
                f"--trials={args.trials}",
                f"--schedule-seed={args.schedule_seed}",
                "--random-schedule",
            ]
            if args.predicate_buffer:
                validate_cmd.append("--predicate-buffer")
            validate = run(validate_cmd)
            if validate.returncode != 0:
                seed += args.seed_step
                continue

            gen_cmd = [
                str(fuzzer),
                f"--seed={seed}",
                f"--lanes={args.lanes}",
                f"--subgroup-width={args.subgroup_width}",
                "--print-ir",
            ]
            predicate_yaml = ""
            if args.predicate_buffer:
                predicate_yaml = f"{out_dir}/test_{count:03d}_seed_{seed}.yaml"
                gen_cmd.append("--predicate-buffer")
                gen_cmd.append(f"--predicate-yaml={predicate_yaml}")
            generated = run(gen_cmd)
            if generated.returncode != 0 or not generated.stdout.strip():
                seed += args.seed_step
                continue

            filename = f"test_{count:03d}_seed_{seed}.mlir"
            out_path = out_dir / filename
            out_path.write_text(generated.stdout, encoding="utf-8")
            record = {
                "file": filename,
                "seed": seed,
                "lanes": args.lanes,
                "subgroup_width": args.subgroup_width,
                "trials": args.trials,
                "schedule_seed": args.schedule_seed,
            }
            if predicate_yaml:
                record["predicate_yaml"] = Path(predicate_yaml).name
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            count += 1
            seed += args.seed_step

    if count < args.count:
        sys.stderr.write(
            f"error: generated {count}/{args.count} tests after "
            f"{attempts} attempts\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
