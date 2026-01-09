#!/usr/bin/env python3

import http.server
import json
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
FUZZ_BIN = ROOT / "build" / "tools" / "simt-step-fuzz" / "simt-step-fuzz"
RUNNER_BIN = ROOT / "build" / "tools" / "simt-step-runner" / "simt-step-runner"
RAISE_BIN = ROOT / "build" / "tools" / "simt-step-raise" / "simt-step-raise"


def parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class TraceHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/run":
            params = urllib.parse.parse_qs(parsed.query)
            flat = {key: vals[0] for key, vals in params.items() if vals}
            self.handle_run(flat)
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/run":
            self.send_json(404, {"error": "unknown endpoint"})
            return
        length = parse_int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            self.send_json(400, {"error": "empty request body"})
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict):
            self.send_json(400, {"error": "JSON body must be an object"})
            return
        self.handle_run(payload)

    def handle_run(self, params):
        lanes = parse_int(params.get("lanes"), 4)
        seed = parse_int(params.get("seed"), 0)
        program = params.get("program", "richer")
        raise_target = params.get("raise", "none")
        init_yaml = params.get("init_yaml") or ""
        collective_cf = parse_bool(params.get("collective_cf"), False)
        sync_cf = parse_bool(params.get("sync_cf"), False)
        sync_mem = parse_bool(params.get("sync_mem"), False)
        collective_mem = parse_bool(params.get("collective_mem"), False)

        if lanes < 1 or lanes > 64:
            self.send_json(400, {"error": "lanes must be between 1 and 64"})
            return
        if seed < 0:
            self.send_json(400, {"error": "seed must be non-negative"})
            return

        program_map = {
            "richer": "richer",
            "randomized": "randomized",
            "random": "randomized",
            "deterministic": "deterministic",
        }
        if program not in program_map:
            self.send_json(
                400,
                {"error": "program must be one of: richer, randomized, deterministic"},
            )
            return
        if raise_target not in ("none", "hlsl", "glsl", "cuda"):
            self.send_json(
                400,
                {"error": "raise must be one of: none, hlsl, glsl, cuda"},
            )
            return
        if collective_cf and sync_cf:
            self.send_json(
                400,
                {"error": "control flow mode cannot be both sync and collective"},
            )
            return
        if collective_mem and sync_mem:
            self.send_json(
                400,
                {"error": "memory mode cannot be both sync and collective"},
            )
            return

        if not FUZZ_BIN.exists():
            self.send_json(
                500,
                {"error": f"simt-step-fuzz not found at {FUZZ_BIN}"},
            )
            return

        trace_path = None
        ir_path = None
        init_path = None
        raised_text = ""
        raise_error = ""
        init_yaml = init_yaml if isinstance(init_yaml, str) else str(init_yaml)
        use_runner = bool(init_yaml.strip())
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jsonl", prefix="simt-step-trace-", delete=False
            ) as trace_file:
                trace_path = trace_file.name
            ir_text = ""
            if use_runner:
                if not RUNNER_BIN.exists():
                    self.send_json(
                        500,
                        {"error": f"simt-step-runner not found at {RUNNER_BIN}"},
                    )
                    return
                gen_cmd = [
                    str(FUZZ_BIN),
                    f"--lanes={lanes}",
                    f"--seed={seed}",
                    f"--program={program_map[program]}",
                    "--print-ir",
                ]
                gen_result = subprocess.run(
                    gen_cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if gen_result.returncode != 0:
                    self.send_json(
                        500,
                        {
                            "error": "simt-step-fuzz failed",
                            "stderr": gen_result.stderr.strip(),
                        },
                    )
                    return
                ir_text = self.extract_ir(gen_result.stdout)
                if not ir_text:
                    self.send_json(500, {"error": "failed to extract IR"})
                    return
                with tempfile.NamedTemporaryFile(
                    suffix=".mlir", prefix="simt-step-mlir-", delete=False
                ) as ir_file:
                    ir_path = ir_file.name
                    ir_file.write(ir_text.encode("utf-8"))
                with tempfile.NamedTemporaryFile(
                    suffix=".yaml", prefix="simt-step-init-", delete=False
                ) as init_file:
                    init_path = init_file.name
                    init_file.write(init_yaml.encode("utf-8"))
                run_cmd = [
                    str(RUNNER_BIN),
                    ir_path,
                    f"--lanes={lanes}",
                    f"--trace-file={trace_path}",
                    f"--init-file={init_path}",
                ]
                if collective_cf:
                    run_cmd.append("--collective-cf")
                if sync_cf:
                    run_cmd.append("--sync-cf")
                if sync_mem:
                    run_cmd.append("--sync-mem")
                if collective_mem:
                    run_cmd.append("--collective-mem")
                run_result = subprocess.run(
                    run_cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if run_result.returncode != 0:
                    self.send_json(
                        500,
                        {
                            "error": "simt-step-runner failed",
                            "stderr": run_result.stderr.strip(),
                        },
                    )
                    return
            else:
                cmd = [
                    str(FUZZ_BIN),
                    f"--lanes={lanes}",
                    f"--seed={seed}",
                    f"--program={program_map[program]}",
                    "--print-ir",
                    "--run",
                    f"--trace-file={trace_path}",
                ]
                if collective_cf:
                    cmd.append("--collective-cf")
                if sync_cf:
                    cmd.append("--sync-cf")
                if sync_mem:
                    cmd.append("--sync-mem")
                if collective_mem:
                    cmd.append("--collective-mem")
                result = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    self.send_json(
                        500,
                        {
                            "error": "simt-step-fuzz failed",
                            "stderr": result.stderr.strip(),
                        },
                    )
                    return
                ir_text = self.extract_ir(result.stdout)

            trace_text = Path(trace_path).read_text(encoding="utf-8")
            payload = {"trace": trace_text}
            if ir_text:
                payload["ir"] = ir_text
            if raise_target in ("glsl", "cuda"):
                raise_error = f"{raise_target.upper()} raiser not supported yet"
            elif raise_target == "hlsl":
                if use_runner:
                    if not RAISE_BIN.exists():
                        raise_error = f"simt-step-raise not found at {RAISE_BIN}"
                    else:
                        raise_cmd = [
                            str(RAISE_BIN),
                            ir_path,
                        ]
                        raise_result = subprocess.run(
                            raise_cmd,
                            cwd=str(ROOT),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        if raise_result.returncode != 0:
                            raise_error = "raiser failed"
                            stderr = raise_result.stderr.strip()
                            if stderr:
                                raise_error = f"{raise_error}: {stderr}"
                        else:
                            raised_text = raise_result.stdout.strip()
                else:
                    raise_cmd = [
                        str(FUZZ_BIN),
                        f"--lanes={lanes}",
                        f"--seed={seed}",
                        f"--program={program_map[program]}",
                        "--raise-hlsl",
                    ]
                    raise_result = subprocess.run(
                        raise_cmd,
                        cwd=str(ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if raise_result.returncode != 0:
                        raise_error = "raiser failed"
                        stderr = raise_result.stderr.strip()
                        if stderr:
                            raise_error = f"{raise_error}: {stderr}"
                    else:
                        raised_text = raise_result.stdout.strip()
            if raise_error:
                payload["raise_error"] = raise_error
            if raised_text:
                payload["raised"] = raised_text
            self.send_json(200, payload)
        finally:
            if trace_path:
                try:
                    Path(trace_path).unlink()
                except OSError:
                    pass
            if ir_path:
                try:
                    Path(ir_path).unlink()
                except OSError:
                    pass
            if init_path:
                try:
                    Path(init_path).unlink()
                except OSError:
                    pass

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def extract_ir(self, stdout_text):
        lines = stdout_text.splitlines()
        if not lines:
            return ""
        stop_prefixes = ("Wave ", "Memory:", "run failed:")
        ir_lines = []
        for line in lines:
            if line.startswith(stop_prefixes):
                break
            ir_lines.append(line)

        while ir_lines and not ir_lines[-1].strip():
            ir_lines.pop()

        start_idx = None
        for idx, line in enumerate(ir_lines):
            if line.lstrip().startswith("module "):
                start_idx = idx
                break
        if start_idx is not None:
            ir_lines = ir_lines[start_idx:]

        return "\n".join(ir_lines)


def main():
    port = 8000
    if len(sys.argv) > 1:
        port = parse_int(sys.argv[1], port)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), TraceHandler)
    print(f"SIMT-Step Trace Viewer running at http://localhost:{port}/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
