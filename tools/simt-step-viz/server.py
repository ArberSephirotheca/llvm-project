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
            self.handle_run(parsed)
            return
        return super().do_GET()

    def handle_run(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        lanes = parse_int(params.get("lanes", ["4"])[0], 4)
        seed = parse_int(params.get("seed", ["0"])[0], 0)
        program = params.get("program", ["richer"])[0]
        collective_cf = parse_bool(params.get("collective_cf", [None])[0], False)
        sync_cf = parse_bool(params.get("sync_cf", [None])[0], False)
        sync_mem = parse_bool(params.get("sync_mem", [None])[0], False)
        collective_mem = parse_bool(params.get("collective_mem", [None])[0], False)

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
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jsonl", prefix="simt-step-trace-", delete=False
            ) as trace_file:
                trace_path = trace_file.name

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

            trace_text = Path(trace_path).read_text(encoding="utf-8")
            ir_text = self.extract_ir(result.stdout)
            payload = {"trace": trace_text}
            if ir_text:
                payload["ir"] = ir_text
            self.send_json(200, payload)
        finally:
            if trace_path:
                try:
                    Path(trace_path).unlink()
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
