#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time


def parse_device_ids(raw: str) -> list[str]:
    ids: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid device id: {part}")
        ids.append(part)
    if not ids:
        raise ValueError("No device ids provided.")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run amd-smi-query alongside an app and stop when the app exits."
    )
    parser.add_argument(
        "-d",
        "--devices",
        required=True,
        help='Device id(s), e.g. "0" or "0,1,2"',
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="Prefix name for output files.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Path to amd-smi-query binary (default: ./amd-smi-query next to this script).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1,
        help="Sampling interval in ms (default: 1).",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run after -- (e.g., -- python3 app.py).",
    )
    return parser


def truncate_partial_last_line(path: str) -> None:
    # If the process is killed while writing, the last CSV row can be partial.
    # Trim any unterminated tail line after the writer has exited.
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return

        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return

        chunk_size = 4096
        pos = size
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            data = f.read(read_size)
            idx = data.rfind(b"\n")
            if idx != -1:
                f.truncate(pos + idx + 1)
                return

        # No newline found anywhere; safest behavior is to clear the file.
        f.truncate(0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        parser.error('Command is required and must follow "--".')
    cmd = args.cmd[1:]
    if not cmd:
        parser.error("No command provided after --.")

    try:
        device_ids = parse_device_ids(args.devices)
    except ValueError as exc:
        parser.error(str(exc))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = args.query or os.path.join(script_dir, "amd-smi-query")
    if not os.path.isfile(query_path) or not os.access(query_path, os.X_OK):
        parser.error(f"amd-smi-query not found or not executable: {query_path}")

    query_procs: list[subprocess.Popen] = []
    output_files = []
    output_paths: list[str] = []

    for dev in device_ids:
        out_name = f"profiling_result_{args.prefix}_{dev}.csv"
        out_path = os.path.abspath(out_name)
        out_f = open(out_path, "w", buffering=1)
        output_files.append(out_f)
        output_paths.append(out_path)
        proc = subprocess.Popen(
            [query_path, "-d", dev, "-i", str(args.interval_ms)],
            stdout=out_f,
            stderr=subprocess.STDOUT,
        )
        query_procs.append(proc)

    app_proc = subprocess.Popen(cmd)

    def shutdown():
        for proc in query_procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        t0 = time.time()
        for proc in query_procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.0, 2.0 - (time.time() - t0)))
                except subprocess.TimeoutExpired:
                    proc.kill()
        for proc in query_procs:
            if proc.poll() is None:
                proc.wait()
        for f in output_files:
            f.flush()
            os.fsync(f.fileno())
            f.close()
        for path in output_paths:
            truncate_partial_last_line(path)

    try:
        app_proc.wait()
    except KeyboardInterrupt:
        app_proc.send_signal(signal.SIGINT)
        try:
            app_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            app_proc.kill()
    finally:
        shutdown()

    return app_proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
