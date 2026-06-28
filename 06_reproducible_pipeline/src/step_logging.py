from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

from project_config import LOG_DIR, PIPELINE_ROOT, PROJECT_ROOT


def configure_console() -> None:
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def make_log_path(step_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{stamp}_{step_name}.log"


def run_logged_action(step_name: str, action: Callable[[Callable[[str], None]], None]) -> Path:
    """Run a Python callable and mirror all step messages into a log file."""
    configure_console()
    log_path = make_log_path(step_name)
    log_path.write_text("", encoding="utf-8-sig")

    def emit(message: str = "") -> None:
        print(message)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")

    emit(f"Step   : {step_name}")
    emit(f"Python : {sys.executable}")
    emit(f"Project: {PROJECT_ROOT}")
    emit(f"Log    : {log_path}")
    emit("")
    try:
        action(emit)
    except Exception as exc:
        emit("")
        emit(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    emit("")
    emit("DONE")
    return log_path


def run_script(step_name: str, script_path: Path, key_outputs: Iterable[Path] = ()) -> Path:
    """Run another Python script and capture stdout/stderr into a step log."""
    configure_console()
    log_path = make_log_path(step_name)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    header = [
        f"Step   : {step_name}",
        f"Script : {script_path}",
        f"Python : {sys.executable}",
        f"Project: {PROJECT_ROOT}",
        f"Log    : {log_path}",
        "",
    ]
    with log_path.open("w", encoding="utf-8-sig") as log:
        for line in header:
            print(line)
            log.write(line + "\n")
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PIPELINE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        log.write(f"\nExit code: {return_code}\n")
    if return_code != 0:
        raise SystemExit(f"{script_path.name} failed with exit code {return_code}. See log: {log_path}")
    show_key_outputs(key_outputs, log_path)
    print("\nDONE")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nDONE\n")
    return log_path


def show_key_outputs(paths: Iterable[Path], log_path: Path) -> None:
    lines = ["", "Key outputs:"]
    for path in paths:
        status = "OK" if path.exists() else "MISSING"
        try:
            label = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(path)
        lines.append(f"  {status:7} {label}")
    for line in lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as log:
        for line in lines:
            log.write(line + "\n")


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
