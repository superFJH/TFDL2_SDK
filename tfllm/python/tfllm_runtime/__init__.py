"""SDK location and console dispatch; importing this module never imports Torch."""
import os
from pathlib import Path
import re
import shlex
import sys


def sdk_root():
    return Path(__file__).resolve().parent / "sdk"


def launch():
    name = Path(sys.argv[0]).name
    program = sdk_root() / "bin" / name
    if not program.is_file() or not os.access(program, os.X_OK):
        raise SystemExit(f"TFLLM command is missing or not executable: {program}")
    with program.open("rb") as stream:
        first_line = stream.readline(512)
    # tfllm/tfllm_profile use '#!/usr/bin/env python3'. Use the console script's
    # interpreter, even when a venv command is invoked without activating it.
    if first_line.startswith(b"#!"):
        words = shlex.split(first_line[2:].decode("utf-8", errors="replace"))
        if any(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(word).name) for word in words):
            os.execv(sys.executable, [sys.executable, str(program), *sys.argv[1:]])
    os.execv(str(program), [str(program), *sys.argv[1:]])
