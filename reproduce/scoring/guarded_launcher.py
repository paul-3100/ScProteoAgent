"""Network kill-switch launcher used for the isolated HuRunWen-口径 CLI passes.

The frozen scorer file itself is NOT modified. This launcher (a) hard-blocks every socket
operation, then (b) executes the frozen copy with runpy under its own __main__, so the CLI
behaviour is byte-for-byte the frozen script's behaviour while no network call can succeed.

Usage:  py -3 guarded_launcher.py <frozen_score_full_run.py> [scorer args...]
"""
import runpy
import socket
import sys
from pathlib import Path


class _NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise _NetworkBlocked("network access blocked by scorer_isolated/guarded_launcher.py")


def install_guard() -> None:
    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.socketpair = _blocked
    if hasattr(socket, "create_server"):
        socket.create_server = _blocked


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: guarded_launcher.py <frozen scorer> [args...]")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"frozen scorer not found: {target}")
    install_guard()
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
