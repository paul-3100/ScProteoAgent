#!/usr/bin/env python3
"""Security negative tests for the released package: credentials and network isolation.

The checks run against an isolated copy of this repository created under the system temporary
directory; the working tree is never used, no real .env, no real key and no provider endpoint
are read.

What is asserted
(a) 'python main_agent.py --help' exits 0 while the copy contains a .env with synthetic sentinel
    credentials; the sentinel values never appear in the output, are never loaded into the child
    environment (a different sentinel is preset in the environment, so loading .env with
    override=True would be visible), and --help does not import llm at all.
(b) With no credentials available, the first attempt to obtain a client raises a clear error that
    names the missing configuration, and no network call is attempted.
(c) Non-live use (help and the import sweep of the released modules) makes no network request.
(d) 'tests/test_module_imports.py --strict' exits non-zero when a dependency is missing, so a
    missing package can never be reported as a pass.

How the isolation is built
- the copy is created under the system temporary directory and removed afterwards (--keep keeps
  it),
- sitecustomize.py in the copy blocks DNS, TCP connect, urllib and http.client calls and appends
  one line per blocked attempt to the file named by V24_2_NETBLOCK_LOG; a dedicated control check
  proves the block is effective, so an empty log is meaningful,
- credential variables are removed from every child environment before the copy's .env is written
  with sentinel values.

Usage: python tests/test_no_credentials_no_network.py [--python EXE] [--keep]
Exit codes: 0 all checks passed, 1 a check failed, 3 the harness could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "SCORE_MODEL",
                   "GOOGLE_API_KEY", "GOOGLE_BASE_URL", "GOOGLE_CX", "SERPAPI_KEY",
                   "ENTREZ_EMAIL")
ENVIRONMENT_SENTINEL = "V24_2_SENTINEL_ENVIRONMENT_DO_NOT_USE"
ENVIRONMENT_SENTINEL_BASE = "V24_2_SENTINEL_ENVIRONMENT_BASE_DO_NOT_USE"
DOTENV_SENTINEL = "V24_2_SENTINEL_DOTENV_DO_NOT_USE"
DOTENV_SENTINEL_BASE = "V24_2_SENTINEL_DOTENV_BASE_DO_NOT_USE"
NETBLOCK_ENV = "V24_2_NETBLOCK_LOG"

SITECUSTOMIZE_SOURCE = '''"""Network block installed by tests/test_no_credentials_no_network.py."""
import os
import socket


def _record(kind, detail):
    path = os.environ.get("V24_2_NETBLOCK_LOG", "")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(kind + " " + detail + chr(10))
    except Exception:
        pass


class NetworkBlocked(RuntimeError):
    pass


def _blocked(kind):
    def _call(*args, **kwargs):
        _record(kind, repr(args[:2])[:200])
        raise NetworkBlocked(
            "network access is blocked by the v24_2 security negative test: " + kind)
    return _call


socket.getaddrinfo = _blocked("socket.getaddrinfo")
socket.create_connection = _blocked("socket.create_connection")
socket.socket.connect = _blocked("socket.socket.connect")
socket.socket.connect_ex = _blocked("socket.socket.connect_ex")
socket.socket.sendto = _blocked("socket.socket.sendto")

try:
    import urllib.request as _urlopen_module
    _urlopen_module.urlopen = _blocked("urllib.request.urlopen")
except Exception:
    pass

try:
    import http.client as _http_client
    _http_client.HTTPConnection.connect = _blocked("http.client.HTTPConnection.connect")
    _http_client.HTTPSConnection.connect = _blocked("http.client.HTTPSConnection.connect")
except Exception:
    pass
'''

HELP_DRIVER_SOURCE = '''import json
import os
import runpy
import sys

repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
sys.argv = ["main_agent.py", "--help"]
result = {"exit": None, "error": None, "llm_imported": False, "tools_imported": False}
try:
    runpy.run_path(os.path.join(repo, "main_agent.py"), run_name="__main__")
    result["exit"] = "returned"
except SystemExit as exc:
    result["exit"] = 0 if exc.code is None else exc.code
except BaseException as exc:
    result["exit"] = "exception"
    result["error"] = type(exc).__name__ + ": " + str(exc)[:300]
result["llm_imported"] = "llm" in sys.modules
result["tools_imported"] = "tools" in sys.modules
result["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
result["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "")
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
'''

CLIENT_DRIVER_SOURCE = '''import json
import sys

repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
result = {"step": "start", "client_type": None, "error_type": None, "error": None}
try:
    import llm
    result["step"] = "llm_imported"
    from llm import LLM
    result["client_type"] = type(LLM).__name__
    result["step"] = "client_obtained"
    LLM.invoke("ping")
    result["step"] = "invoke_returned"
except BaseException as exc:
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)[:400]
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
'''

IMPORT_DRIVER_SOURCE = '''import importlib
import json
import sys

repo, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, repo)
status = {}
for name in names:
    try:
        importlib.import_module(name)
        status[name] = "ok"
    except BaseException as exc:
        status[name] = type(exc).__name__ + ": " + str(exc)[:200]
with open(out, "w", encoding="utf-8") as handle:
    json.dump(status, handle)
'''

NETBLOCK_CONTROL_SOURCE = '''import json
import socket
import sys
import urllib.request

out = sys.argv[1]
result = {}
try:
    socket.create_connection(("example.invalid", 80), timeout=2)
    result["socket.create_connection"] = "NOT_BLOCKED"
except BaseException as exc:
    result["socket.create_connection"] = type(exc).__name__
try:
    urllib.request.urlopen("http://example.invalid/", timeout=2)
    result["urllib.request.urlopen"] = "NOT_BLOCKED"
except BaseException as exc:
    result["urllib.request.urlopen"] = type(exc).__name__
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
'''


class Harness:
    """Owns the isolated copy, the child environments and the check bookkeeping."""

    def __init__(self, python: str, keep: bool):
        self.python = python
        self.keep = keep
        self.scratch = Path(tempfile.mkdtemp(prefix="v24_2_secneg_"))
        self.copy = self.scratch / "repo"
        self.drivers = self.scratch / "drivers"
        self.checks: list[tuple[str, bool, str]] = []

    # -- setup ------------------------------------------------------------------------------
    def build(self) -> None:
        self.copy.mkdir(parents=True)
        self.drivers.mkdir(parents=True)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for item in sorted(REPO.iterdir()):
            if item.name in {".git", "__pycache__", ".env"}:
                continue
            target = self.copy / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=ignore)
            else:
                shutil.copy2(item, target)
        (self.copy / "sitecustomize.py").write_text(SITECUSTOMIZE_SOURCE, encoding="utf-8")
        for name, source in (("help_driver.py", HELP_DRIVER_SOURCE),
                             ("client_driver.py", CLIENT_DRIVER_SOURCE),
                             ("import_driver.py", IMPORT_DRIVER_SOURCE),
                             ("netblock_control.py", NETBLOCK_CONTROL_SOURCE)):
            (self.drivers / name).write_text(source, encoding="utf-8")

    def write_dotenv(self, with_sentinels: bool) -> None:
        dotenv = self.copy / ".env"
        if with_sentinels:
            dotenv.write_text(
                "OPENAI_API_KEY=%s\nOPENAI_BASE_URL=%s\nOPENAI_MODEL=\nSCORE_MODEL=\n"
                % (DOTENV_SENTINEL, DOTENV_SENTINEL_BASE), encoding="utf-8")
        elif dotenv.exists():
            dotenv.unlink()

    # -- execution --------------------------------------------------------------------------
    def environment(self, netlog: Path | None = None) -> dict:
        env = dict(os.environ)
        for name in CREDENTIAL_VARS:
            env.pop(name, None)
        env["PYTHONPATH"] = str(self.copy)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if netlog is None:
            env.pop(NETBLOCK_ENV, None)
        else:
            env[NETBLOCK_ENV] = str(netlog)
        return env

    def run(self, args: list[str], netlog: Path | None = None, sentinel_environment: bool = False,
            timeout: int = 300) -> subprocess.CompletedProcess:
        env = self.environment(netlog)
        if sentinel_environment:
            env["OPENAI_API_KEY"] = ENVIRONMENT_SENTINEL
            env["OPENAI_BASE_URL"] = ENVIRONMENT_SENTINEL_BASE
        return subprocess.run([self.python] + args, cwd=str(self.copy), env=env,
                              capture_output=True, text=True, timeout=timeout)

    def log_attempts(self, netlog: Path) -> list[str]:
        if not netlog.exists():
            return []
        return [line for line in netlog.read_text(encoding="utf-8").splitlines() if line.strip()]

    def record(self, check_id: str, ok: bool, evidence: str) -> None:
        self.checks.append((check_id, ok, evidence))
        print("%-38s %s  %s" % (check_id, "PASS" if ok else "FAIL", evidence))

    def cleanup(self) -> None:
        if not self.keep:
            shutil.rmtree(self.scratch, ignore_errors=True)
        else:
            print("kept scratch directory: %s" % self.scratch)


def module_names() -> list[str]:
    names = []
    for path in sorted(REPO.glob("*.py")):
        if path.stem not in {"sitecustomize", "__init__"}:
            names.append(path.stem)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credential and network isolation negative tests.")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used for the child processes (the full agent environment)")
    parser.add_argument("--keep", action="store_true", help="keep the isolated copy for inspection")
    args = parser.parse_args(argv)

    harness = Harness(python=args.python, keep=args.keep)
    try:
        harness.build()
    except Exception as exc:
        print("BLOCKED: could not build the isolated copy: %s" % exc)
        return 3

    final = "FAIL"
    try:
        # -- control: the network block itself must work ------------------------------------
        control_log = harness.scratch / "control.log"
        control = harness.run([str(harness.drivers / "netblock_control.py"),
                               str(harness.scratch / "control.json")], netlog=control_log)
        control_result = json.loads((harness.scratch / "control.json").read_text(encoding="utf-8"))
        blocked = [value for value in control_result.values() if value == "NetworkBlocked"]
        control_attempts = harness.log_attempts(control_log)
        harness.record("z1_netblock_control", len(blocked) == 2 and len(control_attempts) >= 2,
                       "blocked=%d attempts_logged=%d exit=%d"
                       % (len(blocked), len(control_attempts), control.returncode))

        # -- (a) help with a sentinel .env ---------------------------------------------------
        harness.write_dotenv(with_sentinels=True)
        help_log = harness.scratch / "help.log"
        plain = harness.run(["main_agent.py", "--help"], netlog=help_log, sentinel_environment=True)
        output = (plain.stdout or "") + (plain.stderr or "")
        harness.record("a1_help_exit_zero", plain.returncode == 0,
                       "exit=%d" % plain.returncode)
        sentinel_seen = [name for name, value in (("dotenv", DOTENV_SENTINEL),
                                                  ("dotenv_base", DOTENV_SENTINEL_BASE),
                                                  ("environment", ENVIRONMENT_SENTINEL),
                                                  ("environment_base", ENVIRONMENT_SENTINEL_BASE))
                         if value in output]
        harness.record("a2_help_no_sentinel_output", not sentinel_seen,
                       "sentinel tokens in output: %s" % (", ".join(sentinel_seen) or "none"))

        driver_out = harness.scratch / "help.json"
        harness.run([str(harness.drivers / "help_driver.py"), str(harness.copy), str(driver_out)],
                    netlog=help_log, sentinel_environment=True)
        help_result = json.loads(driver_out.read_text(encoding="utf-8"))
        harness.record("a3_help_driver_exit_zero", help_result["exit"] == 0,
                       "exit=%s error=%s" % (help_result["exit"], help_result["error"]))
        not_overridden = (help_result["OPENAI_API_KEY"] == ENVIRONMENT_SENTINEL
                          and help_result["OPENAI_BASE_URL"] == ENVIRONMENT_SENTINEL_BASE)
        harness.record("a4_dotenv_not_loaded", not_overridden,
                       "OPENAI_API_KEY=%s" % help_result["OPENAI_API_KEY"])
        harness.record("a5_help_does_not_import_llm", help_result["llm_imported"] is False,
                       "llm_imported=%s tools_imported=%s"
                       % (help_result["llm_imported"], help_result["tools_imported"]))
        harness.record("a6_help_no_network", not harness.log_attempts(help_log),
                       "attempts=%d" % len(harness.log_attempts(help_log)))

        # -- (b) missing key: clear error, no network ----------------------------------------
        harness.write_dotenv(with_sentinels=False)
        client_log = harness.scratch / "client.log"
        client_json = harness.scratch / "client.json"
        harness.run([str(harness.drivers / "client_driver.py"), str(harness.copy), str(client_json)],
                    netlog=client_log)
        client_result = json.loads(client_json.read_text(encoding="utf-8"))
        clear_error = (client_result["error_type"] == "RuntimeError"
                       and "OPENAI_API_KEY" in (client_result["error"] or "")
                       and client_result["client_type"] == "MissingLLM")
        harness.record("b1_missing_key_clear_error", clear_error,
                       "client=%s error=%s: %s" % (client_result["client_type"],
                                                   client_result["error_type"],
                                                   (client_result["error"] or "")[:120]))
        harness.record("b2_missing_key_no_network", not harness.log_attempts(client_log),
                       "attempts=%d" % len(harness.log_attempts(client_log)))

        # -- (c) non-live module usage does not touch the network ----------------------------
        import_log = harness.scratch / "imports.log"
        import_json = harness.scratch / "imports.json"
        names = module_names()
        harness.run([str(harness.drivers / "import_driver.py"), str(harness.copy),
                     str(import_json)] + names, netlog=import_log)
        imported = json.loads(import_json.read_text(encoding="utf-8"))
        failed = {name: status for name, status in imported.items() if status != "ok"}
        harness.record("c1_nonlive_no_network", not harness.log_attempts(import_log),
                       "modules=%d attempts=%d"
                       % (len(imported), len(harness.log_attempts(import_log))))
        harness.record("c2_nonlive_modules_import", not failed,
                       "failed=%s" % (json.dumps(failed, sort_keys=True)[:300] if failed else "none"))

        # -- (d) --strict must fail when a dependency is missing -----------------------------
        bare = harness.scratch / "venv_bare"
        created = subprocess.run([args.python, "-m", "venv", "--without-pip", str(bare)],
                                 capture_output=True, text=True, timeout=300)
        bare_python = bare / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if created.returncode != 0 or not bare_python.exists():
            harness.record("d1_strict_import_fails_without_dependency", False,
                           "could not create the dependency-free interpreter (exit=%d)"
                           % created.returncode)
        else:
            # The dependency-free interpreter must run the test itself, so the harness helper
            # (which always prefixes the full-environment interpreter) is not used here.
            strict = subprocess.run([str(bare_python), "tests/test_module_imports.py", "--strict"],
                                    cwd=str(harness.copy), env=harness.environment(),
                                    capture_output=True, text=True, timeout=300)
            strict_output = (strict.stdout or "") + (strict.stderr or "")
            (harness.scratch / "strict_import.log").write_text(strict_output, encoding="utf-8")
            summary = re.search(r"needs_third_party=(\d+)", strict_output)
            missing = int(summary.group(1)) if summary else -1
            tail = " | ".join(line.strip() for line in strict_output.strip().splitlines()[-2:])[:220]
            harness.record("d1_strict_import_fails_without_dependency",
                           strict.returncode != 0 and missing > 0 and "RESULT: FAIL" in strict_output,
                           "exit=%d needs_third_party=%s tail=%s" % (strict.returncode, missing, tail))
        final = "PASS" if all(ok for _, ok, _ in harness.checks) else "FAIL"
    finally:
        print("-" * 72)
        print("checks=%d failed=%d" % (len(harness.checks),
                                       sum(1 for _, ok, _ in harness.checks if not ok)))
        print("RESULT: %s" % final)
        harness.cleanup()
    return 0 if final == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
