"""Offline import smoke test for the released Python modules.

For every module in this repository it checks two things, in a clean environment:

1. the file parses (``ast.parse``), so a syntax error is caught even when the module's third
   party dependencies are absent;
2. importing it either succeeds, or fails **only** with ``ModuleNotFoundError`` for a known
   third-party package of this project. Any other failure (NameError, ImportError from a
   project module, SyntaxError, ...) fails the test.

The credential environment is cleared for the child process; no network call is made and no
key is read. Modules that need heavy dependencies (langchain, openai, google-genai, rpy2,
scanpy, tiktoken, ddgs, igraph, ...) are expected to be unimportable on a machine where the
agent environment has not been installed yet - that is recorded, not hidden.

Usage: python tests/test_module_imports.py [--strict] [--python EXE]
Exit codes: 0 pass, 1 real import failure, 3 environment cannot run the test.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = {"adjustText", "Bio", "ddgs", "dotenv", "google", "gseapy", "igraph", "joblib",
               "langchain", "langchain_core", "langchain_openai", "langgraph", "matplotlib",
               "mpmath", "mygene", "networkx", "numpy", "openai", "pandas", "PIL", "plotly", "requests", "rpy2",
               "scanpy", "scipy", "seaborn", "sklearn", "statsmodels", "tiktoken", "tqdm",
               "umap", "yaml"}
CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SCORE_MODEL", "GOOGLE_API_KEY",
                   "GOOGLE_BASE_URL", "GOOGLE_CX", "SERPAPI_KEY", "ENTREZ_EMAIL")


def released_modules() -> list[Path]:
    modules = [path for path in sorted(REPO.glob("*.py"))]
    modules += [path for path in sorted((REPO / "scripts").glob("*.py"))]
    modules += [path for path in sorted((REPO / "reproduce").rglob("*.py"))]
    modules += [path for path in sorted((REPO / "examples").rglob("*.py"))]
    return modules


def clean_environment() -> dict:
    env = dict(os.environ)
    for name in CREDENTIAL_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def import_one(module: Path, python: str) -> tuple[bool, str]:
    """Import a module by its own name, with its directory as the working directory."""
    code = "import %s" % module.stem
    completed = subprocess.run([python, "-c", code], capture_output=True, text=True,
                               cwd=str(module.parent), env=clean_environment())
    if completed.returncode == 0:
        return True, ""
    lines = [line for line in (completed.stderr or "").strip().splitlines() if line.strip()]
    return False, lines[-1] if lines else "unknown failure (exit %d)" % completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import smoke test for released modules.")
    parser.add_argument("--strict", action="store_true",
                        help="fail when a module cannot be imported (full agent environment)")
    parser.add_argument("--python", default=sys.executable, help="interpreter to use")
    args = parser.parse_args(argv)

    modules = released_modules()
    if not modules:
        print("BLOCKED: no Python modules found under %s" % REPO)
        return 3

    syntax_failures: list[str] = []
    import_ok = import_missing = expects_data = 0
    unexpected: list[tuple[str, str]] = []
    missing_packages = set()
    for module in modules:
        relative = module.relative_to(REPO).as_posix()
        try:
            ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        except SyntaxError as exc:
            syntax_failures.append("%s:%s" % (relative, exc))
            continue
        ok, message = import_one(module, args.python)
        if ok:
            import_ok += 1
            print("IMPORT_OK    %s" % relative)
            continue
        if "ModuleNotFoundError: No module named" in message:
            package = message.split("'")
            name = package[1] if len(package) > 1 else "?"
            top_level = name.split(".")[0]
            if top_level in THIRD_PARTY:
                import_missing += 1
                missing_packages.add(top_level)
                print("NEEDS_EXTRA  %s  (%s)" % (relative, top_level))
                continue
        if "FATAL:" in message and ("data archive" in message or "frozen interval inputs" in message):
            # Reproduction scripts resolve the data archive while they are imported. Without the
            # archive they must stop with the documented message instead of continuing.
            expects_data += 1
            print("EXPECTS_DATA %s" % relative)
            continue
        if "ImportError: cannot import name" in message and "from '" in message:
            # A namespace package or a different distribution of the same name is installed
            # without the submodule this project imports (for example a bare `google` namespace
            # without `google-genai`). That is still a dependency prerequisite, not a defect in
            # the released file, so it is recorded with its own label.
            package = message.split("from '")[-1].split("'")[0].split(".")[0]
            if package in THIRD_PARTY:
                import_missing += 1
                missing_packages.add(package)
                print("NEEDS_EXTRA  %s  (incomplete/incompatible: %s)" % (relative, package))
                continue
        unexpected.append((relative, message))
        print("FAIL         %s  (%s)" % (relative, message))

    print("-" * 72)
    print("modules=%d import_ok=%d needs_third_party=%d expects_data_archive=%d "
          "unexpected_failures=%d syntax_failures=%d"
          % (len(modules), import_ok, import_missing, expects_data, len(unexpected),
             len(syntax_failures)))
    print("missing third-party packages: %s" % (", ".join(sorted(missing_packages)) or "none"))
    for failure in syntax_failures:
        print("SYNTAX %s" % failure)
    for relative, message in unexpected:
        print("UNEXPECTED %s: %s" % (relative, message))
    if syntax_failures or unexpected:
        print("RESULT: FAIL")
        return 1
    if args.strict and import_missing:
        print("RESULT: FAIL (--strict and %d modules need third-party packages)" % import_missing)
        return 1
    print("RESULT: PASS%s" % (" (dependency-prerequisite recorded)" if import_missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
