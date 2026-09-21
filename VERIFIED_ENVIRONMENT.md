# Verified environment

Everything on this page was measured while the release candidate was assembled. Nothing is
extrapolated to other operating systems, interpreters or machines, and anything that could not be
run is listed as *not executed* rather than implied.

## Host

| item | value |
|---|---|
| operating system | Windows 10 Pro, version 10.0.19045, 64-bit |
| shell | PowerShell |
| R | **not installed** on this host, so the rpy2-based limma/enrichment paths were not exercised |
| git | used read-only (hashing and diffs only); no repository was initialised, and nothing was committed or pushed |
| secret scanner | no gitleaks on this host; the scan uses the local rule set and records that coverage gap |

## Interpreters used

| label | interpreter | version | relevant packages |
|---|---|---|---|
| bare | the system CPython launcher, no third-party packages | 3.14.3 | none of the packages below; used to prove that `--help` and the scoring replay need nothing |
| anaconda | an Anaconda base interpreter | 3.13.9 | numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.5, mpmath 1.3.0, matplotlib 3.10.6 |
| venv_agent | a one-off virtual environment created from the anaconda interpreter | 3.13.9 | the agent stack, 110 distributions, 30,040 files / 752.1 MB, `pip check` clean |
| venv_empty | a virtual environment created with `--without-pip`, used only as a negative case | 3.13.9 | nothing |

Python 3.13 was chosen for the installed environment because the agent stack resolves there; 3.14
runs everything that needs no third-party package.

## Installing the agent stack

`requirements-agent.txt` is installed in one command. The first attempt failed inside pip's
resolution, not on the machine: `rpy2` resolves to 3.6.8, which requires
`rpy2-rinterface>=3.6.7`, and that release has no Windows wheel for CPython 3.13, so pip fell back
to a source build that needs a C toolchain. The fix follows the release contract: the R bridge is an
**optional extra**, not part of the base install.

    # base install (about 110 distributions, 323 s here)
    .venv/Scripts/python -m pip install -r requirements-agent.txt
    # optional R bridge, only if you have R and need the limma / clusterProfiler paths
    .venv/Scripts/python -m pip install "rpy2==3.6.6" "rpy2-rinterface==3.6.6"

`mpmath` was added to the list in this revision: `reproduce/go_background/c4_verify.py`
imports it at module level, so its absence would have blocked the GO verifier.

`requirements-lock.txt` records the exact versions of the environment that was installed this
way (110 pinned distributions, measured 2026-09-20 on this host).

## What was executed, and with which interpreter

| check | interpreter | outcome |
|---|---|---|
| frozen environment record (`requirements-lock.txt`) | venv_agent | RECORD: the 110 pinned distributions of the verified environment (CPython 3.13.9, Windows 10 19045, measured 2026-09-20); a reader-facing record, not an installer |
| `main_agent.py --help`, no key, no `.env`, no network, no third-party package | bare | **PASS**, exit 0 (before the deferred-import change the same command exited 1 with `ModuleNotFoundError: langchain_core`) |
| the same `--help` in the installed environment | venv_agent | PASS, output identical to the production engine's help text |
| side effects of `--help` | venv_agent | PASS: no run directory, file and directory counts unchanged |
| `tests/test_module_imports.py --strict` | venv_agent | PASS: 44 modules checked, 40 imported, 4 that expect the data archive (`reproduce/go_background/c1`..`c4`, whose offline behaviour is to stop with their documented message), 0 missing packages, 0 unexpected failures; the count follows the tree (the tree before this round printed 43/39, the round that installed the venv recorded 40/36) (raw: `modules=44 import_ok=40 needs_third_party=0 expects_data_archive=4 unexpected_failures=0 syntax_failures=0`, "missing third-party packages: none") |
| the same test under a dependency-free interpreter | venv_empty | PASS as a negative case: exit 1 with `needs_third_party=16` |
| `tests/test_no_credentials_no_network.py` | venv_agent | PASS: 12 checks, 0 failures (network blocked and controlled, sentinel `.env` not loaded, clear missing-key error, no client constructed) |
| `reproduce/scoring/prepare_scoring_inputs.py` | bare | PASS: 1,461 run files and 55 dataset files assembled, every file hash-matched against the archive |
| `reproduce/scoring/run_replay.py` against `scores/HW_RULE_SCORES.tsv` | bare | PASS: compared 88/88, missing 0, maximum absolute difference **0.0** on `rule_total` and the six dimensions, 88/88 report hashes checked |
| the same driver on an explicit subset | bare | SUBSET_PASS, coverage 16/88, reported as a subset and never as the 88-cell result |
| `tests/test_replay_completeness.py` | bare | PASS: 33 completeness and safety cases, 0 failed |
| `reproduce/intervals/recompute_paired_ci.py` + `ci_verify.py` | anaconda | PASS: all seven intervals reproduced, verifier 10/10 |
| `reproduce/go_background/c1` -> `c4`, writing to a separate working directory | anaconda | PASS: 12 families, 62,584 term rows, historical gate 90/90, fixed 24-position display 15/5/0/4, verifier verdict PASS; 8 of the 10 delivered tables byte-identical, two differences documented in the step README |
| re-deriving the three GMT files from the MSigDB bundle | bare | PASS: byte-identical to the files the analysis used, which confirms the resource release |
| `reproduce/cases/` (five themes) | anaconda | 69 checks PASS, 5 BLOCKED; the blocked items are listed with the paths that were searched |
| `reproduce/figures/plot_panels.py --all` | bare | PASS: six statistical panels written as PNG (450 dpi), PDF and SVG |
| `scripts/validate_dataset_inputs.py` on the demo and on four negative fixtures | bare | PASS: exit 0 on the demo, exit 2 on each broken fixture |
| `examples/saved_results/read_saved_table.py` | bare | PASS |
| `cases_reverified_finalgate` (placeholder) | <filled in by the round owner> | <filled in by the round owner> |

The `cases_reverified_finalgate` row is a deliberate placeholder: the round owner writes the
final case re-verification outcome there, and no outcome is guessed here.

## What was *not* executed here

* **A live agent run.** It needs the user's own key and a paid endpoint. No model call was made, no
  key was read, and no key was validated, so `LIVE_AGENT_DEMO_VERIFIED` is false.
* **Any path that calls R through rpy2** (the limma-based differential analysis and the
  clusterProfiler enrichment). R is not installed on this host and the bridge is an optional extra.
* **Non-Windows hosts.** No Linux or macOS run was attempted, so no platform support is claimed.
* **Hardware and runtime sizing.** No CPU/GPU requirement, memory ceiling or wall-clock budget was
  measured for a full analysis; the individual numbers in the table above are single measurements on
  this host.
* **The assembled manuscript figures.** They were laid out by hand in the authors' files; only the
  statistical panels are redrawn from the frozen tables.
* **The case items that are not reproduced.** They are recorded in `reproduce/cases/README.md` and
  in the component status table of `docs/REPRODUCTION.md`, with the paths that were searched. One
  element of one frozen case table stays a reported difference rather than a passing check; the
  table names it.
* **A live agent run, the R bridge and non-Windows hosts** are listed above as not executed. They
  do not block the offline reproduction of the reported numbers, and the package does not claim
  that they were verified.

## Determinism notes

The scoring replay, the paired intervals and the GO chain are arithmetic over frozen inputs: the
replay reproduced all 88 cells exactly, the interval endpoints reproduced at 17 significant digits
across two interpreters (3.14.3 / numpy 2.5.2 and 3.13.9 / numpy 2.3.5), and the GO tables differ
from the delivered ones only in a recorded path column and in eight last-digit strings of a stored
adjusted-P column. The case scripts compare discrete identity exactly and floats at a pre-registered
relative tolerance of 1e-7, and additionally at the resolution stored in each frozen file. The
figures use fixed styles and a fixed jitter seed (20260918).

