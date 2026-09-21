"""Independent verification of the recomputed paired intervals.

Runs in a fresh process. Recomputes the endpoint table from the frozen inputs,
compares it with the delivered table, replays the sampling indices through an
explicit loop for spot checks, and checks the frozen-mean invariants.

Usage: py -3 reproduce/intervals/ci_verify.py [--data-dir DIR] [--out-dir DIR]
                                              [--verify-dir DIR] [--tag NAME]
Output: <verify-dir>/CI_VERIFY_<tag>.tsv   (default <data-dir>/verification)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from recompute_paired_ci import (METHOD_ORDER, NBOOT, RULES_NAMES, SEED,  # noqa: E402
                                 configure, find_input, load_rules, resolve_data_dir, RULES)
TABLES = SCRIPTS
VER = SCRIPTS

EXPECTED_MEANS = [22.054545454545448, 25.709090909090910, 26.945454545454540,
                  30.209090909090914, 30.418181818181820, 31.154545454545463,
                  33.427272727272730]


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Verify the recomputed paired intervals.")
    parser.add_argument("--data-dir", default=None, help="frozen interval inputs")
    parser.add_argument("--out-dir", default=None,
                        help="directory holding the recomputed tables "
                             "(default: <cwd>/intervals_out, outside the data archive)")
    parser.add_argument("--verify-dir", default=None,
                        help="directory for the verification report "
                             "(default: <out-dir>/verification)")
    parser.add_argument("--tag", default="run", help="tag written into the report filename")
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    configure(data_dir, args.out_dir)
    global TABLES, VER, RULES
    TABLES = Path(args.out_dir).resolve() if args.out_dir else Path.cwd() / "intervals_out"
    VER = Path(args.verify_dir).resolve() if args.verify_dir else TABLES / "verification"
    if TABLES == data_dir or data_dir in TABLES.parents or VER == data_dir or data_dir in VER.parents:
        raise SystemExit(
            ("FATAL: --out-dir/--verify-dir resolve inside the read-only data archive %s. The "
             "archive is an input: choose a working directory outside it (the default is "
             "<cwd>/intervals_out).") % data_dir)
    RULES = find_input(data_dir, RULES_NAMES) or (data_dir / RULES_NAMES[0])
    tag = args.tag
    rows = []

    def check(name, ok, detail=""):
        rows.append([name, "PASS" if ok else "FAIL", detail])
        return ok

    datasets, methods, value = load_rules()
    check("frozen_cells_88", len(value) == 88, "unique (study, method) cells")
    comparators = METHOD_ORDER[1:]
    diff = np.array([[value[(d, "scProteoAgent")] - value[(d, m)] for m in comparators]
                     for d in datasets], dtype=np.float64)
    means = diff.mean(axis=0)
    check("paired_means_unchanged",
          all(abs(means[i] - EXPECTED_MEANS[i]) < 1e-12 for i in range(7)),
          "|".join("%.6f" % x for x in means))

    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, 11, size=(NBOOT, 11), dtype=np.int64)
    boot = diff[idx, :].mean(axis=1)
    q = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")

    stored = list(csv.DictReader((TABLES / "paired_ci_recomputed.tsv").open(encoding="utf-8"),
                                 delimiter="\t"))
    ok_low = ok_high = True
    for i, r in enumerate(stored):
        ok_low &= float(r["ci_low"]) == float(format(q[0][i], ".17g"))
        ok_high &= float(r["ci_high"]) == float(format(q[1][i], ".17g"))
    check("endpoints_reproduced_bitwise", ok_low and ok_high,
          "fresh-process recomputation equals the delivered 17-digit endpoints")

    saved = np.load(TABLES / "paired_ci_indices.npy")
    idx_sha = hashlib.sha256(saved.tobytes()).hexdigest()
    env = json.loads((TABLES / "environment.json").read_text(encoding="utf-8"))
    check("index_matrix_identical",
          bool((saved == idx).all())
          and idx_sha == env["index_sha256"] == hashlib.sha256(idx.tobytes()).hexdigest(),
          "sha256=%s shape=%s" % (idx_sha[:16], list(saved.shape)))
    check("index_byte_layout",
          str(saved.dtype) == "int64" and saved.flags["C_CONTIGUOUS"],
          "int64 C-order (little-endian on this host)")

    loop_targets = [0, 1, 2, 6]
    worst = 0.0
    for j in loop_targets:
        col = diff[:, j]
        for i in range(NBOOT):
            s = 0.0
            for k in range(11):
                s += col[idx[i, k]]
            worst = max(worst, abs(s / 11.0 - boot[i, j]))
    check("loop_spot_check_1e12", worst <= 1e-12,
          "left-multiply spot check over %s (max |diff| = %.3g)"
          % (", ".join(comparators[j] for j in loop_targets), worst))

    old = list(csv.DictReader((TABLES / "paired_ci_old_new.tsv").open(encoding="utf-8"),
                              delimiter="\t"))
    deltas = [max(abs(float(r["delta_low"])), abs(float(r["delta_high"]))) for r in old]
    check("old_new_shift_bounded", max(deltas) < 0.2,
          "max |new-old| = %.4f points (expected small; new endpoints differ by design)"
          % max(deltas))
    check("display_rounding_consistent",
          all(r["ci_low_new_display"] == "%.2f" % float(r["ci_low_new_raw"])
              and r["ci_high_new_display"] == "%.2f" % float(r["ci_high_new_raw"])
              and r["mean_difference_display"] == "%.2f" % means[i]
              for i, r in enumerate(old)),
          "two-decimal display columns match the raw doubles")
    check("no_new_test_or_pvalue",
          "p_value" not in (TABLES / "paired_ci_recomputed.tsv").read_text(encoding="utf-8"),
          "endpoint table carries no P value column")
    check("scoring_unchanged",
          hashlib.sha256(RULES.read_bytes()).hexdigest()
          == env["inputs"]["benchmark_rule_scores.tsv"]["sha256"],
          "frozen rule table hash matches the value recorded at computation time")

    VER.mkdir(parents=True, exist_ok=True)
    out = VER / ("CI_VERIFY_%s.tsv" % tag)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["check", "result", "detail"])
        w.writerows(rows)
    failed = [r for r in rows if r[1] == "FAIL"]
    digest = hashlib.sha256(json.dumps([r[0] for r in rows]).encode()).hexdigest()[:16]
    print(json.dumps({"tag": tag, "pass": len(rows) - len(failed), "total": len(rows),
                      "failed": failed, "check_list_sha": digest}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
