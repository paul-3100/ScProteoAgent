# -*- coding: utf-8 -*-
"""V23 phase C, step 3: structural gate between the two branches and the fixed 24-position comparison.

Contract: GO_CONTRACT.md sections 4-6. Read-only on project inputs; writes only inside the V23 package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from gb_env import resolve_paths  # noqa: E402

# Release path adaptation: the frozen inputs and outputs come from a data archive instead of
# the manuscript working tree. No scientific setting below this block was changed.
_PATHS = resolve_paths()
REPO = _PATHS["data_root"]
TABLES = _PATHS["tables"]
TMP = _PATHS["tmp"]
DISPLAY = _PATHS["display_table"]
FDR = 0.05


def num(value) -> str:
    f = float(value)
    if not np.isfinite(f):
        return "NA"
    if f == 0.0:
        return "0.0"
    if abs(f) < 1e-4:
        return repr(f)  # shortest round-trip form; avoids decimal-place rounding
    return np.format_float_positional(f, unique=True, precision=17, trim="0")
def write_tsv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            cells = []
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = num(v)
                cells.append(str(v).replace("\t", " ").replace("\n", " "))
            fh.write("\t".join(cells) + "\n")


def main() -> int:
    df = pd.read_csv(TABLES / "go_results_all.tsv", sep="\t")
    hist = df[df["branch"] == "HIST_REPLAY"].copy()
    exp = df[df["branch"] == "EXPERIMENTAL_BG"].copy()
    key = ["contrast", "direction", "term_id"]
    h = hist.set_index(key).sort_index()
    e = exp.set_index(key).sort_index()

    # ---- hard structural gate: same Q, same k, same members, same family; only N/K/P/q change
    gate_fail = []
    if list(h.index) != list(e.index):
        gate_fail.append("family membership differs between branches")
    joined = h.join(e, lsuffix="_h", rsuffix="_e", how="inner")
    bad_k = int((joined["k_h"] != joined["k_e"]).sum())
    bad_n = int((joined["n_h"] != joined["n_e"]).sum())
    bad_m = int((joined["family_size_h"] != joined["family_size_e"]).sum())
    bad_genes = int((joined["member_genes_h"] != joined["member_genes_e"]).sum())
    changed_N = int((joined["N_h"] != joined["N_e"]).sum())
    changed_K = int((joined["K_h"] != joined["K_e"]).sum())
    for name, value in (("k", bad_k), ("n", bad_n), ("family_size", bad_m),
                        ("member_genes", bad_genes)):
        if value:
            gate_fail.append(name + " differs in " + str(value) + " rows")
    structural_gate = "PASS" if not gate_fail else "FAIL"
    gate_info = dict(
        structural_gate=structural_gate,
        failures=gate_fail,
        rows_compared=int(len(joined)),
        rows_with_changed_N=changed_N,
        rows_with_changed_K=changed_K,
        rows_with_identical_members=int(len(joined)) - bad_genes,
        identical_k=int(len(joined)) - bad_k,
    )
    (TMP / "c3_structural_gate.json").write_text(json.dumps(gate_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- fixed 24 display positions
    disp = pd.read_csv(DISPLAY, sep="\t")
    # Release field adaptation: the distributed archive names the stored member-count column
    # differently from the historical table. Only the column name is mapped; the frozen
    # consumer code below still reads its historical name, and no value is altered.
    if "matched_proteins" not in disp.columns and "matched_member_genes" in disp.columns:
        disp = disp.rename(columns={"matched_member_genes": "matched_proteins"})
    hlook = {tuple(k): v for k, v in h.iterrows()}
    elook = {tuple(k): v for k, v in e.iterrows()}
    rows = []
    for order, (_, drow) in enumerate(disp.iterrows(), start=1):
        contrast = str(drow["contrast"])
        direction = str(drow["direction"])
        tid = str(drow["term_id"])
        kk = (contrast, direction, tid)
        hist_row = hlook.get(kk)
        exp_row = elook.get(kk)
        stored_present = str(drow["present"]).strip().lower() == "true"
        hist_status = "SIGNIFICANT" if (hist_row is not None and hist_row["q"] <= FDR) else (
            "NOT_SIGNIFICANT" if hist_row is not None else "NOT_TESTED_OR_NO_OVERLAP")
        exp_status = "SIGNIFICANT" if (exp_row is not None and exp_row["q"] <= FDR) else (
            "NOT_SIGNIFICANT" if exp_row is not None else "NOT_TESTED_OR_NO_OVERLAP")
        if hist_row is None or exp_row is None:
            label = "NOT_TESTED"
        elif hist_status == "SIGNIFICANT" and exp_status == "SIGNIFICANT":
            label = "SUPPORTED_BOTH"
        elif hist_status == "SIGNIFICANT":
            label = "HIST_ONLY"
        elif exp_status == "SIGNIFICANT":
            label = "EXP_ONLY"
        else:
            label = "NEITHER"
        note = ""
        if hist_row is None:
            note = ("term is absent from the frozen merged family for this contrast x direction "
                    "(no saved value, no tested term; not a zero and not a tested non-significant entry)")
        elif not stored_present:
            note = ("term was tested in the frozen family but was not among the stored/displayed rows; "
                    "the reconstruction returns its statistics")
        rows.append(dict(
            display_order=order,
            contrast=contrast,
            direction=direction,
            term_id=tid,
            display_term=str(drow["term_name"]),
            source_term_name=str(drow.get("source_term_name", "")),
            historical_stored="TRUE" if stored_present else "FALSE",
            stored_matched_proteins=("" if pd.isna(drow["matched_proteins"]) else drow["matched_proteins"]),
            stored_p_adjust=("" if pd.isna(drow["p_adjust"]) else num(drow["p_adjust"])),
            historical_recomputed_status=hist_status,
            hist_k=("" if hist_row is None else int(hist_row["k"])),
            hist_K=("" if hist_row is None else int(hist_row["K"])),
            hist_N=("" if hist_row is None else int(hist_row["N"])),
            hist_p=("" if hist_row is None else hist_row["p"]),
            hist_q=("" if hist_row is None else hist_row["q"]),
            experimental_status=exp_status,
            exp_k=("" if exp_row is None else int(exp_row["k"])),
            exp_K=("" if exp_row is None else int(exp_row["K"])),
            exp_N=("" if exp_row is None else int(exp_row["N"])),
            exp_p=("" if exp_row is None else exp_row["p"]),
            exp_q=("" if exp_row is None else exp_row["q"]),
            hist_fold_enrichment=("" if hist_row is None else hist_row["fold_enrichment"]),
            exp_fold_enrichment=("" if exp_row is None else exp_row["fold_enrichment"]),
            support_label=label,
            original_display_order=order,
            note=note,
        ))
    write_tsv(
        TABLES / "go_fixed_display_comparison.tsv",
        ["display_order", "contrast", "direction", "term_id", "display_term", "source_term_name",
         "historical_stored", "stored_matched_proteins", "stored_p_adjust",
         "historical_recomputed_status", "hist_k", "hist_K", "hist_N", "hist_p", "hist_q",
         "experimental_status", "exp_k", "exp_K", "exp_N", "exp_p", "exp_q",
         "hist_fold_enrichment", "exp_fold_enrichment", "support_label",
         "original_display_order", "note"],
        rows,
    )

    # ---- significant-set comparison per query
    set_rows = []
    for (contrast, direction), grp in df.groupby(["contrast", "direction"]):
        hg = grp[grp["branch"] == "HIST_REPLAY"]
        eg = grp[grp["branch"] == "EXPERIMENTAL_BG"]
        hset = set(hg.loc[hg["q"] <= FDR, "term_id"])
        eset = set(eg.loc[eg["q"] <= FDR, "term_id"])
        set_rows.append(dict(
            contrast=contrast, direction=direction,
            hist_family_m=int(hg["family_size"].iloc[0]),
            exp_family_m=int(eg["family_size"].iloc[0]),
            hist_significant=len(hset), exp_significant=len(eset),
            supported_both=len(hset & eset),
            hist_only=len(hset - eset),
            exp_only=len(eset - hset),
            neither=len(set(hg["term_id"]) - (hset | eset)),
            hist_only_examples=";".join(sorted(hset - eset)[:10]),
            exp_only_examples=";".join(sorted(eset - hset)[:10]),
        ))
    write_tsv(
        TABLES / "go_significant_set_comparison.tsv",
        ["contrast", "direction", "hist_family_m", "exp_family_m", "hist_significant",
         "exp_significant", "supported_both", "hist_only", "exp_only", "neither",
         "hist_only_examples", "exp_only_examples"],
        set_rows,
    )

    fixed_counts = {}
    for label in ("SUPPORTED_BOTH", "HIST_ONLY", "EXP_ONLY", "NEITHER", "NOT_TESTED"):
        fixed_counts[label] = sum(1 for r in rows if r["support_label"] == label)
    print("structural gate:", structural_gate, gate_info["failures"])
    print("fixed 24 positions:", fixed_counts)
    for r in set_rows:
        print(r["contrast"], r["direction"], "hist", r["hist_significant"], "exp", r["exp_significant"],
              "both", r["supported_both"], "hist_only", r["hist_only"], "exp_only", r["exp_only"])
    print("display rows written:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
