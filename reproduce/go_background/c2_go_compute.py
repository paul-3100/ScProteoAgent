# -*- coding: utf-8 -*-
"""V23 phase C, step 2: run the two frozen ORA branches and the historical reproduction gate.

Branches: HIST_REPLAY (U_H, 19,591) and EXPERIMENTAL_BG (U_E, 4,227).
Contract: GO_CONTRACT.md. Read-only on project inputs; writes only inside the V23 package.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from gb_env import resolve_paths  # noqa: E402

# Release path adaptation: the frozen inputs and outputs come from a data archive instead of
# the manuscript working tree. No scientific setting below this block was changed.
_PATHS = resolve_paths(require_pispa=True)
REPO = _PATHS["data_root"]
TABLES = _PATHS["tables"]
CONFIG = _PATHS["config"]
TMP = _PATHS["tmp"]
RUN = _PATHS["pispa_run"]
GO_DIR = _PATHS["gmt_dir"]

GMT_ORDER = ["GO_BP.symbols.gmt", "GO_CC.symbols.gmt", "GO_MF.symbols.gmt"]
SPLIT_RE = re.compile("[;|,/\\s\\\\]+")
CONTRACTS = ["Cluster_1_vs_Cluster_2", "Cluster_1_vs_Cluster_3", "Cluster_2_vs_Cluster_3"]
LABEL = {
    "Cluster_1_vs_Cluster_2": "C1 - C2",
    "Cluster_1_vs_Cluster_3": "C1 - C3",
    "Cluster_2_vs_Cluster_3": "C2 - C3",
}
DIRECTIONS = ["up", "down"]
MIN_OVERLAP = 1
FDR_CUTOFF = 0.05
FULL_PRECISION = 17
NS_PREFIX = {"GOBP_": "BP", "GOCC_": "CC", "GOMF_": "MF"}
STORED = {
    ("Cluster_1_vs_Cluster_2", "up"): "GO_Cluster_1_vs_Cluster_2_upregulated_proteins.csv",
    ("Cluster_1_vs_Cluster_2", "down"): "GO_Cluster_1_vs_Cluster_2_downregulated_proteins.csv",
    ("Cluster_1_vs_Cluster_3", "up"): "GO_Cluster_1_vs_Cluster_3_upregulated_proteins.csv",
    ("Cluster_1_vs_Cluster_3", "down"): "GO_Cluster_1_vs_Cluster_3_downregulated_proteins.csv",
    ("Cluster_2_vs_Cluster_3", "up"): "GO_Cluster_2_vs_Cluster_3_upregulated_proteins.csv",
    ("Cluster_2_vs_Cluster_3", "down"): "GO_Cluster_2_vs_Cluster_3_downregulated_proteins.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def split_tokens(value) -> list:
    s = str(value)
    if s.lower() == "nan":
        return []
    out = []
    for tok in SPLIT_RE.split(s):
        tok = tok.strip()
        if tok and tok.lower() != "nan":
            out.append(tok.upper())
    return out


def load_gene_sets():
    combined = {}
    for name in GMT_ORDER:
        with (GO_DIR / name).open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                term_id = parts[0].strip()
                desc = parts[1].strip() if parts[1].strip() else term_id
                genes = [g.strip() for g in parts[2:] if g.strip()]
                if not term_id or not genes:
                    continue
                combined[term_id] = {"description": desc, "genes": sorted(set(genes))}
    return combined


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    m = pvals.size
    if m == 0:
        return pvals
    order = np.argsort(pvals, kind="stable")
    ranked = pvals[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = val if val < prev else prev
        q[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


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
    started = time.time()
    gene_sets = load_gene_sets()
    merged_terms = sorted(gene_sets)
    U_H = set()
    for entry in gene_sets.values():
        U_H.update(entry["genes"])
    term_genes = {tid: set(gene_sets[tid]["genes"]) for tid in merged_terms}

    queries = {}
    query_meta = {}
    for c in CONTRACTS:
        for d in DIRECTIONS:
            path = RUN / "processed_proteins" / c / (c + "_" + d + ".csv")
            df = pd.read_csv(path)
            ordered = {}
            for value in df["PG.Genes"].tolist():
                for tok in split_tokens(value):
                    ordered.setdefault(tok, None)
            q_raw = set(ordered)
            queries[(c, d)] = q_raw & U_H
            query_meta[(c, d)] = dict(rows=int(len(df)), unique=int(len(q_raw)), dropped=sorted(q_raw - U_H))

    U_E = set(pd.read_csv(TABLES / "go_backgrounds.tsv", sep="\t")["gene"].astype(str))
    for key, q in queries.items():
        if not q <= U_E:
            raise SystemExit("query not a subset of U_E for " + str(key))

    results = []
    status_rows = []
    summary_rows = []
    for branch in ("HIST_REPLAY", "EXPERIMENTAL_BG"):
        U = U_H if branch == "HIST_REPLAY" else U_E
        N = len(U)
        bg_id = "U_H_union_GO_members" if branch == "HIST_REPLAY" else "U_E_shared"
        for c in CONTRACTS:
            for d in DIRECTIONS:
                q = queries[(c, d)]
                n = len(q)
                fam = []
                no_overlap = 0
                for tid in merged_terms:
                    tg = term_genes[tid]
                    k = len(q & tg)
                    if k < MIN_OVERLAP:
                        no_overlap += 1
                        continue
                    K = len(tg & U)
                    p = float(hypergeom.sf(k - 1, N, K, n))
                    fam.append((p, tid, K, k))
                fam.sort(key=lambda t: (t[0], t[1]))
                m = len(fam)
                pvals = np.array([t[0] for t in fam], dtype=float)
                qvals = bh_adjust(pvals)
                underflow = 0
                for i, (p, tid, K, k) in enumerate(fam):
                    qv = float(qvals[i])
                    if p < 1e-300 or qv < 1e-300:
                        underflow += 1
                    denom = K / N
                    fe = (k / n) / denom if (n > 0 and k > 0 and K > 0 and N > 0 and denom > 0) else ""
                    ns = next((v for pfx, v in NS_PREFIX.items() if tid.startswith(pfx)), "NA")
                    members = sorted(q & term_genes[tid])
                    results.append(dict(
                        branch=branch, contrast=LABEL[c], direction=d, term_id=tid,
                        namespace=ns, description=gene_sets[tid]["description"],
                        N=N, n=n, K=K, k=k, p=p, q=qv, family_size=m,
                        fold_enrichment=fe,
                        member_genes=";".join(members),
                        member_gene_count=len(members),
                        status="SIGNIFICANT" if qv <= FDR_CUTOFF else "NOT_SIGNIFICANT",
                        p_underflow="TRUE" if p < 1e-300 else "FALSE",
                    ))
                status_rows.append(dict(
                    branch=branch, contrast=LABEL[c], direction=d, term_id="", namespace="", status="NO_OVERLAP",
                    no_overlap_terms=no_overlap,
                    note="0-overlap terms are excluded from the BH family (historical policy)",
                ))
                summary_rows.append(dict(
                    branch=branch, background_id=bg_id, contrast=LABEL[c], direction=d,
                    N=N, n=n, candidate_rows=query_meta[(c, d)]["rows"],
                    family_size_m=m,
                    significant_q_le_0_05=int(np.sum(qvals <= FDR_CUTOFF)),
                    no_overlap_terms=no_overlap,
                    dropped_genes=len(query_meta[(c, d)]["dropped"]),
                    underflow_terms=underflow,
                ))

    write_tsv(
        TABLES / "go_results_all.tsv",
        ["branch", "contrast", "direction", "term_id", "namespace", "description",
         "N", "n", "K", "k", "p", "q", "family_size", "fold_enrichment",
         "member_genes", "member_gene_count", "status", "p_underflow"],
        results,
    )
    write_tsv(
        TABLES / "go_term_status.tsv",
        ["branch", "contrast", "direction", "term_id", "namespace", "status",
         "no_overlap_terms", "note"],
        status_rows,
    )
    write_tsv(
        TABLES / "go_query_summary.tsv",
        ["branch", "background_id", "contrast", "direction", "N", "n", "candidate_rows",
         "family_size_m", "significant_q_le_0_05", "no_overlap_terms", "dropped_genes",
         "underflow_terms"],
        summary_rows,
    )

    # ---------------------------------------------------------- reproduction gate
    df = pd.DataFrame(results)
    lookup = {}
    for (branch, contrast, direction, tid), grp in df.groupby(["branch", "contrast", "direction", "term_id"]):
        lookup[(branch, contrast, direction, tid)] = grp.iloc[0]
    gate_rows = []
    for c in CONTRACTS:
        for d in DIRECTIONS:
            stored_path = RUN / "enrichment_results" / STORED[(c, d)]
            sdf = pd.read_csv(stored_path)
            for _, srow in sdf.iterrows():
                tid = str(srow["ID"])
                rec = lookup.get(("HIST_REPLAY", LABEL[c], d, tid))
                stored_genes = set(str(srow["geneID"]).split("/")) if str(srow["geneID"]) != "" else set()
                if rec is None:
                    gate_rows.append(dict(
                        contrast=LABEL[c], direction=d, term_id=tid, stored_present="TRUE",
                        recomputed_status="TERM_NOT_IN_FAMILY",
                        count_match="", geneset_match="", generatio_match="", bgrratio_match="",
                        p_abs_diff="", p_rel_diff="", q_abs_diff="", q_rel_diff="", conclusion="BLOCKED",
                    ))
                    continue
                count_match = int(srow["Count"]) == int(rec["k"])
                geneset_match = stored_genes == set(str(rec["member_genes"]).split(";"))
                generatio_match = str(srow["GeneRatio"]) == str(int(rec["k"])) + "/" + str(int(rec["n"]))
                bgrratio_match = str(srow["BgRatio"]) == str(int(rec["K"])) + "/" + str(int(rec["N"]))
                sp = float(srow["pvalue"])
                sq = float(srow["p.adjust"])
                p_allowed = max(1e-300, 1e-8 * max(abs(sp), abs(rec["p"])))
                q_allowed = max(1e-300, 1e-8 * max(abs(sq), abs(rec["q"])))
                p_ok = abs(sp - rec["p"]) <= p_allowed
                q_ok = abs(sq - rec["q"]) <= q_allowed
                ok = count_match and geneset_match and generatio_match and bgrratio_match and p_ok and q_ok
                gate_rows.append(dict(
                    contrast=LABEL[c], direction=d, term_id=tid, stored_present="TRUE",
                    recomputed_status=str(rec["status"]),
                    count_match=str(count_match), geneset_match=str(geneset_match),
                    generatio_match=str(generatio_match), bgrratio_match=str(bgrratio_match),
                    p_abs_diff=abs(sp - rec["p"]), p_rel_diff=abs(sp - rec["p"]) / max(abs(sp), 1e-300),
                    q_abs_diff=abs(sq - rec["q"]), q_rel_diff=abs(sq - rec["q"]) / max(abs(sq), 1e-300),
                    conclusion="MATCH" if ok else "MISMATCH",
                ))
    gate = pd.DataFrame(gate_rows)
    write_tsv(
        TABLES / "go_historical_reproduction.tsv",
        ["contrast", "direction", "term_id", "stored_present", "recomputed_status",
         "count_match", "geneset_match", "generatio_match", "bgrratio_match",
         "p_abs_diff", "p_rel_diff", "q_abs_diff", "q_rel_diff", "conclusion"],
        gate_rows,
    )
    n_match = int((gate["conclusion"] == "MATCH").sum())
    n_rows = int(len(gate))
    max_p_rel = float(gate["p_rel_diff"].replace("", np.nan).astype(float).max())
    max_q_rel = float(gate["q_rel_diff"].replace("", np.nan).astype(float).max())
    gate_summary = dict(
        stored_rows=n_rows, matched=n_match, mismatched=n_rows - n_match,
        max_p_rel_diff=max_p_rel, max_q_rel_diff=max_q_rel,
        gate="PASS" if n_match == n_rows and n_rows > 0 else "HISTORICAL_REPRODUCTION_BLOCKED",
    )
    (TMP / "c2_gate.json").write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")

    canonical = dict(
        results_sha256=sha256(TABLES / "go_results_all.tsv"),
        summary_sha256=sha256(TABLES / "go_query_summary.tsv"),
        gate_sha256=sha256(TABLES / "go_historical_reproduction.tsv"),
        rows=len(results),
    )
    (TMP / "c2_canonical.json").write_text(json.dumps(canonical, indent=2), encoding="utf-8")

    print("result rows:", len(results))
    for r in summary_rows:
        print(r["branch"], r["contrast"], r["direction"], "N", r["N"], "n", r["n"],
              "m", r["family_size_m"], "sig", r["significant_q_le_0_05"])
    print("gate:", gate_summary)
    print("seconds:", round(time.time() - started, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
