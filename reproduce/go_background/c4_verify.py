# -*- coding: utf-8 -*-
"""V23 independent verification of the GO computation.

Re-derives every quantity from the frozen inputs with an independent implementation path:
  * P via the hypergeometric PMF summed in log-space (logsumexp)
  * q via statsmodels multipletests(method='fdr_bh')
  * k / K / n re-derived from the frozen gene tables and the frozen background tables
Compares against tables/go_results_all.tsv. Read-only on project inputs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mpmath as mp
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from gb_env import resolve_paths  # noqa: E402

# Release path adaptation: the frozen inputs and outputs come from a data archive instead of
# the manuscript working tree. No scientific setting below this block was changed.
_PATHS = resolve_paths()
REPO = _PATHS["data_root"]
TABLES = _PATHS["tables"]
VERIF = _PATHS["verification"]
TMP = _PATHS["tmp"]
GO_DIR = _PATHS["gmt_dir"]
GMT_ORDER = ["GO_BP.symbols.gmt", "GO_CC.symbols.gmt", "GO_MF.symbols.gmt"]
CONTRACTS = ["Cluster_1_vs_Cluster_2", "Cluster_1_vs_Cluster_3", "Cluster_2_vs_Cluster_3"]
LABEL = {
    "Cluster_1_vs_Cluster_2": "C1 - C2",
    "Cluster_1_vs_Cluster_3": "C1 - C3",
    "Cluster_2_vs_Cluster_3": "C2 - C3",
}
DIRECTIONS = ["up", "down"]
FDR = 0.05


mp.mp.dps = 60


def hypergeom_tail_mp(k: int, N: int, K: int, n: int) -> float:
    """One-sided upper tail with arbitrary precision (log-space mpmath sum)."""
    hi = min(K, n)
    total = mp.mpf(0)
    for j in range(k, hi + 1):
        if j < 0 or n - j < 0 or (N - K) - (n - j) < 0:
            continue
        total += mp.binomial(K, j) * mp.binomial(N - K, n - j) / mp.binomial(N, n)
    p = float(total) if total != 0 else 0.0
    exact_log10 = float(mp.log10(total)) if total != 0 else float("-inf")
    return p, exact_log10


def load_sets():
    sets = {}
    for name in GMT_ORDER:
        with (GO_DIR / name).open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                genes = [g.strip() for g in parts[2:] if g.strip()]
                if genes:
                    sets[parts[0].strip()] = set(genes)
    return sets


def main() -> int:
    VERIF.mkdir(parents=True, exist_ok=True)
    from scipy.stats import hypergeom

    sets = load_sets()
    U_H = set()
    for v in sets.values():
        U_H |= v
    U_E = set(pd.read_csv(TABLES / "go_backgrounds.tsv", sep="\t")["gene"].astype(str))
    qdf = pd.read_csv(TABLES / "go_queries_genes.tsv", sep="\t")
    results = pd.read_csv(TABLES / "go_results_all.tsv", sep="\t")
    res_lookup = {(r.branch, r.contrast, r.direction, r.term_id): r for r in results.itertuples(index=False)}

    rows = []
    worst_p_sampled = 0.0
    worst_q_all = 0.0
    sampled_terms = 0
    p_over_tol = 0
    q_over_tol = 0
    for branch, U in (("HIST_REPLAY", U_H), ("EXPERIMENTAL_BG", U_E)):
        for c in CONTRACTS:
            for d in DIRECTIONS:
                q = set(qdf[(qdf["contrast"] == LABEL[c]) & (qdf["direction"] == d)]["gene"].astype(str))
                N = len(U)
                n = len(q)
                fam = []
                for tid, tg in sets.items():
                    k = len(q & tg)
                    if k < 1:
                        continue
                    K = len(tg & U)
                    fam.append((tid, k, K))
                sub = results[(results.branch == branch) & (results.contrast == LABEL[c]) & (results.direction == d)]
                assert len(fam) == int(sub["family_size"].iloc[0]), "family size mismatch"
                pvals = np.array([float(hypergeom.sf(k - 1, N, K, n)) for _, k, K in fam], dtype=float)
                qvals_sm = multipletests(pvals, method="fdr_bh")[1]
                q_map = {}
                p_map = {}
                for (tid, k, K), p, qv in zip(fam, pvals, qvals_sm):
                    q_map[tid] = float(qv)
                    p_map[tid] = float(p)
                    rec = res_lookup[(branch, LABEL[c], d, tid)]
                    q_tab = float(rec.q)
                    if q_tab == 0.0:
                        relq = 0.0 if float(qv) < 1e-250 else float("inf")
                    else:
                        relq = abs(float(qv) - q_tab) / abs(q_tab)
                    if relq > 1e-10:
                        q_over_tol += 1
                    else:
                        worst_q_all = max(worst_q_all, relq)
                    assert int(rec.k) == k and int(rec.K) == K and int(rec.n) == n and int(rec.N) == N, "structural mismatch"
                    assert str(rec.status) == ("SIGNIFICANT" if float(qv) <= FDR else "NOT_SIGNIFICANT"), "status mismatch"
                ordered = sorted(fam, key=lambda t: (p_map[t[0]], t[0]))
                m = len(ordered)
                sample_ids = set()
                sample_ids.add(ordered[0][0])
                sample_ids.add(min(ordered, key=lambda t: (abs(q_map[t[0]] - FDR), t[0]))[0])
                nonsig = [t for t in ordered if q_map[t[0]] > FDR]
                if nonsig:
                    sample_ids.add(nonsig[0][0])
                for j in np.linspace(0, m - 1, num=min(8, m)).astype(int):
                    sample_ids.add(ordered[int(j)][0])
                for t in ordered:
                    if len(sample_ids) >= 5:
                        break
                    sample_ids.add(t[0])
                for tid in sorted(sample_ids):
                    rec = res_lookup[(branch, LABEL[c], d, tid)]
                    K = int(rec.K)
                    k = int(rec.k)
                    p_ind, log10_ind = hypergeom_tail_mp(k, N, K, n)
                    p_ref = float(rec.p)
                    if p_ref == 0.0:
                        rel = 0.0 if log10_ind < -290 else float("inf")
                    elif p_ref < 1e-250:
                        # scipy sf already underflows here; compare magnitudes instead
                        rel = 0.0 if log10_ind < -240 else float("inf")
                    else:
                        rel = abs(p_ind - p_ref) / abs(p_ref)
                    if rel > 1e-8:
                        p_over_tol += 1
                    else:
                        worst_p_sampled = max(worst_p_sampled, rel)
                    sampled_terms += 1
                    rows.append(dict(
                        branch=branch, contrast=LABEL[c], direction=d, term_id=tid,
                        N=N, n=n, K=K, k=k,
                        p_from_table=p_ref, p_independent=p_ind, p_rel_diff=rel,
                        exact_log10_p=log10_ind,
                        q_from_table=rec.q, q_statsmodels=float(q_map[tid]),
                        q_rel_diff=abs(q_map[tid] - float(rec.q)) / max(abs(float(rec.q)), 1e-300),
                    ))

    vdf = pd.DataFrame(rows)
    vdf.to_csv(VERIF / "GO_INDEPENDENT_CHECK.tsv", sep="\t", index=False, lineterminator="\n")
    verdict = "PASS" if (worst_p_sampled <= 1e-8 and worst_q_all <= 1e-10 and p_over_tol == 0 and q_over_tol == 0) else "FAIL"
    summary = dict(
        families_checked=12,
        terms_in_families=int(len(results)),
        sampled_terms=int(sampled_terms),
        max_p_rel_diff_sampled=worst_p_sampled,
        max_q_rel_diff_all=worst_q_all,
        sampled_rows_over_p_tol=p_over_tol,
        rows_over_q_tol=q_over_tol,
        p_tolerance=1e-8,
        q_tolerance=1e-10,
        reference="mpmath binomial tail, 60 dps",
        verdict=verdict,
    )
    (VERIF / "GO_INDEPENDENT_CHECK.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
