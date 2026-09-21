# -*- coding: utf-8 -*-
"""Statistical panel entry point: redraw the quantitative panels of the manuscript from frozen tables.

This is a data-plotting entry, not a figure-assembly tool. It reads the frozen source tables of
the data archive, redraws one statistical panel at a time, and writes PNG + PDF + SVG next to a
copy of the values it used. The assembled manuscript figures (panel placement, lettering,
cropping) were produced by hand in the authors' document/presentation files and are not
reproduced here.

Usage:
  python reproduce/figures/plot_panels.py --list
  python reproduce/figures/plot_panels.py --panel score_distribution --data-dir <archive> --out-dir <dir>
  python reproduce/figures/plot_panels.py --all --data-dir <archive> --out-dir <dir>

Every panel reads one table from <data-dir>/source_tables/ and fails with the path it tried if
the archive is not where it was told to look. Nothing is recomputed: the values in the tables are
the authority, and the script never writes into the archive.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style (release defaults)
FONT = "Arial"
MM = 1 / 25.4
BASE = 7.5          # body text, pt
TITLE = 8.0         # short panel titles, pt
TICK = 7.0          # ticks and legends, pt
LINE = 1.1          # frame line, pt
AXIS = 0.6          # spine line, pt
DATA_LW = 0.9       # data line, pt
POINT_PT = 2.4      # scatter point diameter, pt
ZERO_COLOR = "#8C8C8C"
ZERO_STYLE = (0, (4, 3))
SEED = 20260918
# Agent identity domain (fixed order; scProteoAgent is always the first colour).
AGENT_COLORS = ["#EC4069", "#7BDF71", "#A686D6", "#FF7777",
                "#1976D2", "#72DDE0", "#FFA54B", "#FFE269"]
BRANCH_COLORS = {"full": "#EC4069", "observed": "#1976D2"}
BRANCH_MARKERS = {"full": "o", "observed": "^"}
DIMENSION_COLORS = {"scientific_coverage": "#0072B2", "current_matrix_evidence": "#00A6A6",
                    "boundary_compliance": "#E69F00", "artifact_reproducibility": "#A686D6",
                    "chinese_report_quality": "#7BDF71", "penalty_credit": "#B0B0B0"}
STUDY_ORDER = ["PiSPA", "iPSC", "Brain", "Carr", "pSCoPE", "DVP", "SCPro",
               "Nociceptor", "BloodCell", "Turnover", "Leakage"]


def style_axes(ax, grid=False):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(AXIS)
    ax.tick_params(width=AXIS, length=2.4, labelsize=TICK, colors="black")
    if grid:
        ax.grid(True, axis="y", linewidth=0.4, color="#E6E6E6")
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    return ax


def read_table(data_dir: Path, name: str) -> list[dict]:
    path = data_dir / "source_tables" / name
    if not path.is_file():
        raise SystemExit("FATAL: %s not found in the data archive. Checked %s. Pass --data-dir "
                         "pointing at the archive root (the directory that holds "
                         "source_tables/)." % (name, path))
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit("FATAL: %s is empty." % path)
    return rows


def save(fig, out_dir: Path, panel: str, source_rows: list[dict], note: str,
         margins=(0.14, 0.98, 0.88, 0.30)) -> None:
    fig.subplots_adjust(left=margins[0], right=margins[1], top=margins[2], bottom=margins[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / ("%s.%s" % (panel, ext)), dpi=450, facecolor="white")
    plt.close(fig)
    if source_rows:
        fields = list(source_rows[0].keys())
        with (out_dir / ("%s_source_data.tsv" % panel)).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_rows)
    print("wrote %s (png/pdf/svg) in %s" % (panel, out_dir))
    print("  " + note)


def fnum(value, default=np.nan):
    try:
        text = str(value).strip()
        return float(text) if text not in ("", "NA", "nan", "None") else default
    except ValueError:
        return default


# ---------------------------------------------------------------- panels
def panel_score_distribution(data_dir, out_dir):
    rows = read_table(data_dir, "score_distribution_source.tsv")
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: -np.mean([fnum(r["rule_total"]) for r in rows if r["method"] == m]))
    color = {m: AGENT_COLORS[i % len(AGENT_COLORS)] for i, m in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(92 * MM, 60 * MM))
    style_axes(ax)
    rng = np.random.default_rng(SEED)
    top = 0.0
    for i, method in enumerate(methods):
        values = np.array([fnum(r["rule_total"]) for r in rows if r["method"] == method])
        x = i + rng.uniform(-0.4, 0.4, size=values.size)          # jitter width 0.8
        top = max(top, float(values.max()))
        ax.boxplot([values], positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color[method], alpha=0.40, edgecolor="black",
                                 linewidth=DATA_LW),
                   medianprops=dict(color="black", linewidth=DATA_LW),
                   whiskerprops=dict(color="black", linewidth=DATA_LW),
                   capprops=dict(color="black", linewidth=DATA_LW),
                   manage_ticks=False)
        ax.scatter(x, values, s=POINT_PT ** 2, color=color[method], edgecolor="none", zorder=3)
        mean = float(values.mean())
        ax.plot([i - 0.22, i + 0.22], [mean, mean], color="black", linewidth=LINE, zorder=4)
        ax.scatter([i], [mean], marker="D", s=3.0 ** 2, color="black", zorder=5)
        ax.annotate("%.1f" % mean, (i, top + 1.6), ha="center", va="bottom", fontsize=TICK)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=TICK)
    ax.set_ylabel("rule total (0-100)", fontsize=BASE)
    ax.set_title("Study-level rule scores, 11 studies per system", fontsize=TITLE, loc="left")
    ax.set_ylim(0, 100)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=3.2, color="black",
                              label="one study"),
                       Line2D([], [], marker="D", linestyle="none", markersize=3.2, color="black",
                              label="mean"),
                       Line2D([], [], color="black", linewidth=DATA_LW,
                              label="box: quartiles and median")],
              loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False,
              fontsize=TICK)
    save(fig, out_dir, "score_distribution", rows,
         "88 rule totals (11 studies x 8 systems); mean labelled above each group",
         margins=(0.16, 0.98, 0.87, 0.44))


def panel_dimension_contribution(data_dir, out_dir):
    rows = read_table(data_dir, "dimension_contribution.tsv")
    dims = ["scientific_coverage", "current_matrix_evidence", "boundary_compliance",
            "artifact_reproducibility", "chinese_report_quality", "penalty_credit"]
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: -sum(fnum(r["mean_dimension_score"]) for r in rows
                                        if r["method"] == m))
    fig, ax = plt.subplots(figsize=(92 * MM, 60 * MM))
    style_axes(ax, grid=True)
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for dim in dims:
        vals = np.array([sum(fnum(r["mean_dimension_score"]) for r in rows
                             if r["method"] == m and r["dimension"] == dim) for m in methods])
        ax.barh(y, vals, left=left, height=0.62, color=DIMENSION_COLORS[dim],
                edgecolor="black", linewidth=0.3, label=dim.replace("_", " "))
        left += vals
    for i, m in enumerate(methods):
        ax.annotate("%.1f" % left[i], (left[i] + 0.4, i), va="center", fontsize=TICK)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=TICK)
    ax.invert_yaxis()
    ax.set_xlabel("mean score summed over the six dimensions (max 100)", fontsize=BASE)
    ax.set_title("Mean contribution of each scoring dimension", fontsize=TITLE, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=6.4)
    save(fig, out_dir, "dimension_contribution", rows,
         "segment sums equal the mean composite score of each system",
         margins=(0.22, 0.97, 0.88, 0.44))


def panel_paired_difference(data_dir, out_dir):
    rows = read_table(data_dir, "paired_diff.tsv")
    means = read_table(data_dir, "paired_means.tsv")
    ci = {r["comparator"]: (fnum(r["ci_low"]), fnum(r["ci_high"])) for r in means}
    order = sorted({r["comparator"] for r in rows},
                   key=lambda c: -np.mean([fnum(r["difference"]) for r in rows if r["comparator"] == c]))
    fig, ax = plt.subplots(figsize=(92 * MM, 58 * MM))
    style_axes(ax, grid=True)
    rng = np.random.default_rng(SEED)
    for i, comparator in enumerate(order):
        vals = np.array([fnum(r["difference"]) for r in rows if r["comparator"] == comparator])
        ax.scatter(i + rng.uniform(-0.18, 0.18, size=vals.size), vals, s=POINT_PT ** 2,
                   color="#9A9A9A", edgecolor="none", zorder=2)
        lo, hi = ci[comparator]
        ax.plot([i, i], [lo, hi], color="#00A6A6", linewidth=LINE, zorder=3)
        ax.plot([i - 0.20, i + 0.20], [lo, lo], color="#00A6A6", linewidth=LINE)
        ax.plot([i - 0.20, i + 0.20], [hi, hi], color="#00A6A6", linewidth=LINE)
        ax.scatter([i], [vals.mean()], marker="D", s=3.4 ** 2, color="#00A6A6",
                   edgecolor="black", linewidth=0.3, zorder=4)
    ax.axhline(0, color=ZERO_COLOR, linewidth=LINE, linestyle=ZERO_STYLE, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=TICK)
    ax.set_ylabel("scProteoAgent minus comparator (points)", fontsize=BASE)
    ax.set_title("Paired study-level differences and 95% intervals", fontsize=TITLE, loc="left")
    ax.set_ylim(0, 42)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=3.0, color="#9A9A9A",
                              label="one study (11 per comparator)"),
                       Line2D([], [], marker="D", linestyle="none", markersize=3.2, color="#00A6A6",
                              markeredgecolor="black", markeredgewidth=0.3, label="mean difference"),
                       Line2D([], [], color="#00A6A6", linewidth=LINE,
                              label="frozen 95% percentile interval")],
              loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False, fontsize=6.4)
    save(fig, out_dir, "paired_difference", rows,
         "77 paired differences; intervals are the frozen registry values, not recomputed here",
         margins=(0.14, 0.98, 0.87, 0.44))


def panel_liver_unit_contrasts(data_dir, out_dir):
    rows = read_table(data_dir, "liver_unit_contrasts.tsv")
    contrasts = sorted({r["contrast_label"] for r in rows})
    candidates = sorted({r["protein_display"] for r in rows})
    fig, axes = plt.subplots(1, len(contrasts), figsize=(150 * MM, 55 * MM), sharey=True)
    for ax, contrast in zip(np.atleast_1d(axes), contrasts):
        style_axes(ax, grid=True)
        ax.axvline(0, color=ZERO_COLOR, linewidth=LINE, linestyle=ZERO_STYLE, zorder=1)
        for j, cand in enumerate(candidates):
            for k, branch in enumerate(("full", "observed")):
                sel = [r for r in rows if r["contrast_label"] == contrast
                       and r["protein_display"] == cand and r["branch"] == branch]
                if not sel:
                    continue
                r = sel[0]
                off = -0.16 + 0.32 * k
                ax.errorbar([fnum(r["effect"])], [j + off],
                            xerr=[[fnum(r["effect"]) - fnum(r["ci_low"])],
                                  [fnum(r["ci_high"]) - fnum(r["effect"])]],
                            fmt=BRANCH_MARKERS[branch], color=BRANCH_COLORS[branch],
                            markersize=3.4, markeredgecolor="black", markeredgewidth=0.3,
                            elinewidth=DATA_LW, capsize=1.6, zorder=3)
        ax.set_yticks(range(len(candidates)))
        ax.set_yticklabels(candidates, fontsize=TICK)
        ax.set_xlabel("effect (log2 scale)", fontsize=BASE)
        ax.set_title(contrast, fontsize=TITLE, loc="left")
        ax.set_ylim(-0.6, len(candidates) - 0.4)
    handles = [Line2D([], [], marker=BRANCH_MARKERS[b], linestyle="none", markersize=3.4,
                      color=BRANCH_COLORS[b], markeredgecolor="black", markeredgewidth=0.3,
                      label=b + " matrix") for b in ("full", "observed")]
    handles.append(Line2D([], [], color=ZERO_COLOR, linewidth=LINE, linestyle=ZERO_STYLE,
                          label="no difference"))
    np.atleast_1d(axes)[0].legend(handles=handles, loc="upper center",
                                  bbox_to_anchor=(1.7, -0.24), ncol=3, frameon=False, fontsize=6.6)
    save(fig, out_dir, "liver_unit_contrasts", rows,
         "18 effect estimates (3 candidates x 3 contrasts x 2 branches); unadjusted t intervals",
         margins=(0.10, 0.99, 0.86, 0.36))


def panel_hela_go_bubble(data_dir, out_dir):
    rows = read_table(data_dir, "hela_go_terms.tsv")
    contrasts = sorted({r["contrast"] for r in rows})
    terms = sorted({r["term_name"] for r in rows})
    fig, ax = plt.subplots(figsize=(120 * MM, 62 * MM))
    style_axes(ax, grid=True)
    sizes = [fnum(r["matched_member_genes"]) for r in rows if fnum(r["matched_member_genes"]) > 0]
    smax = max(sizes) if sizes else 1.0
    for row in rows:
        if str(row["present"]).lower() != "true":
            continue
        x = contrasts.index(row["contrast"])
        y = terms.index(row["term_name"])
        n = fnum(row["matched_member_genes"])
        ax.scatter([x], [y], s=22 * n / smax + 6, c=[fnum(row["neg_log10_p_adjust"])],
                   cmap="viridis", vmin=0, vmax=45, edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xticks(range(len(contrasts)))
    ax.set_xticklabels(contrasts, fontsize=TICK)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=6.6)
    ax.set_xlim(-0.6, len(contrasts) - 0.4)
    ax.set_title("Saved GO terms of the migration case (fixed display set)", fontsize=TITLE, loc="left")
    # A swatch column drawn from rectangles, not a colour-bar artist: matplotlib rasterises a
    # continuous colour bar into an embedded image, and every panel here is meant to stay vector.
    cax = fig.add_axes([0.90, 0.28, 0.028, 0.52])
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 45)
    cax.set_xticks([])
    for i in range(45):
        cax.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=plt.cm.viridis(i / 44.0),
                                    edgecolor="none"))
    cax.set_yticks([0, 15, 30, 45])
    cax.tick_params(labelsize=TICK, width=AXIS, length=2.0)
    for side in ("top", "right", "left"):
        cax.spines[side].set_visible(False)
    cax.set_ylabel("-$log10$ adjusted P", fontsize=TICK)
    save(fig, out_dir, "hela_go_bubble", rows,
         "24 frozen display positions; absent rows have no stored value and are not drawn as q=1",
         margins=(0.30, 0.98, 0.90, 0.12))


def panel_tbc1d10b_cell_distribution(data_dir, out_dir):
    rows = read_table(data_dir, "tbc1d10b_cell_values.tsv")
    clusters = sorted({r["cluster"] for r in rows})
    fig, ax = plt.subplots(figsize=(72 * MM, 60 * MM))
    style_axes(ax, grid=True)
    rng = np.random.default_rng(SEED)
    for i, cluster in enumerate(clusters):
        sel = [r for r in rows if r["cluster"] == cluster]
        obs = np.array([fnum(r["abundance_log2_x_plus_1"]) for r in sel
                        if str(r["imputed_pre_transform"]).lower() != "true"])
        imp = np.array([fnum(r["abundance_log2_x_plus_1"]) for r in sel
                        if str(r["imputed_pre_transform"]).lower() == "true"])
        ax.boxplot([[fnum(r["abundance_log2_x_plus_1"]) for r in sel]], positions=[i], widths=0.6,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#B0B0B0", alpha=0.35, edgecolor="black",
                                 linewidth=DATA_LW),
                   medianprops=dict(color="black", linewidth=DATA_LW),
                   whiskerprops=dict(color="black", linewidth=DATA_LW),
                   capprops=dict(color="black", linewidth=DATA_LW), manage_ticks=False)
        ax.scatter(i + rng.uniform(-0.4, 0.4, size=obs.size), obs, s=POINT_PT ** 2,
                   color="#EC4069", edgecolor="none", zorder=3)
        if imp.size:
            ax.scatter(i + rng.uniform(-0.4, 0.4, size=imp.size), imp, s=POINT_PT ** 2,
                       facecolor="white", edgecolor="#EC4069", linewidth=0.45, zorder=3)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, fontsize=TICK)
    ax.set_ylabel("log2(x + 1) abundance", fontsize=BASE)
    ax.set_title("TBC1D10B across proteomic clusters", fontsize=TITLE, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=3.2, color="#EC4069",
                              label="observed"),
                       Line2D([], [], marker="o", linestyle="none", markersize=3.2,
                              markerfacecolor="white", markeredgecolor="#EC4069",
                              label="imputed to the filled minimum")],
              loc="lower right", frameon=False, fontsize=6.4)
    save(fig, out_dir, "tbc1d10b_cell_distribution", rows,
         "89 cells in three clusters; boxes are computed over all values of the cluster",
         margins=(0.17, 0.97, 0.90, 0.16))


PANELS = {
    "score_distribution": panel_score_distribution,
    "dimension_contribution": panel_dimension_contribution,
    "paired_difference": panel_paired_difference,
    "liver_unit_contrasts": panel_liver_unit_contrasts,
    "hela_go_bubble": panel_hela_go_bubble,
    "tbc1d10b_cell_distribution": panel_tbc1d10b_cell_distribution,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Redraw statistical panels from frozen tables.")
    parser.add_argument("--panel", action="append", choices=sorted(PANELS), default=None)
    parser.add_argument("--all", action="store_true", help="draw every panel")
    parser.add_argument("--list", action="store_true", help="list the panels")
    parser.add_argument("--data-dir", type=Path, default=None, help="archive root (holds source_tables/)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.list:
        for name in sorted(PANELS):
            print(name)
        return 0
    selected = sorted(PANELS) if args.all else (args.panel or [])
    if not selected:
        parser.error("choose --panel <id>, --all or --list")
    if args.data_dir is None:
        parser.error("--data-dir is required (the archive root, not this repository)")
    out_dir = args.out_dir or Path.cwd() / "panels_out"
    plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.size": BASE, "axes.linewidth": AXIS})
    for name in selected:
        PANELS[name](args.data_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

