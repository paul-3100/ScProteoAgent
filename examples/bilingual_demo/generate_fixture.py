"""Deterministically generate the synthetic bilingual-demo fixture.

This is a SOFTWARE DEMONSTRATION dataset. It is not study data, it carries no measured
values, and nothing derived from it may be reported as a result. It exists so that a
Chinese run and an English run can be executed on exactly the same matrix, and so that a
follow-up run can reuse a saved matrix.

The design is fixed on purpose: 24 proteins x 12 samples, two groups of 6, positive
values, no missing cells, one donor per sample so the experimental unit is the sample
itself, and three proteins carrying a planted effect (two up, one down). The seed is
20260921, only the standard library is used, and reruns are byte-stable.

Usage: py -3 generate_fixture.py [output_root]
"""
from __future__ import annotations

import csv
import math
import os
import random
import sys

SEED = 20260921
N_PROTEINS = 24
PER_GROUP = 6
GROUPS = ("Control", "Stimulated")

GENE_SYMBOLS = [
    "GAPDH", "ACTB", "HSPA1A", "TUBB", "ENO1", "PKM", "LDHA", "ALDOA",
    "VIM", "TMSB4X", "MYH9", "ACTN1", "FLNA", "TLN1", "VCL", "PFN1",
    "CFL1", "GSN", "CAPZA1", "ARPC2", "RAC1", "RHOA", "CDH1", "ITGB1",
]
PROTEIN_NAMES = [
    "Glyceraldehyde-3-phosphate dehydrogenase", "Actin cytoplasmic 1",
    "Heat shock 70 kDa protein 1A", "Tubulin beta chain",
    "Alpha-enolase", "Pyruvate kinase PKM",
    "L-lactate dehydrogenase A chain", "Fructose-bisphosphate aldolase A",
    "Vimentin", "Thymosin beta-4", "Myosin-9", "Alpha-actinin-1", "Filamin-A",
    "Talin-1", "Vinculin", "Profilin-1", "Cofilin-1", "Gelsolin",
    "F-actin-capping protein subunit alpha-1", "Actin-related protein 2/3 complex subunit 2",
    "Ras-related C3 botulinum toxin substrate 1", "Transforming protein RhoA",
    "Cadherin-1", "Integrin beta-1",
]

# Planted effects: protein index (0-based) -> stimulated over control fold change.
PLANTED_EFFECTS = {6: 2.60, 11: 0.45, 18: 1.90}
FOLLOWUP_PROTEINS = ("P00007", "P00012", "P00019")


def build_rows():
    rng = random.Random(SEED)
    samples = []
    for group in GROUPS:
        for index in range(1, PER_GROUP + 1):
            samples.append(("%s_%d" % (group[:4].upper(), index), group))
    rows = []
    for index in range(N_PROTEINS):
        protein_id = "P%05d" % (index + 1)
        baseline = rng.uniform(200.0, 4000.0)
        values = []
        for _sample_name, group in samples:
            fold = PLANTED_EFFECTS.get(index, 1.0) if group == "Stimulated" else 1.0
            value = baseline * fold * math.exp(rng.gauss(0.0, 0.10))
            values.append(round(max(1.0, value), 1))
        rows.append((protein_id, GENE_SYMBOLS[index], PROTEIN_NAMES[index], values))
    return samples, rows


def write_matrix(path, samples, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator=chr(10))
        writer.writerow(["PG.ProteinGroups", "PG.Genes", "PG.ProteinNames"]
                        + [name for name, _group in samples])
        for protein_id, gene, name, values in rows:
            writer.writerow([protein_id, gene, name] + ["%.1f" % v for v in values])


def write_sampleinfo(path, samples):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator=chr(10))
        writer.writerow(["FileName", "Cluster", "Donor"])
        for position, (name, group) in enumerate(samples, start=1):
            writer.writerow([name, group, "D%02d" % position])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    samples, rows = build_rows()
    for task_dir in ("zh_run", "en_run", "en_followup"):
        target = os.path.join(root, task_dir)
        os.makedirs(target, exist_ok=True)
        write_matrix(os.path.join(target, "ProteinQuant.csv"), samples, rows)
        write_sampleinfo(os.path.join(target, "SampleInfo.csv"), samples)
        print("wrote matrix and metadata to", target)
    print("proteins:", len(rows), "samples:", len(samples))
    print("planted fold changes:",
          {("P%05d" % (k + 1)): v for k, v in PLANTED_EFFECTS.items()})
    print("follow-up proteins:", FOLLOWUP_PROTEINS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
