from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover - fallback is used only on minimal installs
    hypergeom = None

PROJECT_DIR = Path(__file__).resolve().parent
KB_SOURCE_DIR = PROJECT_DIR / "kb_source"
DEFAULT_OFFLINE_DIR = KB_SOURCE_DIR / "offline_enrichment"

SPECIES_ALIASES = {
    "human": "human",
    "hs": "human",
    "hsa": "human",
    "homo_sapiens": "human",
    "homo sapiens": "human",
    "mouse": "mouse",
    "mm": "mouse",
    "mmu": "mouse",
    "mus": "mouse",
    "mus_musculus": "mouse",
    "mus musculus": "mouse",
}

MSIGDB_SOURCE = {
    "human": KB_SOURCE_DIR / "gene_sets" / "msigdb.v2026.1.Hs.symbols.gmt",
    "mouse": KB_SOURCE_DIR / "gene_sets" / "msigdb.v2026.1.Mm.symbols.gmt",
}

NAMESPACE_PREFIXES = {
    "GO": ("GOBP_", "GOCC_", "GOMF_"),
    "GO_BP": ("GOBP_",),
    "GOBP": ("GOBP_",),
    "GO_CC": ("GOCC_",),
    "GOCC": ("GOCC_",),
    "GO_MF": ("GOMF_",),
    "GOMF": ("GOMF_",),
    "KEGG": ("KEGG_",),
    "REACTOME": ("REACTOME_",),
    "Reactome": ("REACTOME_",),
}


def canonical_species(species: str) -> str:
    key = str(species or "human").strip().lower()
    return SPECIES_ALIASES.get(key, key)


def resource_dir(path: str | os.PathLike[str] | None = None) -> Path:
    return Path(path).resolve() if path else DEFAULT_OFFLINE_DIR


def normalize_namespace(namespace: str) -> str:
    ns = str(namespace or "GO").strip()
    upper = ns.upper().replace("-", "_")
    if upper in {"GOBP", "GO_BP"}:
        return "GO_BP"
    if upper in {"GOCC", "GO_CC"}:
        return "GO_CC"
    if upper in {"GOMF", "GO_MF"}:
        return "GO_MF"
    if upper == "REACTOME":
        return "Reactome"
    if upper == "KEGG":
        return "KEGG"
    if upper == "GO":
        return "GO"
    return ns


def namespace_file(namespace: str) -> str:
    return f"{normalize_namespace(namespace)}.symbols.gmt"


def species_dir(species: str, base_dir: str | os.PathLike[str] | None = None) -> Path:
    return resource_dir(base_dir) / canonical_species(species)


def split_gene_tokens(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    tokens: List[str] = []
    for token in re.split(r"[;|,/\s\\]+", str(value)):
        token = token.strip()
        if not token or token.lower() == "nan":
            continue
        tokens.append(token)
    return tokens


def normalize_gene_symbol(gene: str, species: str = "human") -> str:
    gene = str(gene or "").strip()
    if not gene:
        return ""
    # Keep HGNC symbols uppercase; keep mouse-style symbols title-cased.
    if canonical_species(species) == "mouse":
        return gene[:1].upper() + gene[1:].lower() if gene.isupper() else gene
    return gene.upper()


def normalize_gene_list(genes: Iterable[object], species: str = "human") -> List[str]:
    seen = set()
    normalized: List[str] = []
    for value in genes:
        for token in split_gene_tokens(value):
            gene = normalize_gene_symbol(token, species)
            if gene and gene not in seen:
                seen.add(gene)
                normalized.append(gene)
    return normalized


def read_gmt(path: str | os.PathLike[str]) -> Dict[str, Dict[str, object]]:
    gmt_path = Path(path)
    gene_sets: Dict[str, Dict[str, object]] = {}
    if not gmt_path.exists():
        return gene_sets
    with gmt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term_id = parts[0].strip()
            description = parts[1].strip() if parts[1].strip() else term_id
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not term_id or not genes:
                continue
            gene_sets[term_id] = {
                "description": description,
                "genes": sorted(set(genes)),
            }
    return gene_sets


def write_gmt(path: str | os.PathLike[str], gene_sets: Mapping[str, Mapping[str, object]]) -> None:
    gmt_path = Path(path)
    gmt_path.parent.mkdir(parents=True, exist_ok=True)
    with gmt_path.open("w", encoding="utf-8", newline="\n") as handle:
        for term_id in sorted(gene_sets):
            entry = gene_sets[term_id]
            desc = str(entry.get("description") or term_id)
            genes = sorted(set(str(g) for g in entry.get("genes", []) if str(g).strip()))
            if genes:
                handle.write("\t".join([term_id, desc, *genes]) + "\n")


def build_from_msigdb(
    species: str,
    base_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    species_name = canonical_species(species)
    source = MSIGDB_SOURCE.get(species_name)
    if not source or not source.exists():
        return {}

    targets = {
        "GO": NAMESPACE_PREFIXES["GO"],
        "GO_BP": NAMESPACE_PREFIXES["GO_BP"],
        "GO_CC": NAMESPACE_PREFIXES["GO_CC"],
        "GO_MF": NAMESPACE_PREFIXES["GO_MF"],
        "KEGG": NAMESPACE_PREFIXES["KEGG"],
        "Reactome": NAMESPACE_PREFIXES["REACTOME"],
    }
    out_dir = species_dir(species_name, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    buffers: Dict[str, Dict[str, Dict[str, object]]] = {name: {} for name in targets}

    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term_id = parts[0].strip()
            desc = parts[1].strip() if parts[1].strip() else term_id
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not genes:
                continue
            for name, prefixes in targets.items():
                if any(term_id.startswith(prefix) for prefix in prefixes):
                    buffers[name][term_id] = {"description": desc, "genes": genes}

    for name, entries in buffers.items():
        target = out_dir / namespace_file(name)
        if entries and (overwrite or not target.exists()):
            write_gmt(target, entries)
        counts[name] = len(entries)
    return counts


def available_namespaces(species: str, base_dir: str | os.PathLike[str] | None = None) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    folder = species_dir(species, base_dir)
    for name in ["GO", "GO_BP", "GO_CC", "GO_MF", "KEGG", "Reactome"]:
        path = folder / namespace_file(name)
        if path.exists() and path.stat().st_size > 0:
            out[name] = path
    return out


def resolve_namespace_path(
    namespace: str,
    species: str,
    base_dir: str | os.PathLike[str] | None = None,
    auto_build: bool = True,
) -> Path:
    species_name = canonical_species(species)
    ns = normalize_namespace(namespace)
    path = species_dir(species_name, base_dir) / namespace_file(ns)
    if auto_build and not path.exists():
        build_from_msigdb(species_name, base_dir=base_dir, overwrite=False)
    return path


def load_gene_sets(
    namespace: str,
    species: str = "human",
    base_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, Dict[str, object]]:
    ns = normalize_namespace(namespace)
    if ns == "GO":
        combined: Dict[str, Dict[str, object]] = {}
        for child in ("GO_BP", "GO_CC", "GO_MF"):
            combined.update(load_gene_sets(child, species=species, base_dir=base_dir))
        return combined
    path = resolve_namespace_path(ns, species, base_dir=base_dir)
    return read_gmt(path)


def bh_adjust(pvalues: Sequence[float]) -> List[float]:
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (idx, pvalue) in enumerate(reversed(indexed), start=1):
        original_rank = n - rank + 1
        adj = min(prev, float(pvalue) * n / original_rank)
        prev = adj
        adjusted[idx] = min(adj, 1.0)
    return adjusted


def hypergeom_sf(k: int, population: int, term_size: int, query_size: int) -> float:
    if k <= 0:
        return 1.0
    if hypergeom is not None:
        return float(hypergeom.sf(k - 1, population, term_size, query_size))
    denom = math.comb(population, query_size)
    total = 0
    upper = min(term_size, query_size)
    for i in range(k, upper + 1):
        total += math.comb(term_size, i) * math.comb(population - term_size, query_size - i)
    return min(total / denom, 1.0) if denom else 1.0


def run_ora(
    genes: Iterable[object],
    namespace: str,
    species: str = "human",
    base_dir: str | os.PathLike[str] | None = None,
    fdr_cutoff: float | None = None,
    top_n: int | None = None,
    min_overlap: int = 1,
) -> pd.DataFrame:
    species_name = canonical_species(species)
    query_genes = normalize_gene_list(genes, species_name)
    gene_sets = load_gene_sets(namespace, species_name, base_dir=base_dir)
    if not query_genes or not gene_sets:
        return pd.DataFrame(columns=ora_columns())

    universe = sorted({gene for entry in gene_sets.values() for gene in entry["genes"]})
    universe_set = set(universe)
    query = [gene for gene in query_genes if gene in universe_set]
    query_set = set(query)
    if not query_set:
        return pd.DataFrame(columns=ora_columns())

    rows = []
    population = len(universe_set)
    query_size = len(query_set)
    for term_id, entry in gene_sets.items():
        term_genes = set(entry["genes"]) & universe_set
        overlap = sorted(query_set & term_genes)
        if len(overlap) < min_overlap:
            continue
        pvalue = hypergeom_sf(len(overlap), population, len(term_genes), query_size)
        rows.append({
            "ID": term_id,
            "Description": clean_term_description(term_id),
            "GeneRatio": f"{len(overlap)}/{query_size}",
            "BgRatio": f"{len(term_genes)}/{population}",
            "pvalue": pvalue,
            "p.adjust": pvalue,
            "qvalue": pvalue,
            "geneID": "/".join(overlap),
            "Count": len(overlap),
            "source": "offline_gmt",
        })

    if not rows:
        return pd.DataFrame(columns=ora_columns())

    pvalues = [row["pvalue"] for row in rows]
    adjusted = bh_adjust(pvalues)
    for row, adj in zip(rows, adjusted):
        row["p.adjust"] = adj
        row["qvalue"] = adj

    df = pd.DataFrame(rows)
    df = df.sort_values(["p.adjust", "pvalue", "Count"], ascending=[True, True, False])
    if fdr_cutoff is not None:
        df = df[df["p.adjust"] <= float(fdr_cutoff)]
    if top_n is not None and int(top_n) > 0:
        df = df.head(int(top_n))
    return df.reset_index(drop=True)


def clean_term_description(term_id: str) -> str:
    term = str(term_id)
    for prefix in ("GOBP_", "GOCC_", "GOMF_", "KEGG_", "REACTOME_", "WP_", "HALLMARK_"):
        if term.startswith(prefix):
            term = term[len(prefix):]
            break
    desc = term.replace("_", " ").strip().title()
    for species_suffix in (" Homo Sapiens Human", " Mus Musculus House Mouse"):
        if desc.endswith(species_suffix):
            desc = desc[: -len(species_suffix)]
    return desc


def ora_columns() -> List[str]:
    return [
        "ID",
        "Description",
        "GeneRatio",
        "BgRatio",
        "pvalue",
        "p.adjust",
        "qvalue",
        "geneID",
        "Count",
        "source",
    ]


def annotate_genes_with_pathways(
    genes: Iterable[object],
    namespace: str = "KEGG",
    species: str = "human",
    base_dir: str | os.PathLike[str] | None = None,
    max_terms_per_gene: int = 20,
) -> Dict[str, str]:
    species_name = canonical_species(species)
    gene_sets = load_gene_sets(namespace, species=species_name, base_dir=base_dir)
    gene_to_terms: Dict[str, List[str]] = defaultdict(list)
    for term_id, entry in gene_sets.items():
        desc = clean_term_description(term_id)
        for gene in entry["genes"]:
            gene_to_terms[normalize_gene_symbol(gene, species_name)].append(desc)

    annotations: Dict[str, str] = {}
    for raw_gene in genes:
        for token in split_gene_tokens(raw_gene):
            normalized = normalize_gene_symbol(token, species_name)
            terms = sorted(set(gene_to_terms.get(normalized, [])))[:max_terms_per_gene]
            if terms:
                annotations[token] = "; ".join(terms)
    return annotations


def write_manifest(base_dir: str | os.PathLike[str] | None = None) -> Dict[str, object]:
    base = resource_dir(base_dir)
    manifest = {
        "resource_dir": str(base),
        "runtime_policy": "offline_first",
        "species": {},
    }
    for species in ("human", "mouse"):
        build_from_msigdb(species, base_dir=base, overwrite=False)
        namespaces = available_namespaces(species, base_dir=base)
        manifest["species"][species] = {
            name: {
                "path": str(path.relative_to(base)),
                "size_bytes": path.stat().st_size,
                "term_count": sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")),
            }
            for name, path in namespaces.items()
        }
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
