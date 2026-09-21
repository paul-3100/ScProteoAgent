import warnings
warnings.filterwarnings("ignore")
import os
if os.name == 'nt':
    os.environ["RPY2_CFFI_MODE"] = "ABI"
import requests
import networkx as nx
from Bio import Entrez
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import re
import copy
import gseapy as gp
import json
from json import JSONDecodeError
from typing import List, Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from langchain_core.prompts import ChatPromptTemplate
import mygene
import urllib.parse
import logging
import textwrap
from langchain.tools import tool
from sklearn.decomposition import PCA
from itertools import combinations
import hashlib
from scipy.stats import spearmanr
import umap
import time
from scipy.spatial.distance import pdist, squareform
from config import get_config, set_config
from llm import LLM
from adjustText import adjust_text
from scipy.stats import ttest_ind
from matplotlib.patches import Ellipse
from statsmodels.stats.multitest import multipletests
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from pathlib import Path
try:
    from PIL import Image
except Exception:  # pragma: no cover - visual validation is best effort without Pillow.
    Image = None

from prompts import QC_EVAL_PROMPT
import offline_enrichment as offline_enrich
from evidence_utils import append_evidence, append_external_knowledge
from evaluation_evidence import prepare_user_visible_evidence_pack
import figure_recipes
logging.getLogger("biothings.client").setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parent


def _windows_long_path(path: str | os.PathLike) -> str:
    """Return a Windows long-path-safe string when a generated artifact path is long."""
    path_str = os.fspath(path)
    if os.name != "nt":
        return path_str
    abs_path = os.path.abspath(path_str)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    if len(abs_path) >= 240:
        return "\\\\?\\" + abs_path
    return path_str


def _path_exists(path: str | os.PathLike) -> bool:
    return os.path.exists(path) or os.path.exists(_windows_long_path(path))


def _safe_to_csv(df: pd.DataFrame, path: str | os.PathLike, *args, **kwargs) -> None:
    output_dir = os.path.dirname(os.path.abspath(os.fspath(path)))
    if output_dir:
        os.makedirs(_windows_long_path(output_dir), exist_ok=True)
    df.to_csv(_windows_long_path(path), *args, **kwargs)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _validate_binary_image(path: str | os.PathLike) -> None:
    """Fail fast when a generated image was accidentally text-reencoded."""
    suffix = Path(os.fspath(path)).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return
    safe_path = _windows_long_path(path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"Expected image was not written: {path}")
    if suffix == ".png":
        with open(safe_path, "rb") as handle:
            head = handle.read(8)
        if head != PNG_SIGNATURE:
            raise ValueError(
                f"Invalid PNG signature for {path}: {head.hex(' ')}. "
                "The file may have been written through a text path."
            )
    if Image is not None:
        with Image.open(safe_path) as img:
            img.verify()

R_AVAILABLE = False
R_IMPORT_ERROR = None
try:
    import rpy2
    from rpy2.robjects import pandas2ri, conversion, default_converter
    from rpy2 import robjects
    from rpy2.robjects.vectors import StrVector
    rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
    rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None
    robjects.r('suppressMessages(library(clusterProfiler))')
    robjects.r('suppressMessages(library(org.Hs.eg.db))')
    robjects.r('suppressMessages(library(org.Mm.eg.db))')
    robjects.r('suppressMessages(library(ReactomePA))')
except Exception as e:
    rpy2 = None
    pandas2ri = None
    conversion = None
    default_converter = None
    robjects = None
    StrVector = None
    R_IMPORT_ERROR = e

# 关键：创建一个新的 converter，把 pandas converter 拼进去
if conversion is not None:
    converter = conversion.Converter('pandas+default')
    converter += default_converter
    converter += pandas2ri.converter
    R_AVAILABLE = True
else:
    converter = None


def _require_r():
    if not R_AVAILABLE:
        raise RuntimeError(
            "R/rpy2-dependent tool is unavailable. Install R, rpy2, clusterProfiler, "
            "org.Hs.eg.db, org.Mm.eg.db, and ReactomePA, or use local Python enrichment. "
            f"Original import error: {R_IMPORT_ERROR}"
        )

Entrez.email = os.environ.get("ENTREZ_EMAIL", "example@example.com")
matplotlib.use("Agg")



def preprocess_data_and_sampleinfo(
    data_BC,
    sampleinfo,
):

    # 统一处理各种“nan”情况
    data_BC = data_BC.replace(["nan", "NaN", "NAN", ""], np.nan)
    # 转为数值（防止有字符串混入）
    data_BC = data_BC.apply(pd.to_numeric, errors="coerce")
    # 最终全部填充为0
    data_BC = data_BC.fillna(0)
    sampleinfo = sampleinfo.dropna(subset=["Cluster"])

    return data_BC, sampleinfo


def normalize_contrast(key: str):
    parts = key.split("_vs_")
    parts = sorted(parts)  # 排序保证一致
    return "_vs_".join(parts)

def _round_dict(obj, decimals=4):
    if isinstance(obj, dict):
        return {k: _round_dict(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_dict(v, decimals) for v in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    else:
        return obj

def save_json(data, file_path):
    data_copy = copy.deepcopy(data)
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(_windows_long_path(directory), exist_ok=True)
    safe_path = _windows_long_path(file_path)
    if os.path.exists(safe_path):
        pre_info = load_json(file_path)
        data_copy = recursive_merge(pre_info, data_copy)
    try:
        with open(safe_path, 'w', encoding='utf-8') as f:
            json.dump(data_copy, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise OSError(f"[Error] Saving JSON failed: {e}") from e

def load_json(file_path, encode='utf-8'):
    safe_path = _windows_long_path(file_path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"[Error] Loading JSON failed: file not found: {file_path}")
    try:
        with open(safe_path, 'r', encoding=encode) as f:
            data = json.load(f)
        return data
    except JSONDecodeError as e:
        raise ValueError(f"[Error] Loading JSON failed: invalid JSON in {file_path}: {e}") from e
    except OSError as e:
        raise OSError(f"[Error] Loading JSON failed: {e}") from e


def recursive_merge(base, updates):
    """
    递归合并两个字典：
    - 如果同一个 key 的 value 都是 dict，继续递归合并
    - 如果是 list，则 extend
    - 否则直接覆盖
    """
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in merged:
            # 情况 1：两个值都是 dict → 递归合并
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = recursive_merge(merged[key], value)
            # 情况 2：两个值都是 list → extend
            elif isinstance(merged[key], list) and isinstance(value, list):
                for val in value:
                    if val not in merged[key]:
                        merged[key].append(copy.deepcopy(val))
            # 情况 3：类型不匹配 → 覆盖
            else:
                merged[key] = copy.deepcopy(value)
        else:
            # key 不存在，直接赋值
            merged[key] = copy.deepcopy(value)

    return merged

def select_top_genes(
    df,
    logfc_cutoff=1,
    p_cutoff=0.05,
    top_n_stats=100,
    final_n=100,
    n_clusters=10,
):
    """
    选择具有代表性的差异蛋白基因（推荐用于 LLM 下游分析）
    步骤：
    1. 统计稳健筛选
    2. 功能表达模式聚类（代表性）
    3. 每个 cluster 中选最显著基因
    """

    df = df.copy()

    # Step 1 —— 统计学初筛
    df = df[
        (df["adj.P.Val"] < p_cutoff)
        & (df["logFC"].abs() > logfc_cutoff)
    ].sort_values("B", ascending=False)

    # 取前 top_n_stats 个基因用于后续聚类
    pre_candidates = df.head(top_n_stats)
    if pre_candidates.empty:
        raise ValueError("过滤后没有基因，请检查阈值设置。")

    # Step 2 —— 聚类：让结果更具代表性
    # 使用 logFC + t + AveExpr 做 clustering
    features = pre_candidates[["logFC", "t", "AveExpr"]].values
    features_scaled = StandardScaler().fit_transform(features)

    # 设定 cluster 数量
    kmeans = KMeans(n_clusters=min(n_clusters, len(pre_candidates)), random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)

    pre_candidates["cluster"] = cluster_labels

    # Step 3 —— 从每个 cluster 中选出最有统计证据的基因（按 B 值）
    selected = (
        pre_candidates
        .sort_values("B", ascending=False)
        .groupby("cluster")
        .head(2)   # 每类选 top2，保证代表性
        .sort_values("B", ascending=False)
    )

    # 如果超过 final_n，则再按 B 值取前 10
    selected_final = selected.head(final_n)

    return selected_final

def exclude_gene(
    gene_list: List[str],
    exclude_genes: List[str]
) -> List[str]:
    """
    从 gene_list 中去除：
    1) 完全匹配 exclude_genes 的基因
    2) 以 exclude_genes 中任一元素为前缀的一类基因

    参数
    ----
    gene_list : list[str]
        原始基因列表
    exclude_genes : list[str]
        需要排除的基因或基因前缀

    返回
    ----
    list[str]
        过滤后的基因列表
    """

    if not gene_list or not exclude_genes:
        return gene_list

    exclude_set = set(exclude_genes)

    filtered_genes = []
    for g in gene_list:
        if g in exclude_set:
            continue
        if any(g.startswith(ex) for ex in exclude_set):
            continue
        filtered_genes.append(g)

    return filtered_genes


def resolve_path(file_key: str) -> str:
    config = get_config()
    if file_key not in config:
        raise ValueError(f"Unknown file key: {file_key}")
    return config[file_key]


def _ensure_dir(path: str) -> str:
    os.makedirs(_windows_long_path(path), exist_ok=True)
    return path


def _ensure_parent_dir(file_path: str) -> str:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(_windows_long_path(parent), exist_ok=True)
    return file_path


def _build_output_dir(this_run_folder: str, subfolder: str) -> str:
    return _ensure_dir(os.path.join(this_run_folder, subfolder))


def _build_output_path(this_run_folder: str, out_file: str, subfolder: str = "visualize_results") -> str:
    out_dir = _build_output_dir(this_run_folder, subfolder)
    basename = os.path.basename(out_file)
    path = os.path.join(out_dir, basename)
    return path


def _read_csv_checked(file_path: str, *, description: str | None = None, **kwargs) -> pd.DataFrame:
    read_path = _windows_long_path(file_path)
    if not _path_exists(file_path):
        label = description or "file"
        raise FileNotFoundError(f"The {label} at {file_path} is not found.")
    return pd.read_csv(read_path, **kwargs)


def _get_analysis_design() -> Dict[str, Any]:
    try:
        config = get_config()
    except Exception:
        return {}
    design = config.get("analysis_design", {})
    return design if isinstance(design, dict) else {}


def _design_value(key: str, default: Any = None) -> Any:
    design = _get_analysis_design()
    return design.get(key, default)


def _sample_id_col(sampleinfo: pd.DataFrame) -> str:
    design_col = _design_value("sample_id_col")
    if design_col in sampleinfo.columns:
        return design_col
    for col in ["FileName", "R.FileName", "Sample", "SampleID", "sample", "sample_id"]:
        if col in sampleinfo.columns:
            return col
    return sampleinfo.columns[0]


def _protein_gene_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    protein_col = None
    for col in ["PG.ProteinGroups", "Protein.Group", "ProteinGroups", "ProteinID", "protein"]:
        if col in df.columns:
            protein_col = col
            break
    if protein_col is None:
        protein_col = df.columns[0]
    gene_col = None
    for col in ["PG.Genes", "Genes", "Gene", "gene"]:
        if col in df.columns:
            gene_col = col
            break
    return protein_col, gene_col


def _design_contrasts(sampleinfo: pd.DataFrame, group_col: str) -> List[Dict[str, str]]:
    design = _get_analysis_design()
    contrasts = design.get("differential", {}).get("contrasts", []) if isinstance(design.get("differential"), dict) else []
    normalized = []
    available = set(sampleinfo[group_col].dropna().astype(str).tolist()) if group_col in sampleinfo.columns else set()
    for idx, contrast in enumerate(contrasts or []):
        if not isinstance(contrast, dict):
            continue
        group_a = str(contrast.get("group_a", "")).strip()
        group_b = str(contrast.get("group_b", "")).strip()
        if not group_a or not group_b:
            continue
        if available and (group_a not in available or group_b not in available):
            continue
        name = str(contrast.get("name") or f"{group_a}_vs_{group_b}")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or f"contrast_{idx + 1}"
        normalized.append({"name": name, "group_a": group_a, "group_b": group_b})
    return normalized


def _confidence_from_design_and_backend(backend: str) -> str:
    design = _get_analysis_design()
    if design.get("source") == "auto_inferred" or backend == "python_fallback":
        return "moderate"
    return "high"


def _save_figure(file_path: str, *, fig=None, close_obj=None, vector_formats: Any = None, **savefig_kwargs) -> str:
    _ensure_parent_dir(file_path)
    save_target = fig if fig is not None else plt
    safe_file_path = _windows_long_path(file_path)
    save_kwargs = dict(savefig_kwargs)
    save_kwargs.setdefault("dpi", 300)
    save_target.savefig(safe_file_path, **save_kwargs)

    # Optional vector exports (pdf/svg, Step 04): PNG stays the primary output;
    # extra formats are written next to it with the same rendering options.
    extra_formats = list(vector_formats) if vector_formats is not None else list(_ACTIVE_VECTOR_FORMATS)
    stem, ext = os.path.splitext(safe_file_path)
    current_ext = ext.lstrip(".").lower()
    for fmt in extra_formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        if not fmt_clean or fmt_clean == "png" or fmt_clean == current_ext:
            continue
        try:
            save_target.savefig(f"{stem}.{fmt_clean}", **save_kwargs)
        except Exception as exc:
            warnings.warn(f"Vector export to {fmt_clean} failed for {file_path}: {exc}")

    if close_obj is not None:
        plt.close(close_obj)
    elif fig is not None:
        plt.close(fig)
    else:
        plt.close()

    _validate_binary_image(file_path)

    if str(file_path).lower().endswith(".png") and os.path.exists(safe_file_path) and os.path.getsize(safe_file_path) < 1024:
        warnings.warn(f"Saved PNG appears very small: {file_path}")

    return file_path


VISUAL_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#D4A72C",
]


def _set_visual_theme(context: str = "notebook") -> None:
    sns.set_theme(style="white", context=context)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.linewidth": 1.1,
        "axes.grid": False,
        "grid.alpha": 0.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "text.color": "black",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


def _apply_nature_axes(ax, *, full_frame: bool = True) -> None:
    ax.grid(False)
    ax.set_facecolor("white")
    if full_frame:
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", colors="black", width=1.0, length=4, direction="out")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def _apply_nature_heatmap_frame(ax) -> None:
    ax.grid(False)
    ax.set_facecolor("white")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", colors="black", width=1.0, length=3, direction="out")


# ---------------------------------------------------------------------------
# Group-label normalization, unified categorical palette and large-n plotting
# rules (figure_upgrade_20260904 Step 04). Empty/NaN grouping cells must never
# leak into figures as literal "nan" tick labels: they are shown as
# "Unannotated" with a fixed neutral gray so every plot uses one convention.
# ---------------------------------------------------------------------------
UNANNOTATED_GROUP_LABEL = "Unannotated"
_MISSING_GROUP_TOKENS = {"", "nan", "none", "null", "na", "n/a", "<na>", "nat"}
# Above this many samples, per-sample bars and full scatter strips switch to
# compact representations (sorted line + quartile band, sampled points).
LARGE_N_PANEL_THRESHOLD = 600

_ACTIVE_VECTOR_FORMATS: List[str] = []


def _set_active_vector_formats(formats: Any) -> None:
    """Register extra vector formats (pdf/svg/...) for subsequent figure saves.

    PNG remains the primary, always-generated output; vector formats are saved
    in addition. The registry is reset at every ``visualize()`` entry, so a
    stale value cannot leak across runs.
    """
    global _ACTIVE_VECTOR_FORMATS
    cleaned: List[str] = []
    for fmt in formats or []:
        text = str(fmt).lower().lstrip(".")
        if text and text != "png" and text not in cleaned:
            cleaned.append(text)
    _ACTIVE_VECTOR_FORMATS = cleaned


def _normalize_group_labels(values: Any) -> pd.Series:
    """Normalize a grouping column: missing/NaN/blank labels become 'Unannotated'."""
    s = values if isinstance(values, pd.Series) else pd.Series(list(values))
    text = s.astype(object).where(pd.notna(s), "").astype(str).str.strip()
    return text.mask(text.str.lower().isin(_MISSING_GROUP_TOKENS), UNANNOTATED_GROUP_LABEL)


def _count_unannotated_labels(values: Any) -> int:
    return int((_normalize_group_labels(values) == UNANNOTATED_GROUP_LABEL).sum())


def _group_label_order(values: Any) -> List[str]:
    """Unique normalized labels, numeric labels first then lexicographic."""
    uniq = {str(x) for x in pd.unique(_normalize_group_labels(values))}
    return sorted(uniq, key=lambda x: (not x.isdigit(), x))


def _categorical_palette(labels: Any) -> Dict[str, tuple]:
    """Deterministic VISUAL_PALETTE-based palette shared by all sample plots.

    'Unannotated' always gets the same neutral gray; other labels cycle the
    shared palette (extended with husl when there are more than 10 labels).
    """
    if isinstance(labels, (list, tuple, set)):
        order = list(dict.fromkeys(str(x) for x in labels))
    else:
        order = _group_label_order(labels)
    base = list(VISUAL_PALETTE)
    n_plain = sum(1 for label in order if label != UNANNOTATED_GROUP_LABEL)
    if n_plain > len(base):
        base = base + [tuple(c) for c in sns.color_palette("husl", n_colors=n_plain - len(base))]
    palette: Dict[str, tuple] = {}
    k = 0
    for label in order:
        if label == UNANNOTATED_GROUP_LABEL:
            palette[label] = (0.55, 0.55, 0.55)
        else:
            palette[label] = base[k % len(base)]
            k += 1
    return palette


def _safe_filename(value: object, max_len: int = 120) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^\w.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return (text or "unknown")[:max_len]


def _wrap_label(value: object, width: int = 28) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)) or text


def _wrap_axis_ticklabels(ax, axis: str = "x", width: int = 28, rotation: int | None = None) -> None:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        label.set_text(_wrap_label(label.get_text(), width=width))
    if axis == "x":
        ax.set_xticklabels(labels, rotation=rotation if rotation is not None else 30, ha="right")
    else:
        ax.set_yticklabels(labels)


def _legend_outside(ax, title: str | None = None, ncol: int = 1):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        title=title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        ncol=ncol,
    )


# Structured caption metadata promoted from plot-function return dicts into figure
# records (`caption_metadata` inside figure_manifest.json). Threshold, filter and
# up/down count information lives here instead of being drawn inside the figure.
FIGURE_CAPTION_METADATA_KEYS = (
    "filter_note",
    "thresholds",
    "n_sig",
    "n_up",
    "n_down",
    "n_significant",
    "n_proteins",
    "n_contrasts",
    "pval_cutoff",
    "logfc_cutoff",
    "top_n_per_contrast",
    "max_proteins",
    # Step 04 quality-hardening metadata (drives caption wording downstream):
    "n_unannotated",
    "large_n_mode",
    "omitted_panels",
)


def _figure_record(
        file_path: str,
        category: str,
        caption: str,
        inputs: List[str] | None = None,
        params: Dict[str, Any] | None = None,
        plot_type: str | None = None,
        metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
    record = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "category": category,
        "plot_type": plot_type or os.path.splitext(os.path.basename(file_path))[0],
        "caption": caption,
        "inputs": inputs or [],
        "params": params or {},
        "size_bytes": int(os.path.getsize(file_path)) if os.path.exists(file_path) else 0,
    }
    if metadata:
        record["caption_metadata"] = dict(metadata)
    return record


def _add_figure_record(records: List[Dict[str, Any]], info: Any, category: str, caption: str,
                       inputs: List[str] | None = None, params: Dict[str, Any] | None = None,
                       plot_type: str | None = None) -> None:
    if not info:
        return
    file_path = info.get("file_path") if isinstance(info, dict) else str(info)
    if not file_path:
        return
    metadata = None
    if isinstance(info, dict):
        metadata = {key: info[key] for key in FIGURE_CAPTION_METADATA_KEYS if key in info}
        if not metadata:
            metadata = None
    records.append(_figure_record(file_path, category, caption, inputs, params, plot_type, metadata))


def _write_figure_catalog(this_run_folder: str, records: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, str]:
    out_dir = _build_output_dir(this_run_folder, "visualize_results")
    manifest_path = os.path.join(out_dir, "figure_manifest.json")
    index_path = os.path.join(out_dir, "figure_index.md")
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "visualize_params": params,
        "figures": records,
    }

    def _catalog_json_safe(obj):
        if isinstance(obj, dict):
            safe = {}
            for key, value in obj.items():
                safe_key = key if isinstance(key, (str, int, float, bool)) or key is None else str(key)
                safe[safe_key] = _catalog_json_safe(value)
            return safe
        if isinstance(obj, (list, tuple)):
            return [_catalog_json_safe(item) for item in obj]
        return obj

    payload = _catalog_json_safe(payload)
    registered = {os.path.basename(str(record.get("file_path", ""))) for record in payload["figures"]}
    try:
        for png_name in sorted(os.listdir(out_dir)):
            if not png_name.lower().endswith(".png") or png_name in registered:
                continue
            payload["figures"].append(_figure_record(os.path.join(out_dir, png_name), "Sample structure", "辅助图（渲染管线产出，自动编目）：数据文件的补充视图。"))
            registered.add(png_name)
    except OSError:
        pass
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    category_order = [
        ("QC", "QC and data quality"),
        ("Sample structure", "Sample structure"),
        ("Differential evidence", "Differential evidence"),
        ("Mechanism evidence", "Mechanism evidence"),
        ("Enrichment evidence", "Enrichment evidence"),
        ("Dataset recipe", "Dataset recipe figures"),
    ]
    lines = [
        "# Figure Index",
        "",
        "本索引记录 `visualize()` 生成的静态图册、输入来源和可复现参数，供最终报告引用。",
        "",
    ]
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    for category, title in category_order:
        items = by_category.get(category, [])
        if not items:
            continue
        lines.extend([f"## {title}", ""])
        for record in items:
            rel_path = os.path.basename(record["file_path"])
            lines.append(f"- `![]({rel_path})` - {record['caption']}")
            if record.get("inputs"):
                lines.append(f"  - Inputs: {', '.join(os.path.basename(str(x)) for x in record['inputs'])}")
        lines.append("")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return {"figure_manifest": manifest_path, "figure_index": index_path}


def _placeholder_plot(this_run_folder: str, out_file: str, title: str, message: str) -> Dict[str, Any]:
    """Empty-data panels are no longer rendered as gray-text placeholder images.

    Step 04 rule: a figure with no matching data is omitted from the album
    entirely (no file is written). The returned marker carries an empty
    ``file_path`` so the figure catalog, album and captions skip it, plus an
    ``omitted_reason`` for downstream notes.
    """
    return {"file_path": "", "empty": True, "omitted_reason": f"{title}: {message}"}


def _parse_ratio(value: object) -> float:
    text = str(value or "")
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            denom = float(right)
            return float(left) / denom if denom else np.nan
        except Exception:
            return np.nan
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _parse_enrichment_file_name(file_name: str) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    stem = stem.replace("_proteins", "")
    direction = "unknown"
    for suffix in ["_upregulated", "_downregulated"]:
        if stem.endswith(suffix):
            direction = suffix.strip("_")
            stem = stem[:-len(suffix)]
            break
    namespace = "Enrichment"
    for prefix in ["Reactome_", "KEGG_", "GO_BP_", "GO_CC_", "GO_MF_", "GO_"]:
        if stem.startswith(prefix):
            namespace = prefix.strip("_")
            stem = stem[len(prefix):]
            break
    return {"namespace": namespace, "contrast": stem or "ALL", "direction": direction}


def _collect_differential_records(differential_file_paths: List[str], pval_cut: float = 0.05,
                                  logfc_cut: float = 1.0) -> pd.DataFrame:
    frames = []
    for file_path_in in differential_file_paths:
        if not os.path.exists(file_path_in):
            continue
        dep = _read_csv_checked(file_path_in, description="differential protein file")
        if dep.empty or not {"contrast", "logFC", "adj.P.Val"}.issubset(dep.columns):
            continue
        dep = dep.copy()
        dep["adj.P.Val"] = pd.to_numeric(dep["adj.P.Val"], errors="coerce").fillna(1.0).clip(lower=1e-300)
        dep["logFC"] = pd.to_numeric(dep["logFC"], errors="coerce")
        dep = dep.dropna(subset=["logFC"])
        protein_source = "PG.ProteinGroups" if "PG.ProteinGroups" in dep.columns else ("protein" if "protein" in dep.columns else dep.columns[0])
        gene_source = "PG.Genes" if "PG.Genes" in dep.columns else protein_source
        dep["protein_id"] = dep[protein_source].astype(str)
        dep["protein_label"] = dep[gene_source].fillna(dep["protein_id"]).astype(str).str.split(";").str[0]
        dep["neglog10p"] = -np.log10(dep["adj.P.Val"])
        dep["score"] = dep["logFC"].abs() * dep["neglog10p"]
        dep["direction"] = np.where(dep["logFC"] >= 0, "Up", "Down")
        dep["is_significant"] = (dep["adj.P.Val"] < pval_cut) & (dep["logFC"].abs() >= logfc_cut)
        dep["source_file"] = os.path.basename(file_path_in)
        frames.append(dep)
    if not frames:
        return pd.DataFrame(columns=[
            "contrast", "logFC", "adj.P.Val", "protein_id", "protein_label", "neglog10p",
            "score", "direction", "is_significant", "source_file"
        ])
    return pd.concat(frames, ignore_index=True)


def _group_summary_for_proteins(data_BC: pd.DataFrame, sampleinfo: pd.DataFrame,
                                protein_ids: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_anal, sampleinfo_plot, sample_cols, _ = _prepare_sample_plot_data(data_BC, sampleinfo)
    sample_groups = _normalize_group_labels(sampleinfo_plot.set_index("FileName").loc[sample_cols, "Cluster"])
    protein_ids = [p for p in protein_ids if p in data_anal.index]
    if not protein_ids:
        return pd.DataFrame(), pd.DataFrame()
    means = {}
    detects = {}
    for group, samples in sample_groups.groupby(sample_groups).groups.items():
        group_samples = list(samples)
        mat = data_anal.loc[protein_ids, group_samples]
        means[str(group)] = mat.mean(axis=1)
        detects[str(group)] = mat.replace(0, np.nan).notna().mean(axis=1)
    mean_df = pd.DataFrame(means)
    detect_df = pd.DataFrame(detects)
    return mean_df, detect_df


def _request_with_retry(
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: int | float = 20,
    retries: int = 2,
    backoff: float = 1.0,
    raise_for_status: bool = True,
    **kwargs,
):
    requester = session.request if session is not None else requests.request
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requester(method, url, timeout=timeout, **kwargs)
            if raise_for_status:
                response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_error = e
            if attempt >= retries:
                raise
            time.sleep(backoff * (attempt + 1))

    raise last_error


def _normalize_uniprot_category(category: str) -> str | None:
    """Normalize flexible category inputs to function, location, or all."""
    if category is None:
        return "function"

    raw = str(category).strip().lower()
    if not raw:
        return "function"

    normalized = re.sub(r"[\s\\/_-]+", " ", raw)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    def has_any(text: str, terms: List[str]) -> bool:
        return any(term in text for term in terms)

    function_terms = [
        "function",
        "functional",
        "molecular function",
        "functional annotation",
        "protein function",
        "功能",
        "功能注释",
        "蛋白功能",
    ]
    location_terms = [
        "location",
        "localization",
        "localisation",
        "subcellular",
        "organelle",
        "compartment",
        "定位",
        "亚细胞",
        "亚细胞定位",
        "细胞定位",
    ]
    all_terms = [
        "all",
        "both",
        "function and location",
        "location and function",
        "functional and location",
        "功能和定位",
        "定位和功能",
        "功能定位",
    ]

    if has_any(normalized, all_terms):
        return "all"

    has_function = has_any(normalized, function_terms)
    has_location = has_any(normalized, location_terms)

    if has_function and has_location:
        return "all"
    if has_location:
        return "location"
    if has_function:
        return "function"

    return None

@tool
def load_original_differential_proteins(
    contrast_type: str = "ALL",
    top_n: int = 20,
    fdr_cutoff: float = 0.05,
    *args, **kwargs
):
    """
    读取原始的差异蛋白列表。

    参数:
        top_n : 每个 contrast 内标记为 top hit 的蛋白数量
        fdr_cutoff : 显著性阈值（adj.P.Val）
    """
    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")

    differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv") and contrast_type.lower() in fn.lower()]
    if not differential_proteins_path_list:
        differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv")]

    if len(differential_proteins_path_list) == 0:
        print("No differential protein files found in processed_proteins folder. Try to run limma first.")
        _ = run_pairwise_limma.func()
    if len(differential_proteins_path_list) == 0:
        raise FileNotFoundError("No differential protein files found in processed_proteins folder after running limma.")

    llm_summary = {}

    for fn in differential_proteins_path_list:

        contrast = fn.replace("differential_", "").replace(".csv", "")
        safe_contrast = contrast.replace(" ", "")  # ✅ 统一安全命名
        df = pd.read_csv(os.path.join(processed_proteins_folder, fn))

    # ---- 1. 字段规范化  ----

        required_cols = {
            "PG.Genes": "gene",
            "contrast": "contrast",
            "logFC": "logFC",
            "adj.P.Val": "adj_pval",
        }

        available_cols = {k:v for k, v in required_cols.items() if k in df.columns}

        df_llm = df[list(available_cols.keys())].rename(columns=available_cols)

        # ---- 2. 方向信息 ----
        df_llm["direction"] = np.where(df_llm["logFC"] > 0, "up", "down")

        # ---- 3. contrast 内排序 & top hit ----
        df_llm["rank_in_contrast"] = (
            df_llm
            .groupby("contrast")["logFC"]
            .apply(lambda x: x.abs().rank(ascending=False, method="first"))
            .astype(int)
            .values
        )

        df_llm["is_top_hit"] = (
            (df_llm["rank_in_contrast"] <= top_n) &
            (df_llm["adj_pval"] < fdr_cutoff)
        )

        # ---- 4. 排序 ----
        df_llm = df_llm.sort_values(
            by=["contrast", "rank_in_contrast"]
        )

        # ---- 5. LLM摘要信息 ----
        n_hits = df_llm.shape[0]
        n_sig = (df_llm["adj_pval"] < fdr_cutoff).sum()

        df_summary = {
            "n_hits": int(n_hits),
            "n_significant": int(n_sig),
            "significance_saturation": bool(n_hits == n_sig),
        }
        df_temp = df_llm[["gene", "logFC"]].copy()
        df_dict = {gene: round(logfc, 4) for gene, logfc in zip(df_temp["gene"], df_temp["logFC"])}
        df_dict = dict(list(df_dict.items())[:top_n])  # 只保留 top_n 个基因
        llm_summary[safe_contrast] = {
            "n_total_hits": int(df_llm.shape[0]),
            "n_contrasts": int(df_llm["contrast"].nunique()),
            "filtering_rule": {
                "top_n_per_contrast": top_n,
                "fdr_cutoff": fdr_cutoff
            },
            "csv_summary": df_summary,
            "csv_file_top_n": df_dict
        }

    return llm_summary



def build_gene_info(df_part, gene_col, pval_col):

    average_logfc = df_part[pval_col].abs().mean()
    average_pval = df_part[pval_col].mean()
    gene_name_list = df_part[gene_col].astype(str).tolist()
    gene_info = {
        "average_logFC": round(float(average_logfc), 4),
        "average_p_value": round(float(average_pval), 4),
        "gene_names": ';'.join(gene_name_list)
    }

    return gene_info

@tool
def search_pubmed_for_information(
        genes: List[str]=None,   # 改成 genes，非必须
        keywords=None,           # 可以是 str 或 list
        max_hits_per_gene: int=5,     # 参数名也改掉
        max_sentences: int=5,
        *args, **kwargs):
    """
    检索 PubMed 文献并提取与基因或关键词相关的摘要句子。

    参数：
        genes (List[str], optional): 需要检索的信息基因列表，可为空。
        keywords (str | list, optional): PubMed 检索附加关键词，可为字符串或列表。
        max_hits_per_gene (int): 每个基因的最大检索文献数量。
        max_sentences (int): 从摘要中保留的关键句子数量上限。
    """

    this_run_folder = resolve_path('this_run_folder_path')

    summary = {}
    results = {}
    gene_set = set(genes) if genes else set()

    # 构造关键词逻辑
    keyword_query = ""
    if isinstance(keywords, str) and keywords.strip():
        keyword_query = f" AND {keywords.strip()}"
    elif isinstance(keywords, (list, tuple)) and len(keywords) > 0:
        # 自动拼接 OR
        joined = " OR ".join([str(k).strip() for k in keywords if k])
        if joined:
            keyword_query = f" AND ({joined})"

    if not keyword_query:
        keyword_query = " "

    # 如果没有基因，就只跑一次纯关键词检索
    search_targets = set(genes) if genes else [None]

    # 加载现有的搜索结果
    out_folder = os.path.join(this_run_folder, "pubmed_results")
    os.makedirs(out_folder, exist_ok=True)
    pubmed_path = os.path.join(out_folder, "pubmed_search.json")
    pre_info = {}
    if os.path.exists(pubmed_path):
        pre_info = load_json(pubmed_path)

    for gene in search_targets:
        # 构造查询语句
        if gene:
            query = f"{gene} AND human{keyword_query}"
        else:
            query = f"human{keyword_query}"
        if keyword_query in pre_info and gene in pre_info[keyword_query]:
            results[gene] = pre_info[keyword_query][gene]
            continue
        try:
            # 搜索 PubMed
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_hits_per_gene)
            record = Entrez.read(handle)
            ids = record.get("IdList", [])
            if not ids:
                continue

            # 获取文献
            fetch = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
            articles = Entrez.read(fetch).get("PubmedArticle", [])

            for idx, art in enumerate(articles):
                # title = art["MedlineCitation"]["Article"]["ArticleTitle"]
                abs_list = art["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(abs_list) if abs_list else ""

                # 摘要分句
                sentences = re.split(r'(?<=[.!?])\s+', abstract)

                if gene:
                    filtered = [
                        s for s in sentences
                        if any(re.search(rf'\b{g}\b', s, re.IGNORECASE) for g in gene_set)
                    ]
                # 先去除限制，这是因为pubmed搜索的内容太多了
                # else:
                #     # 没有基因时，保留所有句子（按 max_sentences 截断）
                #     filtered = sentences

                filtered = filtered[:max_sentences]

                if filtered:
                    # results[gene] = ' '.join(filtered)
                    # results[query] = {
                    #     # "gene": gene if gene else "N/A",
                    #     # "pubmed_id": ids[idx],
                    #     # "title": title,
                    #     "sentences": filtered
                    # }
                    results[gene] = {
                        "pubmed_id": ids[idx],
                        "sentences": ' '.join(filtered)
                    }

        except Exception as e:
            # print(f"[Warning] PubMed search failed for {gene}: {e}")
            continue
    summary[keyword_query] = results
    summary["genes"] = genes
    summary["keywords"] = keywords

    # 保存结果
    out_folder = os.path.join(this_run_folder, "pubmed_results")
    os.makedirs(out_folder, exist_ok=True)
    pubmed_path = os.path.join(out_folder, "pubmed_search.json")
    save_json(summary, pubmed_path)
    records = []
    for query_key, query_results in results.items():
        if isinstance(query_results, dict):
            records.append({
                "query": query_key,
                "pmid": query_results.get("pubmed_id"),
                "snippet": query_results.get("sentences", ""),
                "source": "PubMed",
            })
    append_external_knowledge(
        this_run_folder,
        tool="search_pubmed_for_information",
        query={"genes": genes, "keywords": keywords, "max_hits_per_gene": max_hits_per_gene},
        evidence_source="external_literature",
        records=records,
    )
    append_evidence(
        this_run_folder,
        source="external_literature",
        tool="search_pubmed_for_information",
        claim="PubMed literature snippets were retrieved as external background annotations, not as current-matrix evidence.",
        files=[pubmed_path],
        metrics={"n_records": len(records)},
        confidence="low",
        limitations=["External literature cannot raise confidence for matrix-derived claims by itself."],
    )

    return summary



def _fetch_local_database(item_list, top_n, features, data_source, *args, **kwargs):

    results = {}
    for idx, item in enumerate(item_list):
        if idx > top_n:
            break
        file_path = os.path.join(data_source, f"{item.upper()}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    item_data = json.load(f)
                    filtered = {prop: item_data[prop] for prop in features if prop in item_data}
                    results[item] = filtered
            except Exception as e:
                results[item] = {"Error": f"Failed to parse JSON: {e}"}
        # else:
        #     results[item] = {"Empty": "Data not found"}

    return results

@tool
def search_local_database(
    contrast_type: str = "ALL",
    exclude_genes: List[str] = [],
    top_n: int=100,
    features: List[str] = ['Protein Function', 'Disorder'],
    *args, **kwargs
    ):
    """
    检索本地基因/蛋白数据库以返回特征。

    参数：
        exclude_genes (List[str], optional): 需要排除的基因或基因前缀，如角蛋白["KRT"]。
        top_n (int): 每个基因返回的最大条目数量，默认为10。
        features (List[str]): 需要提取的数据库字段名称。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")

    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    all_features = ['Protein Function', 'Gene', 'Information', 'Protein Expression And Localization',
                    'Proteins In Blood', 'Cancer & Cell Lines','Disorder','Additional Inferred Compounds',
                    'Publications','GeneCards Summary','Pathways','SuperPathways','Inferred Drugs','Drugs']
    if not set(features) <= set(all_features):
        features = ['Protein Function', 'Disorder', 'Pathways']
        print(f"Features must be subset of {all_features}. For convenient, default is set to {features}")
    top_n = max(1, min(int(top_n), 50))

    # Data source
    data_source = os.getenv("RAG_DATA", './kb_source/gene_protein')
    if not data_source or not os.path.exists(data_source):
        raise FileNotFoundError(f"Local gene/protein database not found: {data_source}")

    results = {}
    contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f)) and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())]
    if not contrast_list:
        contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f))]

    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")  # ✅ 统一安全命名

        def _find_direction_file(direction):
            stems = list(dict.fromkeys([contrast, safe_contrast, normalize_contrast(contrast)]))
            dirs = list(dict.fromkeys([contrast, safe_contrast, normalize_contrast(contrast)]))
            candidates = []
            for dirname in dirs:
                if not dirname:
                    continue
                base = os.path.join(processed_proteins_folder, dirname)
                for stem in stems:
                    if stem:
                        candidates.append(os.path.join(base, f"{stem}_{direction}.csv"))
                if os.path.isdir(base):
                    for filename in os.listdir(base):
                        if filename.lower().endswith(f"_{direction}.csv"):
                            candidates.append(os.path.join(base, filename))
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate
            return None

        up_path = _find_direction_file("up")
        down_path = _find_direction_file("down")
        if not up_path or not down_path:
            missing = []
            if not up_path:
                missing.append("up")
            if not down_path:
                missing.append("down")
            results[safe_contrast] = {
                "status": "skipped_missing_processed_files",
                "missing_files": missing,
                "note": "Processed protein CSV files were incomplete for this contrast; local database search skipped for this contrast.",
            }
            continue
        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        # up_gene_list = up_df.sort_values("logFC").iloc[:, 1].astype(str).tolist()
        # down_gene_list = down_df.sort_values("logFC").iloc[:, 1].astype(str).tolist()


        up_gene_list = up_df.iloc[:, 1].astype(str).tolist()
        down_gene_list = down_df.iloc[:, 1].astype(str).tolist()

        sub_up_gene_list = [gene for item in up_gene_list for gene in item.split(';')]
        sub_down_gene_list = [gene for item in down_gene_list for gene in item.split(';')]

        sub_up_gene_list = exclude_gene(sub_up_gene_list, exclude_genes)
        sub_down_gene_list = exclude_gene(sub_down_gene_list, exclude_genes)

        sub_up_results = _fetch_local_database(sub_up_gene_list, top_n, features, data_source)
        sub_down_results = _fetch_local_database(sub_down_gene_list, top_n, features, data_source)
        results[safe_contrast] = {f"upregulated_proteins":sub_up_results, f"downregulated_proteins":sub_down_results}

    llm_summary = {
        "contrast_type": contrast_type,
        "exclude_genes": exclude_genes,
        "top_n": top_n,
        "features": features,
        "status": "Local database search is completed, but results need to be checked for details.",
        "results": results
    }

    out_folder = os.path.join(this_run_folder, "local_db_results")
    os.makedirs(out_folder, exist_ok=True)
    db_path = os.path.join(out_folder, "local_db_search.json")
    save_json(results, db_path)

    return llm_summary




def _fetch_uniprot_info(
        queries: List[str],
        group_dict=None,   # ⭐ 新增：支持分组统计
        top_n=10,
        base_url="https://rest.uniprot.org/uniprotkb/search",
        max_results_per_query=1,
        all_pre_info={},
        category="function",
        chunk_size=50,
        *args, **kwargs):

    results = {}
    session = requests.Session()
    category = _normalize_uniprot_category(category)
    if category is None:
        raise ValueError(
            "category must resolve to one of: function, location, all "
            "(examples: function, functional, location, subcellular_location, all)"
        )

    queries = queries[:top_n]

    # ===============================
    # 1️⃣ 缓存
    # ===============================
    to_query = []
    for q in queries:
        if q in all_pre_info:
            results[q] = all_pre_info[q]
        else:
            to_query.append(q)

    # ===============================
    # 2️⃣ batch 查询
    # ===============================
    def fetch_batch(gene_list):

        query_str = " OR ".join([f"gene:{g}" for g in gene_list])
        encoded = urllib.parse.quote(query_str, safe='')

        url = (
            f"{base_url}"
            f"?query={encoded}"
            f"&format=json"
            f"&size={len(gene_list) * max_results_per_query}"
        )

        try:
            resp = _request_with_retry("GET", url, session=session, timeout=15)
            data = resp.json()
            mapping = {}

            for entry in data.get("results", []):
                gene_names = []
                for gene in entry.get("genes", []):
                    if "geneName" in gene and "value" in gene["geneName"]:
                        gene_names.append(gene["geneName"]["value"])

                # ===== FUNCTION =====
                function = ""
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        function = " ".join([
                            t["value"].split('.')[0]
                            for t in comment.get("texts", [])
                        ])
                        break

                # ===== SUBCELLULAR LOCATION =====
                sub_locs = []
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        for loc in comment.get("subcellularLocations", []):
                            loc_val = loc.get("location", {}).get("value", "")
                            if loc_val:
                                sub_locs.append(loc_val)
                for g in gene_names:
                    if category == "function":
                        mapping[g] = {"function": function}
                    elif category == "location":
                        mapping[g] = {"location": "; ".join(sub_locs)}
                    else:
                        mapping[g] = {
                            "function": function,
                            "location": "; ".join(sub_locs)
                        }
            return mapping
        except Exception:
            return {}
    # ===============================
    # 3️⃣ 分块执行
    # ===============================
    for i in range(0, len(to_query), chunk_size):
        chunk = to_query[i:i + chunk_size]
        batch_result = fetch_batch(chunk)
        for g in chunk:
            if g in batch_result or g.upper() in batch_result or g.lower() in batch_result:
                key = g if g in batch_result else (g.upper() if g.upper() in batch_result else g.lower())
                info = batch_result[key]
                results[g] = info
                if category == "all" or category in info:
                    all_pre_info[g] = info
                # if info["function"] or info["location"]:
                #     all_pre_info[g] = info
            # else:
            #     info = {"function": "", "location": ""}
            # # info = batch_result.get(g, {"function": "", "location": ""})
            # results[g] = info
            # if info["function"] or info["location"]:
            #     all_pre_info[g] = info

    # ===============================
    # 4️⃣ 分类函数（新增）
    # ===============================
    def classify_location(location_str):
        if not isinstance(location_str, str) or location_str == "":
            return "Unknown"
        loc = location_str.lower()
        if any(k in loc for k in ["cytosol", "cytoplasm"]):
            return "Cytosol"
        if any(k in loc for k in ["membrane", "plasma membrane"]):
            return "Membrane"
        if any(k in loc for k in ["cytoskeleton", "microtubule", "intermediate filament", "keratin"]):
            return "Cytoskeleton"
        if "nucleus" in loc:
            return "Nucleus"
        if "mitochond" in loc:
            return "Mitochondria"
        if "endoplasmic reticulum" in loc:
            return "ER"
        return "Other"

    # ===============================
    # 5️⃣ mapping → DataFrame（新增）
    # ===============================
    def mapping_to_df(mapping, category):
        rows = []

        for gene, info in mapping.items():
            if category == "function":
                rows.append({
                    "gene": gene,
                    "function": info.get("function", ""),
                })
            elif category == "location":
                rows.append({
                    "gene": gene,
                    "location": info.get("location", ""),
                })
            else:
                rows.append({
                    "gene": gene,
                    "function": info.get("function", ""),
                    "location": info.get("location", "")
                })

        df = pd.DataFrame(rows)
        if category in {"location", "all"}:
            df["category"] = df["location"].apply(classify_location)
        return df

    anno_df = mapping_to_df(results, category)

    # ===============================
    # 6️⃣ 分组统计（新增）
    # ===============================

    stats_df = None
    if category in {"location", "all"}:
        if group_dict is not None:

            stats = []
            for group, genes in group_dict.items():
                sub = anno_df[anno_df["gene"].isin(genes)]

                count = sub["category"].value_counts()
                ratio = sub["category"].value_counts(normalize=True)

                tmp = pd.DataFrame({
                    "category": count.index,
                    "count": count.values,
                    "ratio": ratio.values,
                })
                tmp["group"] = group
                stats.append(tmp)

            stats_df = pd.concat(stats, ignore_index=True)

    return results, anno_df, stats_df, all_pre_info

def format_for_llm(anno_df, stats_df):
    """
    输出适合 LLM 分析的结构化结果
    """

    # ===============================
    # 1️⃣ 每组的 category 分布
    # ===============================
    group_summary = {}

    for group in stats_df["group"].unique():
        sub = stats_df[stats_df["group"] == group]

        # 按比例排序
        sub = sub.sort_values("ratio", ascending=False)

        group_summary[group] = {
            row["category"]: round(row["ratio"], 4)
            for _, row in sub.iterrows()
        }

    # ===============================
    # 2️⃣ 每组代表蛋白（增强解释性）
    # ===============================
    group_examples = {}

    for group in stats_df["group"].unique():
        sub_stats = stats_df[stats_df["group"] == group]

        examples = {}

        for _, row in sub_stats.iterrows():
            cat = row["category"]

            genes = anno_df[
                (anno_df["category"] == cat)
            ]["gene"].tolist()

            examples[cat] = genes

        group_examples[group] = examples

    # ===============================
    # 3️⃣ 自动对比（LLM最重要）
    # ===============================
    comparison = {}

    groups = list(group_summary.keys())
    if len(groups) == 2:
        g1, g2 = groups

        comparison = {
            "enriched_in_" + g1: [],
            "enriched_in_" + g2: [],
            "enriched_in_both": []
        }

        all_cats = set(group_summary[g1]) | set(group_summary[g2])

        for cat in all_cats:
            v1 = group_summary[g1].get(cat, 0)
            v2 = group_summary[g2].get(cat, 0)

            if v1 > v2:
                comparison["enriched_in_" + g1].append(cat)
            elif v2 > v1:
                comparison["enriched_in_" + g2].append(cat)
            else:
                comparison["enriched_in_both"].append(cat)

    # ===============================
    # 4️⃣ 最终结构
    # ===============================
    llm_ready = {
        "summary": group_summary,
        # "examples": group_examples,
        "comparison": comparison
    }

    return llm_ready


def compute_category_stats(anno_df, group_dict):
    """
    计算每个category的显著性（Fisher exact test）
    """

    results = []

    groups = list(group_dict.keys())
    if len(groups) != 2:
        raise ValueError("目前只支持两个组比较")

    g1, g2 = groups

    genes_g1 = set(group_dict[g1])
    genes_g2 = set(group_dict[g2])

    all_categories = anno_df["category"].unique()

    results = {}
    for cat in all_categories:

        # 是否属于该category
        anno_df["is_cat"] = (anno_df["category"] == cat)

        # 统计数量
        a = anno_df[(anno_df["gene"].isin(genes_g1)) & (anno_df["is_cat"])].shape[0]
        b = anno_df[(anno_df["gene"].isin(genes_g1)) & (~anno_df["is_cat"])].shape[0]
        c = anno_df[(anno_df["gene"].isin(genes_g2)) & (anno_df["is_cat"])].shape[0]
        d = anno_df[(anno_df["gene"].isin(genes_g2)) & (~anno_df["is_cat"])].shape[0]

        # Fisher检验
        _, p_value = fisher_exact([[a, b], [c, d]])

        results[cat] = {
            "up_ratio": a / (a + b + 1e-9),
            "down_ratio": c / (c + d + 1e-9),
            "p_value": p_value
        }

    return results

def convert_stats_df(stats_df):
    result = defaultdict(dict)

    for item in stats_df.to_dict("records"):
        group = item["group"]
        category = item["category"]
        ratio = item["ratio"]

        result[group][category] = ratio

    return dict(result)

@tool
def search_uniprot_for_protein_knowledge(
        contrast_type: str = "ALL",
        exclude_genes: List[str] = [],
        top_n: int = 500,
        category: str = "function",
        *args, **kwargs):
    """
    查询 UniProt 获取基因/蛋白的功能。

    参数：
        contrast_type (str): 对比类型过滤，如 "ALL"（默认）。
        exclude_genes (List[str], optional): 需要排除的基因或基因前缀，如角蛋白["KRT"]。
        top_n (int): 每组返回的最大基因/蛋白数量。
    """
    # if top_n > 100:
    #     top_n = 100
    this_run_folder = resolve_path('this_run_folder_path')

    base_url = "https://rest.uniprot.org/uniprotkb/search"

    if not this_run_folder or not os.path.exists(this_run_folder):
        raise ValueError("Please provide a valid this_run_folder path")

    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    # 为了方便测试，直接调用查询好的结果
    all_pre_info = {}
    out_folder = os.path.join(this_run_folder, "uniprot_results")
    os.makedirs(out_folder, exist_ok=True)
    category = _normalize_uniprot_category(category)

    if category is None:
        raise ValueError(
            "category must resolve to one of: function, location, all "
            "(examples: function, functional, location, subcellular_location, all)"
        )

    all_uniprot_info_path = os.path.join(out_folder, f"all_uniprot_info_{category}.json")
    if os.path.exists(all_uniprot_info_path):
        all_pre_info = load_json(all_uniprot_info_path)

    results = {}
    contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f)) and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())]
    if not contrast_list:
        contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f))]


    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")  # ✅ 统一安全命名
        up_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_up.csv")
        down_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_down.csv")
        if not os.path.exists(up_path) or not os.path.exists(down_path):
            results[safe_contrast] = {
                "upregulated_proteins": {},
                "downregulated_proteins": {},
                "STRING database confidence threshold": score_threshold,
                "note": f"Skipping PPI network construction because processed up/down protein files were not found for contrast {contrast}.",
                "missing_files": [p for p in [up_path, down_path] if not os.path.exists(p)],
            }
            continue
        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        up_gene_list = up_df.loc[:, "PG.Genes"].astype(str).tolist()
        down_gene_list = down_df.loc[:, "PG.Genes"].astype(str).tolist()

        sub_up_gene_list = exclude_gene(up_gene_list, exclude_genes)
        sub_down_gene_list = exclude_gene(down_gene_list, exclude_genes)

        group_dict = {"upregulated_proteins": sub_up_gene_list, "downregulated_proteins": sub_down_gene_list}

        all_genes = list(set(sub_up_gene_list + sub_down_gene_list))

        uniprot_results, anno_df, stats_df, all_pre_info = _fetch_uniprot_info(all_genes, group_dict=group_dict,
                                                                               top_n=top_n, base_url=base_url,
                                                                               all_pre_info=all_pre_info, category=category)
        stats_test_df = compute_category_stats(anno_df, group_dict) if category in {"location", "all"} else {}

        temp = {}

        # temp["location_category_summary"] = stats_df.to_dict(orient='records') if stats_df is not None else {}
        temp["location_category_summary"] = convert_stats_df(_round_dict(stats_df, 4)) if stats_df is not None else {}
        if category in {"function", "all"}:
            temp["protein_knowledge"] = {"upregulated_proteins":{k: v for k, v in uniprot_results.items() if k in sub_up_gene_list or k.upper() in sub_up_gene_list or k.lower() in sub_up_gene_list},
                                        "downregulated_proteins":{k: v for k, v in uniprot_results.items() if k in sub_down_gene_list or k.upper() in sub_down_gene_list or k.lower() in sub_down_gene_list}}
        temp["llm_data"] = format_for_llm(anno_df, stats_df) if stats_df is not None else {}
        temp["significance_evaluation"] = stats_test_df

        results[safe_contrast] = temp


    llm_summary = {
        "contrast_type": contrast_type,
        "exclude_genes": exclude_genes,
        "top_n": top_n,
        "category": category,
        "status": "UniProt search is correctly completed",
        "results": results
    }

    llm_summary = _round_dict(llm_summary, decimals=4)
    results = _round_dict(results, decimals=4)

    out_folder = os.path.join(this_run_folder, "uniprot_results")
    os.makedirs(out_folder, exist_ok=True)
    uniprot_path = os.path.join(out_folder, f"uniprot_info_{category}.json")
    all_uniprot_path = os.path.join(out_folder, f"all_uniprot_info_{category}.json")

    save_json(results, uniprot_path)
    save_json(all_pre_info, all_uniprot_path)

    return llm_summary









def clean_gene_list(gene_list, exclude_genes):
    return list(set(
        gene
        for item in gene_list
        for gene in str(item).split(';')
        if gene and gene not in (exclude_genes or [])
    ))


def _fetch_single_gene(session, gene, base_url, species, timeout, all_pre_info):
    if gene in all_pre_info:
        return gene, all_pre_info[gene]

    try:
        # Step 1: find
        find_url = f"{base_url}/find/{species}/{gene}"
        res = _request_with_retry("GET", find_url, session=session, timeout=timeout)

        if not res.text.strip():
            return gene, None

        entry_id = res.text.split("\n")[0].split("\t")[0]

        # Step 2: get
        get_url = f"{base_url}/get/{entry_id}"
        res = _request_with_retry("GET", get_url, session=session, timeout=timeout)

        pathways = []
        lines = res.text.splitlines()
        idx = 0

        while idx < len(lines):
            line = lines[idx]
            if line.startswith("PATHWAY"):
                while idx < len(lines) and (
                        lines[idx].startswith("PATHWAY") or lines[idx].startswith(" ")
                ):
                    pathway_ori = lines[idx].replace("PATHWAY", "").strip()
                    if "  " in pathway_ori:
                        pathways.append(pathway_ori.split("  ")[1])
                    idx += 1
                continue
            idx += 1

        result = '; '.join(pathways) if pathways else None
        return gene, result

    except Exception:
        return gene, None


def _fetch_kegg_info_parallel(
        gene_list: List[str],
        all_pre_info: Dict,
        species="hsa",
        max_workers=10,
        base_url="https://rest.kegg.jp",
        timeout=10,
):
    results = {}

    gene_list = list(set(gene_list))
    genes_to_query = [g for g in gene_list if g not in all_pre_info]

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_single_gene,
                session,
                gene,
                base_url,
                species,
                timeout,
                all_pre_info
            ): gene for gene in genes_to_query
        }

        for future in as_completed(futures):
            gene, result = future.result()
            if result:
                all_pre_info[gene] = result

    for gene in gene_list:
        if gene in all_pre_info:
            results[gene] = all_pre_info[gene]

    return results, all_pre_info


@tool
def search_kegg_for_information(
        contrast_type: str = "ALL",
        exclude_genes: List[str] = [],
        species: str = "hsa",
        top_n: int = 100,
):
    """
    高性能 KEGG 查询（并发 + 缓存 + 去重）
    species: 物种KEGG代码，如人类为"hsa"，小鼠为"mus"等，默认为"hsa"
    """

    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")

    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    # ===============================
    # 加载缓存
    # ===============================
    out_folder = os.path.join(this_run_folder, "kegg_results")
    os.makedirs(out_folder, exist_ok=True)

    all_kegg_path = os.path.join(out_folder, "all_kegg_info.json")
    kegg_path = os.path.join(out_folder, "kegg_info.json")

    if os.path.exists(kegg_path):
        pre_info = load_json(kegg_path)
        # if contrast_type.lower() == "all":
        #     return pre_info
        # else:
        if contrast_type in pre_info:
            return pre_info[contrast_type]
        if normalize_contrast(contrast_type) in pre_info:
            return pre_info[normalize_contrast(contrast_type)]

    all_pre_info = load_json(all_kegg_path) if os.path.exists(all_kegg_path) else {}
    pre_info = load_json(kegg_path) if os.path.exists(kegg_path) else {}

    results = {}

    contrast_list = [
        f for f in os.listdir(processed_proteins_folder)
        if os.path.isdir(os.path.join(processed_proteins_folder, f))
        and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())
    ]

    if not contrast_list:
        contrast_list = [
            f for f in os.listdir(processed_proteins_folder)
            if os.path.isdir(os.path.join(processed_proteins_folder, f))
        ]

    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")

        if safe_contrast in pre_info:
            results[safe_contrast] = pre_info[safe_contrast]
            continue

        up_path = os.path.join(processed_proteins_folder, safe_contrast,
                               f"{contrast}_up.csv")
        down_path = os.path.join(processed_proteins_folder, safe_contrast,
                                 f"{contrast}_down.csv")

        if not os.path.exists(up_path) or not os.path.exists(down_path):
            raise FileNotFoundError(f"Processed protein files not found for {contrast}")

        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        gene_col = "PG.Genes" if "PG.Genes" in up_df.columns and "PG.Genes" in down_df.columns else up_df.columns[1]
        up_genes = clean_gene_list(up_df.loc[:, gene_col].tolist(), exclude_genes)
        down_genes = clean_gene_list(down_df.loc[:, gene_col].tolist(), exclude_genes)

        up_genes = up_genes[:top_n]
        down_genes = down_genes[:top_n]

        combined_genes = list(set(up_genes + down_genes))

        if os.getenv("SC_ENRICH_ALLOW_ONLINE", "0") == "1":
            combined_results, all_pre_info = _fetch_kegg_info_parallel(
                combined_genes,
                all_pre_info,
                species=species,
                max_workers=10
            )
            source = "KEGG_REST_online"
        else:
            combined_results = offline_enrich.annotate_genes_with_pathways(
                combined_genes,
                namespace="KEGG",
                species=species,
            )
            all_pre_info.update(combined_results)
            source = "offline_gmt"

        up_results = {g: combined_results[g] for g in up_genes if g in combined_results}
        down_results = {g: combined_results[g] for g in down_genes if g in combined_results}

        results[safe_contrast] = {
            "upregulated_proteins": up_results,
            "downregulated_proteins": down_results,
            "source": source,
        }


    final_results = {contrast_type: results,}
    save_json(final_results, kegg_path)
    save_json(all_pre_info, all_kegg_path)

    llm_summary = {
        "contrast_type": contrast_type,
        "exclude_genes": exclude_genes,
        "top_n": top_n,
        "status": "KEGG search completed (offline-first mode)",
        "results": results
    }

    return llm_summary



def _fetch_drug_info_batch(
        gene_list,
        all_pre_info,
        base_url="https://dgidb.org/api/graphql",
        timeout=20,
        session=None
):
    if not gene_list:
        return {}, all_pre_info

    gene_list = list(set(gene_list))
    genes_to_query = [g for g in gene_list if g not in all_pre_info]

    if not genes_to_query:
        return {g: all_pre_info[g] for g in gene_list if g in all_pre_info}, all_pre_info

    if session is None:
        session = requests.Session()

    headers = {"Content-Type": "application/json"}

    query = """
    query ($genes: [String!]) {
      genes(names: $genes) {
        nodes {
          name
          interactions {
            drug { name }
            interactionScore
            interactionTypes {
              type
              directionality
            }
          }
        }
      }
    }
    """

    payload = {"query": query, "variables": {"genes": genes_to_query}}

    resp = _request_with_retry(
        "POST",
        base_url,
        session=session,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    data = resp.json()

    inhibitory_types = {
        "inhibitor", "antagonist", "blocker",
        "suppressor", "negative modulator"
    }
    nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
    for node in nodes:
        gene_name = node.get("name")
        interactions = node.get("interactions", [])
        drug_list = {}
        for it in interactions:
            drug_name = it.get("drug", {}).get("name")
            itypes = it.get("interactionTypes", [])
            for t in itypes:
                t_type = (t.get("type") or "").lower()
                t_dir = (t.get("directionality") or "").lower()

                if t_dir == "inhibitory" or t_type in inhibitory_types:
                    if t_dir not in drug_list:
                        drug_list[t_dir] = drug_name
                    else:
                        drug_list[t_dir] += " ; " + drug_name
        if drug_list:
            all_pre_info[gene_name] = drug_list
    results = {
        g: all_pre_info[g]
        for g in gene_list
        if g in all_pre_info
    }

    return results, all_pre_info

@tool
def search_DGIdb_for_drug(
    contrast_type: str = "ALL",
    exclude_genes: List[str] = [],
    top_n: int = 100,
):
    """
    search drug info from DGIdb (high-performance version)
    """
    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")

    if not os.path.exists(processed_proteins_folder):
        _ = extract_differential_proteins.func()

    out_folder = os.path.join(this_run_folder, "drug_results")
    os.makedirs(out_folder, exist_ok=True)

    all_drug_path = os.path.join(out_folder, "all_drug_info.json")
    drug_path = os.path.join(out_folder, "drug_info.json")

    all_pre_info = load_json(all_drug_path) if os.path.exists(all_drug_path) else {}
    pre_info = load_json(drug_path) if os.path.exists(drug_path) else {}

    results = {}

    contrast_list = [
        f for f in os.listdir(processed_proteins_folder)
        if os.path.isdir(os.path.join(processed_proteins_folder, f))
        and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())
    ]

    if not contrast_list:
        contrast_list = [
            f for f in os.listdir(processed_proteins_folder)
            if os.path.isdir(os.path.join(processed_proteins_folder, f))
        ]

    session = requests.Session()

    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")

        if safe_contrast in pre_info:
            results[safe_contrast] = pre_info[safe_contrast]
            continue

        up_path = os.path.join(processed_proteins_folder, safe_contrast,
                               f"{contrast}_up.csv")
        down_path = os.path.join(processed_proteins_folder, safe_contrast,
                                 f"{contrast}_down.csv")

        if not os.path.exists(up_path) or not os.path.exists(down_path):
            raise FileNotFoundError(f"{contrast} files not found")

        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        def clean(lst):
            return list(set(
                gene
                for item in lst
                for gene in str(item).split(';')
                if gene and gene not in (exclude_genes or [])
            ))

        up_genes = clean(up_df["PG.Genes"].tolist())[:top_n]
        down_genes = clean(down_df["PG.Genes"].tolist())[:top_n]

        combined_genes = list(set(up_genes + down_genes))
        combined_results, all_pre_info = _fetch_drug_info_batch(
            combined_genes,
            all_pre_info,
            session=session
        )

        up_results = {g: combined_results[g] for g in up_genes if g in combined_results}
        down_results = {g: combined_results[g] for g in down_genes if g in combined_results}

        results[safe_contrast] = {
            "upregulated_proteins": up_results,
            "downregulated_proteins": down_results
        }


    save_json(results, drug_path)
    save_json(all_pre_info, all_drug_path)
    drug_records = []
    for contrast_name, payload in results.items():
        if not isinstance(payload, dict):
            continue
        for direction, genes_payload in payload.items():
            if not isinstance(genes_payload, dict):
                continue
            for gene, drugs in genes_payload.items():
                drug_records.append({
                    "contrast": contrast_name,
                    "direction": direction,
                    "gene": gene,
                    "database": "DGIdb",
                    "interactions": drugs,
                })
    append_external_knowledge(
        this_run_folder,
        tool="search_DGIdb_for_drug",
        query={"contrast_type": contrast_type, "top_n": top_n},
        evidence_source="drug_database",
        records=drug_records,
    )
    append_evidence(
        this_run_folder,
        source="drug_database",
        tool="search_DGIdb_for_drug",
        claim="DGIdb drug-gene interactions were retrieved as database annotations and candidate intervention clues only.",
        files=[drug_path],
        metrics={"n_records": len(drug_records)},
        confidence="low",
        limitations=["DGIdb records are not current-matrix evidence and do not validate drug response in this dataset."],
    )

    llm_summary = {
        "contrast_type": contrast_type,
        "exclude_genes": exclude_genes,
        "top_n": top_n,
        "status": "DGIdb drug search completed (high-performance mode)",
        "results": results
    }

    return llm_summary



def _fetch_ppi_netwrok(
    gene_list,
    species=9606,
    score_threshold=400,
    batch_size=200,
    max_workers=5,
    timeout=30
):
    """
    优化版 STRING PPI 网络获取函数

    参数：
    - gene_list: 输入基因列表
    - species: 物种ID（默认9606人类）
    - score_threshold: 置信度阈值（0-1000）
    - batch_size: 每批处理的基因数（防止URL过长）
    - max_workers: 并行线程数
    """

    G = nx.Graph()

    # ---------------- Step 1: mapping ----------------
    mapping_url = "https://string-db.org/api/json/get_string_ids"

    mapping_res = _request_with_retry(
        "POST",
        mapping_url,
        data={
            "identifiers": "\n".join(gene_list),
            "species": species
        },
        timeout=timeout,
    )
    mapping_data = mapping_res.json()

    id_map = {item['queryItem']: item['stringId'] for item in mapping_data}
    string_ids = list(set(id_map.values()))

    if not string_ids:
        return {'nodes': '', 'edges': [], 'hub_proteins': []}

    # ---------------- Step 2: 分批 ----------------
    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    batches = list(chunk_list(string_ids, batch_size))

    network_url = "https://string-db.org/api/json/network"

    # ---------------- Step 3: 单批请求函数 ----------------
    def fetch_batch(batch_ids):
        try:
            res = _request_with_retry(
                "POST",
                network_url,
                data={
                    "identifiers": "\n".join(batch_ids),
                    "species": species
                },
                timeout=timeout,
            )
            return res.json()
        except Exception as e:
            print(f"[WARNING] batch failed: {e}")
            return []

    # ---------------- Step 4: 并行请求 ----------------
    all_interactions = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_batch, b) for b in batches]

        for future in as_completed(futures):
            result = future.result()
            if result:
                all_interactions.extend(result)

    # ---------------- Step 5: 构建网络 ----------------
    gene_set = set(gene_list)

    for inter in all_interactions:
        try:
            score = inter.get('score', 0) * 1000
            g1 = inter.get('preferredName_A')
            g2 = inter.get('preferredName_B')

            if g1 in gene_set and g2 in gene_set and score >= score_threshold:
                G.add_edge(g1, g2, weight=score)
        except Exception:
            continue

    # ---------------- Step 6: hub proteins ----------------
    degrees = dict(G.degree())
    hub = sorted(degrees, key=degrees.get, reverse=True)[:5] if degrees else []

    nodes = list(G.nodes())
    edges = [f"{u}-{v}" for u, v in G.edges()]

    return {
        'nodes': ';'.join(nodes),
        'edges': edges,
        'hub_proteins': hub
    }

@tool
def build_ppi_network(
        contrast_type: str = "ALL",
        exclude_genes: List[str] = [],
        species="human",
        score_threshold=700, *args, **kwargs
        ):
    """
    构建蛋白质–蛋白质互作(PPI)网络。

    参数:
        contrast_type (str, optional): 对比类型，默认为 "ALL"，表示处理所有对比。
        exclude_genes (List[str], optional): 需要排除的基因或基因前缀，如角蛋白["KRT"]。
        species (int, optional): NCBI Taxonomy ID, 默认为人类9606, 小鼠为10090。
        score_threshold (int, optional): STRING数据库置信度阈值(0-1000)，默认700。
    """
    # Step0: 输入检查
    this_run_folder = resolve_path('this_run_folder_path')

    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    # 为了方便测试，直接调用查询好的结果
    out_folder = os.path.join(this_run_folder, "ppi_results")
    os.makedirs(out_folder, exist_ok=True)
    ppi_path = os.path.join(out_folder, "ppi_network.json")
    if os.getenv("SC_PPI_ALLOW_ONLINE", "0") != "1":
        skipped_result = {
            "status": "skipped_online_disabled",
            "evidence_source": "not_used",
            "note": "STRING PPI construction requires online access and is skipped by default. Set SC_PPI_ALLOW_ONLINE=1 to enable it explicitly.",
            "contrast_type": contrast_type,
            "STRING database confidence threshold": score_threshold,
        }
        save_json({contrast_type: {}, "ppi_status": skipped_result, "skipped_contrasts": []}, ppi_path)
        return skipped_result
    pre_info = {}
    if os.path.exists(ppi_path):
        pre_info = load_json(ppi_path)
        # if contrast_type.lower() == "all":
        #     return pre_info
        if contrast_type in pre_info:
            return pre_info[contrast_type]
        elif normalize_contrast(contrast_type) in pre_info:
            return pre_info[normalize_contrast(contrast_type)]


        # if pre_info:
        #     return pre_info

    results = {}
    skipped_contrasts = []
    contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f)) and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())]
    if not contrast_list:
        contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f))]

    for contrast in contrast_list:

        safe_contrast = contrast.replace(" ", "")  # 统一安全命名
        if safe_contrast in pre_info:
            results[safe_contrast] = pre_info[safe_contrast]
            continue

        def _find_direction_file(direction):
            stems = list(dict.fromkeys([contrast, safe_contrast, normalize_contrast(contrast)]))
            dirs = list(dict.fromkeys([contrast, safe_contrast, normalize_contrast(contrast)]))
            candidates = []
            for dirname in dirs:
                if not dirname:
                    continue
                base = os.path.join(processed_proteins_folder, dirname)
                for stem in stems:
                    if stem:
                        candidates.append(os.path.join(base, f"{stem}_{direction}.csv"))
                if os.path.isdir(base):
                    for filename in os.listdir(base):
                        if filename.lower().endswith(f"_{direction}.csv"):
                            candidates.append(os.path.join(base, filename))
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate
            return None

        up_path = _find_direction_file("up")
        down_path = _find_direction_file("down")
        if not up_path or not down_path:
            missing = []
            if not up_path:
                missing.append("up")
            if not down_path:
                missing.append("down")
            skipped_contrasts.append({"contrast": contrast, "missing": missing})
            results[safe_contrast] = {"upregulated_proteins": {},
                                      "downregulated_proteins": {},
                                      "STRING database confidence threshold": score_threshold,
                                      "status": "skipped_missing_processed_files",
                                      "missing_files": missing,
                                      "note": "Processed protein CSV files were incomplete for this contrast; PPI construction skipped for this contrast."}
            continue
        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)


        if species == "mouse":
            species_id = 10090
            up_df["PG.Genes"] = up_df["PG.Genes"].str.capitalize()
            down_df["PG.Genes"] = down_df["PG.Genes"].str.capitalize()
        else:
            species_id = 9606


        up_gene_list = up_df.iloc[:, 1].astype(str).tolist()
        down_gene_list = down_df.iloc[:, 1].astype(str).tolist()

        sub_up_gene_list = exclude_gene(up_gene_list, exclude_genes)
        sub_down_gene_list = exclude_gene(down_gene_list, exclude_genes)

        if len(sub_up_gene_list) == 0 or len(sub_down_gene_list) == 0:
            results[safe_contrast] = {"upregulated_proteins": {},
                                      "downregulated_proteins": {},
                                      "STRING database confidence threshold": score_threshold,
                                      "note": "No valid genes, skipping PPI network construction."}
            continue
        sub_up_results = _fetch_ppi_netwrok(sub_up_gene_list, species_id, score_threshold)
        time.sleep(1)
        sub_down_results = _fetch_ppi_netwrok(sub_down_gene_list, species_id, score_threshold)
        time.sleep(1)

        results[safe_contrast] = {"upregulated_proteins": sub_up_results,
                                  "downregulated_proteins": sub_down_results,
                                  "STRING database confidence threshold": score_threshold}

    out_folder = os.path.join(this_run_folder, "ppi_results")
    os.makedirs(out_folder, exist_ok=True)
    ppi_path = os.path.join(out_folder, "ppi_network.json")

    final_results = {contrast_type: results,
                     "skipped_contrasts": skipped_contrasts}
    save_json(final_results, ppi_path)

    return results

def clean_and_map_genes(df, gene_col="PG.Genes", id_type="symbol"):
    """
    清洗差异蛋白/基因列表并映射为标准 HGNC gene symbol
    """

    if gene_col not in df.columns:
        raise ValueError(f"CSV 必须包含列 '{gene_col}'")

    raw_genes = df[gene_col].dropna().astype(str).tolist()

    # 去掉非标准后缀并大写
    cleaned_genes = [g.split('_')[0].split('-')[0].upper() for g in raw_genes]

    mg = mygene.MyGeneInfo()
    query_scope = 'symbol' if id_type=='symbol' else 'uniprot'
    mapping = mg.querymany(cleaned_genes, scopes=query_scope, fields='symbol', species='human', as_dataframe=True)

    mapped_genes = mapping['symbol'].dropna().unique().tolist()
    return mapped_genes

def load_gene_sets(gene_set, local_gmt_path):
    lib = None
    source = None
    local_path = Path(local_gmt_path)
    if not local_path.is_absolute():
        local_path = PROJECT_DIR / local_path

    if local_path.exists():
        lib = gp.parser.read_gmt(str(local_path))
        source = "local_gmt"
    elif os.getenv("SC_ENRICH_ALLOW_ONLINE", "0") == "1":
        try:
            lib = gp.get_library(gene_set)
            source = "Enrichr_online"
        except Exception:
            lib = None

    if lib is None:
        raise RuntimeError(f"Gene set load failed for {gene_set}. Missing local GMT: {local_path}")

    return lib, source


def expand_dataframe(df):
    rows = []
    for _, row in df.iterrows():
        proteins = [x.strip() for x in str(row.get("PG.ProteinGroups", "")).split(";") if x.strip()]
        genes = [x.strip() for x in str(row.get("PG.Genes", "")).split(";") if x.strip()]
        if not genes:
            continue
        if not proteins:
            proteins = [""]
        for idx, gene in enumerate(genes):
            new_row = row.copy()
            new_row["PG.Genes"] = gene
            if len(proteins) == len(genes):
                new_row["PG.ProteinGroups"] = proteins[idx]
            elif idx < len(proteins):
                new_row["PG.ProteinGroups"] = proteins[idx]
            else:
                new_row["PG.ProteinGroups"] = proteins[0]
            rows.append(new_row)

    if not rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def build_ranking(subdf, gene_col, ranking_col):
    ranking = (
        subdf
        .dropna(subset=[gene_col, ranking_col])
        .drop_duplicates(subset=gene_col)
        .set_index(gene_col)[ranking_col]
    )

    # ===== 处理重复值（避免 GSEA 警告）=====
    if ranking.duplicated().any():
        # 方法1：加微小扰动（推荐，影响最小）
        ranking = ranking + np.random.normal(0, 1e-6, size=len(ranking))

        # 方法2（可选）：使用 rank 打破平局
        # ranking = ranking.rank(method="first")

    return ranking


def run_single_gsea(subdf, contrast, gene_col, ranking_col,
                    gene_sets_input, permutation_num,
                    min_size, max_size, fdr_threshold):

    safe_contrast = contrast.replace(" ", "")
    ranking = build_ranking(subdf, gene_col, ranking_col)
    if len(ranking) == 0:
        return safe_contrast, "No valid ranking genes"

    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets_input,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        outdir=None,
        no_plot=True,
    )

    res = pre_res.res2d.reset_index()

    temp = {"upregulated": [], "downregulated": []}

    for _, row in res.iterrows():
        fdr = row.get("FDR q-val") or row.get("fdr") or 1.0
        nes = row.get("NES", 0)
        term = row.get("Term") or row.get("Gene_set")
        lead_genes = row.get("Lead_genes") or ""

        if fdr <= fdr_threshold:
            status = "upregulated" if nes > 0 else "downregulated"
            temp[status].append({
                "pathway": term,
                "NES": round(nes, 4),
                "FDR": round(float(fdr), 4),
                "genes": lead_genes
            })

    return safe_contrast, temp

@tool
def run_gsea_enrichment(
    contrast_type="ALL",
    species="human",
    min_size=15,
    max_size=2000,
    fdr_threshold=0.25,
    permutation_num=500,
    n_jobs=-1,
    *args, **kwargs
):
    """
    run GSEA enrichment for all contrasts in processed_proteins folder
    """

    this_run_folder = resolve_path('this_run_folder_path')

    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()


    # 为了方便测试，直接调用查询好的结果
    out_folder = os.path.join(this_run_folder, "enrichment_results")
    os.makedirs(out_folder, exist_ok=True)
    gsea_path = os.path.join(out_folder, "gsea_results.json")

    if os.path.exists(gsea_path):
        pre_info = load_json(gsea_path)
        # if contrast_type.lower() == "all":
        #     return pre_info
        # else:
        if contrast_type in pre_info:
            return pre_info[contrast_type]
        if normalize_contrast(contrast_type) in pre_info:
            return pre_info[normalize_contrast(contrast_type)]


    gene_col = "PG.Genes"
    ranking_col = "logFC"
    contrast_col = "contrast"
    convert = False

    if species.lower() in ["human", "hs", "homo_sapiens"]:
        gene_set = "KEGG_2021_Human"
        local_gmt_path = str(offline_enrich.resolve_namespace_path("KEGG", "human"))

    elif species.lower() in ["mouse", "mm", "mus_musculus"]:
        gene_set = "KEGG_2021_Mouse"
        local_gmt_path = str(offline_enrich.resolve_namespace_path("KEGG", "mouse"))
        convert = True
    else:
        raise ValueError(f"Unsupported species: {species}")

    lib, source = load_gene_sets(gene_set, local_gmt_path)
    local_gmt_resolved = Path(local_gmt_path)
    if not local_gmt_resolved.is_absolute():
        local_gmt_resolved = PROJECT_DIR / local_gmt_resolved
    gene_sets_input = gene_set if source == "Enrichr_online" else str(local_gmt_resolved)

    differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv") and contrast_type.lower() in fn.lower()]
    if not differential_proteins_path_list:
        differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv")]


    results = {}
    for file in differential_proteins_path_list:
        df = pd.read_csv(os.path.join(processed_proteins_folder, file))
        if convert:
            df["PG.Genes"] = df["PG.Genes"].str.capitalize()
        if ranking_col not in df.columns:
            raise ValueError(f"Missing column {ranking_col}")
        df_expanded = expand_dataframe(df)
        grouped = list(df_expanded.groupby(contrast_col))
        res_list = Parallel(n_jobs=n_jobs)(
            delayed(run_single_gsea)(
                subdf,
                contrast,
                gene_col,
                ranking_col,
                gene_sets_input,
                permutation_num,
                min_size,
                max_size,
                fdr_threshold
            )
            for contrast, subdf in grouped
        )

        results.update(dict(res_list))

    summary = {
        "GSEA parameters": {
            "gene_set": gene_set,
            "min_size": min_size,
            "max_size": max_size,
            "fdr_threshold": fdr_threshold,
        },
        "GeneSet_metadata": {
            "database": gene_set,
            "source": source,
            "library_size": len(lib),
            "total_unique_genes": len(set(sum(lib.values(), []))),
        },
        "results": results
    }

    out_folder = os.path.join(this_run_folder, "enrichment_results")
    os.makedirs(out_folder, exist_ok=True)
    gsea_path = os.path.join(out_folder, "gsea_results.json")
    final_results = {contrast_type: results,}
    save_json(final_results, gsea_path)


    return summary


# ---------------------------------------------------------------------------
# Dataset figure recipe layer (figure_upgrade_20260904 Step 05)
# ---------------------------------------------------------------------------

# Registry for recipe plot functions (Step 06 implements them). A function may
# either register here (robust to definition order) or simply live in the tools
# module namespace under the exact declared name.
_FIGURE_RECIPE_FUNCTION_REGISTRY: Dict[str, Any] = {}


def register_figure_recipe_function(name: str):
    """Decorator: register ``name`` so the recipe dispatcher can find the plot function."""
    def _decorate(fn):
        _FIGURE_RECIPE_FUNCTION_REGISTRY[str(name)] = fn
        return fn
    return _decorate


def _resolve_figure_recipe_function(name: str) -> Any:
    fn = _FIGURE_RECIPE_FUNCTION_REGISTRY.get(str(name))
    if fn is not None:
        return fn
    candidate = globals().get(str(name))
    return candidate if callable(candidate) else None


def _resolve_recipe_dataset_name(this_run_folder: str) -> str:
    """Best-effort dataset name: analysis design -> evaluation requirements -> run folder."""
    candidates: List[str] = []
    design = _get_analysis_design()
    input_folder = str(design.get("input_folder", "") or "")
    if input_folder:
        candidates.append(os.path.basename(os.path.normpath(input_folder)))
    requirements_path = os.path.join(this_run_folder, "evaluation_requirements.used.json")
    if _path_exists(requirements_path):
        try:
            with open(_windows_long_path(requirements_path), "r", encoding="utf-8") as handle:
                requirements = json.load(handle)
            candidates.append(str((requirements or {}).get("dataset", "") or "").strip())
        except Exception:
            pass
    candidates.append(os.path.basename(os.path.normpath(this_run_folder)))
    for candidate in candidates:
        matched = figure_recipes.match_dataset_from_folder(candidate)
        if matched:
            return matched
    return ""


def _resolve_recipe_input_paths(this_run_folder: str, input_tables) -> tuple:
    """Resolve recipe input tables to run-artifact paths.

    ``SampleInfo.csv`` resolves through ``resolve_path('sampleinfo_path')``;
    everything else is relative to the run folder. Missing tables map to "".
    """
    input_paths: Dict[str, str] = {}
    for table in input_tables:
        path = ""
        if str(table) == "SampleInfo.csv":
            try:
                path = resolve_path('sampleinfo_path')
            except Exception:
                path = ""
        else:
            path = os.path.join(this_run_folder, str(table))
        input_paths[str(table)] = path if path and _path_exists(path) else ""
    missing = [table for table, path in input_paths.items() if not path]
    return input_paths, missing


def _render_dataset_figure_recipes(
        this_run_folder: str,
        *,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        protein_gene_map: pd.DataFrame,
        figure_records: List[Dict[str, Any]],
        visualize_params: Dict[str, Any],
        figure_recipe_params: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
    """Render per-dataset recipe figures after the generic figure set.

    Backward-compatible degradation rules: a missing plot function (Step 06
    backlog) is recorded as ``pending_function``; a recipe whose declared input
    tables are all absent is skipped as ``missing_inputs`` (no placeholder
    image, Step 04 rule); a failing plot function is recorded as ``error``.
    Unknown datasets fall back to the generic figure set only. A per-run
    ``figure_recipe_manifest.json`` documents every decision.
    """
    summary: Dict[str, Any] = {
        "enabled": True,
        "dataset": "",
        "n_recipes": 0,
        "n_rendered": 0,
        "n_pending_function": 0,
        "n_missing_inputs": 0,
        "n_errors": 0,
        "n_omitted": 0,
        "recipes": [],
    }
    dataset = _resolve_recipe_dataset_name(this_run_folder)
    summary["dataset"] = dataset
    recipes = figure_recipes.get_recipes_for_dataset(dataset)
    if not recipes:
        summary["note"] = "no dataset figure recipes; generic figure set only"
        return summary
    summary["n_recipes"] = int(len(recipes))
    overrides = figure_recipe_params if isinstance(figure_recipe_params, dict) else {}
    for recipe in recipes:
        entry: Dict[str, Any] = {
            "recipe_id": recipe.recipe_id,
            "dataset": recipe.dataset,
            "figure_kind": recipe.figure_kind,
            "paper_figure": recipe.paper_figure,
            "caption_slug": recipe.caption_slug,
            "plot_function": recipe.plot_function,
            "title": recipe.title,
            "out_file": recipe.out_file,
            "input_tables": list(recipe.input_tables),
        }
        summary["recipes"].append(entry)
        input_paths, missing_inputs = _resolve_recipe_input_paths(this_run_folder, recipe.input_tables)
        entry["missing_inputs"] = missing_inputs
        plot_fn = _resolve_figure_recipe_function(recipe.plot_function)
        if plot_fn is None:
            entry["status"] = "pending_function"
            summary["n_pending_function"] += 1
            continue
        available_inputs = {table: path for table, path in input_paths.items() if path}
        if not available_inputs:
            entry["status"] = "missing_inputs"
            summary["n_missing_inputs"] += 1
            continue
        params = dict(recipe.params)
        params.update(dict(overrides.get(recipe.recipe_id, {}) or {}))
        try:
            info = plot_fn(
                this_run_folder,
                data_BC=data_BC,
                sampleinfo=sampleinfo,
                protein_gene_map=protein_gene_map,
                input_paths=input_paths,
                params=params,
            )
        except Exception as exc:  # a recipe failure must never break visualize()
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            summary["n_errors"] += 1
            continue
        file_path = str(info.get("file_path", "") or "") if isinstance(info, dict) else str(info or "")
        if not file_path:
            entry["status"] = "omitted"
            entry["omitted_reason"] = (
                str(info.get("omitted_reason", "") or "no matching data")
                if isinstance(info, dict) else "empty result"
            )
            summary["n_omitted"] += 1
            continue
        kind_label = "原论文等价图" if recipe.figure_kind == "original_equivalent" else "拓展图"
        paper_note = f"（对应原论文图 {recipe.paper_figure}）" if recipe.paper_figure else ""
        caption = f"[{kind_label}{paper_note}] {recipe.title}：{recipe.description}"
        record_params = dict(visualize_params)
        record_params.update({
            "recipe_id": recipe.recipe_id,
            "figure_kind": recipe.figure_kind,
            "paper_figure": recipe.paper_figure,
        })
        _add_figure_record(
            figure_records,
            info,
            "Dataset recipe",
            caption,
            inputs=list(available_inputs.values()),
            params=record_params,
            plot_type=recipe.plot_type,
        )
        entry["status"] = "rendered"
        entry["file_path"] = file_path
        summary["n_rendered"] += 1
    try:
        manifest_path = os.path.join(
            _build_output_dir(this_run_folder, 'visualize_results'),
            "figure_recipe_manifest.json",
        )
        with open(_windows_long_path(manifest_path), "w", encoding="utf-8") as handle:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": summary["dataset"],
                "n_recipes": summary["n_recipes"],
                "n_rendered": summary["n_rendered"],
                "n_pending_function": summary["n_pending_function"],
                "n_missing_inputs": summary["n_missing_inputs"],
                "n_errors": summary["n_errors"],
                "n_omitted": summary["n_omitted"],
                "recipes": summary["recipes"],
            }, handle, ensure_ascii=False, indent=2)
        summary["manifest"] = manifest_path
    except Exception:
        pass
    return summary


# ---------------------------------------------------------------------------
# Dataset recipe plot functions (figure_upgrade_20260904 Step 06)
#
# Every function below implements one recipe declared in figure_recipes.py and
# follows the dispatcher calling contract::
#
#     info = fn(this_run_folder, *, data_BC, sampleinfo, protein_gene_map,
#               input_paths, params)
#
# Shared rules (Step 02 + Step 04): no meta/placeholder text inside figures —
# thresholds, effect sizes and omission reasons travel in the returned caption
# metadata; data gaps return an empty ``file_path`` (structured omission)
# instead of an empty figure; every output goes through ``_build_output_path``
# + ``_save_figure``; declared run artifacts are preferred over the in-memory
# matrices so the functions also work in reduced smoke environments.
# ---------------------------------------------------------------------------


def _recipe_param(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(params, dict):
        return default
    value = params.get(key, default)
    return default if value is None else value


def _recipe_input_csv(input_paths: Dict[str, str], table: str) -> pd.DataFrame:
    """Read one declared recipe input table; missing/unreadable → empty frame."""
    path = input_paths.get(str(table), "") if isinstance(input_paths, dict) else ""
    if not path or not _path_exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(_windows_long_path(path))
    except Exception:
        return pd.DataFrame()


def _recipe_input_json(input_paths: Dict[str, str], table: str) -> Dict[str, Any]:
    path = input_paths.get(str(table), "") if isinstance(input_paths, dict) else ""
    if not path or not _path_exists(path):
        return {}
    try:
        with open(_windows_long_path(path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _recipe_effective_sampleinfo(sampleinfo: pd.DataFrame, input_paths: Dict[str, str]) -> pd.DataFrame:
    """Sampleinfo with ``FileName``+``Cluster``; declared artifacts fill the gap."""
    if sampleinfo is not None and not sampleinfo.empty and {"FileName", "Cluster"}.issubset(sampleinfo.columns):
        return sampleinfo
    for table in ["processed_proteins/SampleInfo_Filtered.csv", "SampleInfo.csv"]:
        df = _recipe_input_csv(input_paths, table)
        if not df.empty and {"FileName", "Cluster"}.issubset(df.columns):
            return df
    return sampleinfo if sampleinfo is not None else pd.DataFrame()


def _recipe_sample_columns(data_BC: pd.DataFrame, sampleinfo: pd.DataFrame) -> List[str]:
    """Sample columns of ``data_BC`` ordered as in ``sampleinfo['FileName']``."""
    if data_BC is None or data_BC.empty or sampleinfo is None or sampleinfo.empty:
        return []
    if "FileName" not in sampleinfo.columns:
        return []
    seen: set = set()
    cols: List[str] = []
    for name in sampleinfo["FileName"].astype(str).tolist():
        if name in data_BC.columns and name not in seen:
            seen.add(name)
            cols.append(name)
    return cols


def _recipe_quant_matrix(input_paths: Dict[str, str], data_BC: pd.DataFrame,
                         table: str = "processed_proteins/ProteinQuant_ComBat.csv"):
    """Prefer the declared run artifact; fall back to the in-memory matrix."""
    df = _recipe_input_csv(input_paths, table)
    if not df.empty:
        return df, "run_artifact"
    matrix = data_BC if isinstance(data_BC, pd.DataFrame) else pd.DataFrame()
    return matrix, "in_memory"


def _recipe_gene_index(data_BC: pd.DataFrame, protein_gene_map: pd.DataFrame) -> Dict[str, Any]:
    """Upper-case gene symbol → matrix row positions (+ original display labels).

    ``PG.Genes`` on the quant matrix is authoritative; ``protein_gene_map`` is
    consulted only when the matrix carries no gene column.
    """
    empty = {"by_gene": {}, "label": {}, "id_col": "", "n_rows": 0}
    if data_BC is None or data_BC.empty:
        return empty
    id_col = next(
        (col for col in ["PG.ProteinGroups", "Protein.Group", "ProteinGroups", "protein", "ProteinID"]
         if col in data_BC.columns),
        data_BC.columns[0],
    )
    ids = data_BC[id_col].astype(str)
    if "PG.Genes" in data_BC.columns:
        genes = data_BC["PG.Genes"]
    else:
        genes = pd.Series([""] * len(data_BC), index=data_BC.index, dtype="object")
        if protein_gene_map is not None and not protein_gene_map.empty:
            map_id_col = next(
                (col for col in ["PG.ProteinGroups", "Protein.Group", "ProteinGroups", "protein"]
                 if col in protein_gene_map.columns), None)
            map_gene_col = next(
                (col for col in ["PG.Genes", "Genes", "Gene", "gene"] if col in protein_gene_map.columns), None)
            if map_id_col and map_gene_col:
                mapping = protein_gene_map[[map_id_col, map_gene_col]].copy()
                mapping[map_id_col] = mapping[map_id_col].astype(str)
                lookup = dict(zip(mapping[map_id_col], mapping[map_gene_col].astype(str)))
                genes = ids.map(lambda value: lookup.get(value, ""))
    tokens = genes.astype(str).str.split(";")
    rows = pd.DataFrame({"row": np.arange(len(data_BC)), "id": ids.values, "token": tokens.values})
    rows = rows.explode("token")
    rows["key"] = rows["token"].astype(str).str.strip().str.upper()
    rows = rows[rows["key"].ne("") & rows["key"].ne("NAN")]
    if rows.empty:
        return empty
    by_gene: Dict[str, List[int]] = {}
    for token, group in rows.groupby("key", sort=False):
        by_gene[str(token)] = [int(x) for x in group["row"].tolist()]
    label: Dict[int, str] = {}
    first = rows.drop_duplicates("row")
    for row_pos, token in zip(first["row"], first["token"]):
        label[int(row_pos)] = str(token)
    return {"by_gene": by_gene, "label": label, "id_col": id_col, "n_rows": int(len(data_BC))}


# Historical HGNC renames seen in proteomics matrices: a declared marker using
# the legacy symbol must still match the matrix (e.g. BloodCell recipe H1F0 vs
# matrix symbol H1-0).
_RECIPE_GENE_ALIASES: Dict[str, str] = {
    "H1F0": "H1-0",
    "H1-0": "H1F0",
}


def _recipe_gene_matrix(data_BC: pd.DataFrame, gene_index: Dict[str, Any], wanted,
                        sample_cols: List[str]) -> pd.DataFrame:
    """Rows = requested genes found in the matrix, columns = sample columns."""
    if not sample_cols or not gene_index.get("n_rows"):
        return pd.DataFrame()
    collected: Dict[str, pd.Series] = {}
    for gene in wanted:
        key = str(gene).strip().upper()
        positions = gene_index["by_gene"].get(key, [])
        if not positions:
            alias = _RECIPE_GENE_ALIASES.get(key, "")
            positions = gene_index["by_gene"].get(alias, []) if alias else []
        if not positions:
            continue
        block = data_BC.iloc[positions][sample_cols].apply(pd.to_numeric, errors="coerce")
        collected[str(gene)] = block.mean(axis=0, skipna=True)
    if not collected:
        return pd.DataFrame()
    return pd.DataFrame(collected).T


def _recipe_group_series(sampleinfo: pd.DataFrame, sample_cols: List[str],
                         group_col: str = "Cluster") -> pd.Series:
    """Normalized group label per sample column (FileName-aligned)."""
    labels = pd.Series(np.full(len(sample_cols), np.nan), index=pd.Index(sample_cols), dtype="object")
    if sampleinfo is None or sampleinfo.empty or group_col not in sampleinfo.columns or "FileName" not in sampleinfo.columns:
        return labels
    dedup = sampleinfo.drop_duplicates(subset=["FileName"])
    mapping = dict(zip(dedup["FileName"].astype(str), dedup[group_col]))
    for sample in sample_cols:
        if sample in mapping:
            labels.loc[sample] = mapping[sample]
    return _normalize_group_labels(labels)


def _recipe_group_mean_matrix(mat: pd.DataFrame, group_series: pd.Series,
                              group_order=None) -> pd.DataFrame:
    """Mean per group (columns), ordered by ``group_order`` when provided."""
    if mat is None or mat.empty:
        return pd.DataFrame()
    grouped: Dict[str, pd.Series] = {}
    for group in pd.unique(group_series):
        samples = [s for s in mat.columns if group_series.get(s) == group]
        if samples:
            block = mat[samples].apply(pd.to_numeric, errors="coerce")
            grouped[str(group)] = block.mean(axis=1, skipna=True)
    if not grouped:
        return pd.DataFrame()
    mean_df = pd.DataFrame(grouped)
    if group_order:
        ordered = [str(g) for g in group_order if str(g) in mean_df.columns]
        ordered += [c for c in mean_df.columns if c not in ordered]
        mean_df = mean_df[ordered]
    return mean_df


def _recipe_row_zscore(mat: pd.DataFrame, clip: float = 2.5) -> pd.DataFrame:
    if mat is None or mat.empty:
        return pd.DataFrame()
    num = mat.apply(pd.to_numeric, errors="coerce")
    z = num.sub(num.mean(axis=1), axis=0).div(num.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    return z.fillna(0.0).clip(-clip, clip)


def _recipe_ordered_present(order, values) -> List[str]:
    present = [str(v) for v in values]
    ordered = [str(g) for g in (order or []) if str(g) in present]
    ordered += [v for v in present if v not in ordered]
    return ordered


def _recipe_differential_table(input_paths: Dict[str, str], tables) -> pd.DataFrame:
    """Normalized logFC/FDR rows from declared differential tables."""
    frames = []
    for table in tables:
        df = _recipe_input_csv(input_paths, table)
        if df.empty or "logFC" not in df.columns or "adj.P.Val" not in df.columns:
            continue
        out = pd.DataFrame(index=df.index)
        id_col = next((c for c in ["PG.ProteinGroups", "protein", "Protein.Group", "ProteinGroups"]
                       if c in df.columns), None)
        gene_col = next((c for c in ["PG.Genes", "Genes", "Gene", "gene"] if c in df.columns), None)
        out["protein_id"] = df[id_col].astype(str) if id_col else ""
        out["gene_label"] = (df[gene_col].astype(str).str.split(";").str[0]
                             if gene_col else out["protein_id"])
        out["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
        out["adj.P.Val"] = pd.to_numeric(df["adj.P.Val"], errors="coerce")
        if "contrast" in df.columns:
            out["contrast"] = df["contrast"].astype(str)
        else:
            out["contrast"] = os.path.splitext(os.path.basename(str(table)))[0].replace("differential_", "")
        frames.append(out.dropna(subset=["logFC", "adj.P.Val"]))
    if not frames:
        return pd.DataFrame(columns=["protein_id", "gene_label", "logFC", "adj.P.Val", "contrast"])
    return pd.concat(frames, ignore_index=True)


def _recipe_logfc_lookup(diff_df: pd.DataFrame, wanted) -> Dict[str, Dict[str, float]]:
    """gene_upper → smallest-FDR logFC / adj.P / contrast across the tables."""
    lookup: Dict[str, Dict[str, float]] = {}
    if diff_df is None or diff_df.empty:
        return lookup
    df = diff_df.copy()
    df["gene_key"] = df["gene_label"].astype(str).str.strip().str.upper()
    df = df[df["gene_key"].ne("") & df["gene_key"].ne("NAN")].sort_values("adj.P.Val")
    for gene in wanted:
        key = str(gene).strip().upper()
        rows = df[df["gene_key"] == key]
        if rows.empty:
            continue
        first = rows.iloc[0]
        lookup[key] = {
            "logFC": float(first["logFC"]),
            "adj_p": float(first["adj.P.Val"]),
            "contrast": str(first.get("contrast", "")),
        }
    return lookup


def _recipe_enrichment_table(input_paths: Dict[str, str], tables) -> pd.DataFrame:
    """Normalized ORA rows (Description / p.adjust / Count / direction) from enrichment CSVs."""
    frames = []
    for table in tables:
        df = _recipe_input_csv(input_paths, table)
        if df.empty or "Description" not in df.columns or "p.adjust" not in df.columns:
            continue
        out = pd.DataFrame(index=df.index)
        out["Description"] = df["Description"].astype(str)
        out["p.adjust"] = pd.to_numeric(df["p.adjust"], errors="coerce")
        out["Count"] = pd.to_numeric(df["Count"], errors="coerce") if "Count" in df.columns else np.nan
        out["namespace"] = df["namespace"].astype(str) if "namespace" in df.columns else ""
        text = str(table)
        out["direction"] = "downregulated" if "downregulated" in text else "upregulated"
        if "contrast" in df.columns:
            out["contrast"] = df["contrast"].astype(str)
        else:
            parsed = _parse_enrichment_file_name(os.path.basename(text))
            out["contrast"] = parsed["contrast"]
            out["direction"] = parsed["direction"] if parsed["direction"] != "unknown" else out["direction"]
        frames.append(out.dropna(subset=["p.adjust"]))
    if not frames:
        return pd.DataFrame(columns=["Description", "p.adjust", "Count", "namespace", "direction", "contrast"])
    return pd.concat(frames, ignore_index=True)


def _recipe_module_group_matrix(group_summary: pd.DataFrame, modules=None, group_order=None) -> pd.DataFrame:
    """mean_score matrix (modules × groups); ``*_minus_*`` delta rows excluded."""
    if (group_summary is None or group_summary.empty
            or not {"module", "group", "mean_score"}.issubset(group_summary.columns)):
        return pd.DataFrame()
    df = group_summary.copy()
    df = df[~df["group"].astype(str).str.contains("_minus_", regex=False)]
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    if modules:
        wanted = [str(m) for m in modules]
        df = df[df["module"].astype(str).isin(wanted)]
    if df.empty:
        return pd.DataFrame()
    matrix = df.pivot_table(index="module", columns="group", values="mean_score", aggfunc="mean")
    if group_order:
        ordered = [str(g) for g in group_order if str(g) in matrix.columns]
        ordered += [c for c in matrix.columns if c not in ordered]
        matrix = matrix[ordered]
    return matrix


def _recipe_pca_coords(quant: pd.DataFrame, sample_cols: List[str], max_proteins: int = 2000) -> pd.DataFrame:
    """Two-component PCA over samples using the highest-variance proteins."""
    if not sample_cols:
        return pd.DataFrame()
    block = quant[sample_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if block.empty:
        return pd.DataFrame()
    variances = block.var(axis=1, skipna=True).replace(0, np.nan).dropna()
    if variances.empty:
        return pd.DataFrame()
    top = variances.sort_values(ascending=False).head(max_proteins).index
    block_t = block.loc[top].T
    block_t = block_t.fillna(block.loc[top].mean(axis=1)).fillna(0.0)
    block_t.columns = block_t.columns.astype(str)
    if block_t.shape[0] < 3 or block_t.shape[1] < 3:
        return pd.DataFrame()
    coords = PCA(n_components=2, random_state=0).fit_transform(block_t.to_numpy(dtype=float))
    return pd.DataFrame(coords, index=block_t.index, columns=["PC1", "PC2"])


def _recipe_save_info(file_path: str, fig, **metadata) -> Dict[str, Any]:
    """Save through the standard chain and attach caption metadata (None dropped)."""
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    info: Dict[str, Any] = {"file_path": file_path}
    info.update({key: value for key, value in metadata.items() if value is not None})
    return info


def _recipe_direction_heatmap(ax, matrix: pd.DataFrame, *, cbar_label: str = "log2 Fold Change",
                              sig_mask: pd.DataFrame | None = None) -> None:
    """Diverging heatmap with numeric annotation; significant cells get a ring."""
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    values = matrix.apply(pd.to_numeric, errors="coerce")
    vmax = float(np.nanmax(np.abs(values.to_numpy(dtype=float)))) if values.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    annot = values.round(2).astype(str) if values.size else None
    if annot is not None:
        annot = annot.replace("nan", "")
    sns.heatmap(values, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax, rasterized=True,
                annot=annot, fmt="",
                linewidths=0.6, linecolor="white",
                cbar_kws={"label": cbar_label}, annot_kws={"fontsize": 7.5})
    if sig_mask is not None and sig_mask.shape == values.shape:
        ys, xs = np.where(sig_mask.to_numpy(dtype=bool))
        ax.scatter(xs + 0.5, ys + 0.5, s=150, facecolors="none", edgecolors="#111111", linewidths=1.3)
    _apply_nature_heatmap_frame(ax)


def plot_recipe_pispa_cluster_type_composition(this_run_folder: str, *, data_BC, sampleinfo,
                                               protein_gene_map, input_paths, params):
    """PiSPA Fig. 5b equivalent: cluster × migration-state composition bars."""
    cross = _recipe_input_csv(input_paths, "evaluation_evidence/group_cross_table_Cluster_by_Type.csv")
    context_col = str(_recipe_param(params, "context_col", "Type"))
    if cross.empty:
        base = sampleinfo
        if not isinstance(base, pd.DataFrame) or base.empty or context_col not in base.columns:
            base = _recipe_input_csv(input_paths, "SampleInfo.csv")
        if not base.empty and "Cluster" in base.columns and context_col in base.columns:
            cross = pd.crosstab(base["Cluster"].astype(str), base[context_col].astype(str))
    if cross.empty:
        return _placeholder_plot(this_run_folder, "recipe_pispa_cluster_type_composition.png",
                                 "No cluster×state composition table",
                                 "group_cross_table_Cluster_by_Type.csv was absent and the sample sheet has no context column")
    if "Cluster" in cross.columns:
        cross = cross.set_index("Cluster")
    cluster_order = _recipe_param(params, "cluster_order", [])
    clusters = _recipe_ordered_present(cluster_order, cross.index.astype(str))
    if not clusters:
        return _placeholder_plot(this_run_folder, "recipe_pispa_cluster_type_composition.png",
                                 "No clusters available", "The composition table has no usable cluster rows")
    cross = cross.loc[clusters].apply(pd.to_numeric, errors="coerce").fillna(0)
    categories = [str(c) for c in cross.columns]
    palette = _categorical_palette(categories)
    file_path = _build_output_path(this_run_folder, "recipe_pispa_cluster_type_composition.png")
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.1]})
    x = np.arange(len(clusters))
    bottom = np.zeros(len(clusters))
    for category in categories:
        values = cross[category].to_numpy(dtype=float)
        axes[0].bar(x, values, bottom=bottom, color=palette[category], edgecolor="white",
                    linewidth=0.6, label=category)
        bottom += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clusters)
    axes[0].set_ylabel("Cells (count)")
    axes[0].set_title("Composition counts")
    _apply_nature_axes(axes[0])
    _legend_outside(axes[0], title=context_col)
    totals = cross.sum(axis=1).replace(0, np.nan)
    share = cross.div(totals, axis=0)
    bottom = np.zeros(len(clusters))
    for category in categories:
        values = share[category].fillna(0).to_numpy(dtype=float) * 100
        axes[1].bar(x, values, bottom=bottom, color=palette[category], edgecolor="white", linewidth=0.6)
        for xi, (value, base_value) in enumerate(zip(values, bottom)):
            if value >= 6:
                axes[1].text(xi, base_value + value / 2, f"{value:.0f}%", ha="center", va="center",
                             fontsize=8.5, color="white" if value >= 25 else "#333333")
        bottom += values
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clusters)
    axes[1].set_ylabel("Share of cells (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Composition share")
    _apply_nature_axes(axes[1])
    fig.suptitle("Cluster × cell-state composition", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    migrated_label = str(_recipe_param(params, "migrated_label", "Migrated"))
    migrated_col = next((c for c in cross.columns if migrated_label.lower() in str(c).lower()), None)
    migrated_share: Dict[str, float] = {}
    if migrated_col is not None:
        for cluster in clusters:
            total = float(cross.loc[cluster].sum())
            migrated_share[cluster] = round(float(cross.loc[cluster, migrated_col]) / total, 3) if total else 0.0
    return _recipe_save_info(
        file_path, fig,
        n_unannotated=_count_unannotated_labels(cross.index),
        thresholds={"migrated_share_by_cluster": migrated_share} if migrated_share else None,
        filter_note="left panel: absolute counts; right panel: within-cluster share",
    )


def plot_recipe_pispa_rho_gtpase_candidate_boxplot(this_run_folder: str, *, data_BC, sampleinfo,
                                                  protein_gene_map, input_paths, params):
    """PiSPA Fig. 5f equivalent: candidate-gene abundance boxplots across clusters."""
    genes = [str(g) for g in _recipe_param(params, "genes", [])]
    group_order = [str(g) for g in _recipe_param(params, "group_order", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, genes, sample_cols)
    if mat.empty:
        return _placeholder_plot(this_run_folder, "recipe_pispa_rho_gtpase_candidate_boxplot.png",
                                 "No candidate genes matched",
                                 "None of the declared Rho GTPase/ERM/adhesion genes were found in the quantified matrix")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    long_rows = []
    for gene in mat.index:
        values = pd.to_numeric(mat.loc[gene], errors="coerce")
        for sample, value in values.items():
            if pd.notna(value):
                long_rows.append({"gene": gene, "group": str(group_series.get(sample, UNANNOTATED_GROUP_LABEL)),
                                  "value": float(value)})
    long_df = pd.DataFrame(long_rows)
    groups_present = _recipe_ordered_present(group_order, long_df["group"].unique())
    diff_tables = ([t for t in input_paths if str(t).startswith("processed_proteins/differential_")]
                   if isinstance(input_paths, dict) else [])
    gene_lookup = _recipe_logfc_lookup(_recipe_differential_table(input_paths, diff_tables), list(mat.index))
    n_sig = sum(1 for info in gene_lookup.values() if info["adj_p"] < 0.05)
    n = len(mat.index)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.9 * nrows + 0.6), squeeze=False)
    palette = _categorical_palette(groups_present)
    for idx, gene in enumerate(mat.index):
        ax = axes[idx // ncols][idx % ncols]
        sub = long_df[long_df["gene"] == gene]
        data_by_group = [sub.loc[sub["group"] == g, "value"].to_numpy() for g in groups_present]
        boxes = ax.boxplot(data_by_group, patch_artist=True, widths=0.55, showfliers=False,
                           medianprops=dict(color="#222222", linewidth=1.2))
        for patch, group in zip(boxes["boxes"], groups_present):
            patch.set_facecolor(palette[group])
            patch.set_alpha(0.5)
            patch.set_edgecolor("#444444")
        rng = np.random.default_rng(17 + idx)
        for gi, values in enumerate(data_by_group):
            if len(values) == 0:
                continue
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(np.full(len(values), gi + 1) + jitter, values, s=10, color="#333333",
                       alpha=0.6, linewidths=0, rasterized=True)
        ax.set_xticks(range(1, len(groups_present) + 1))
        ax.set_xticklabels([_wrap_label(g, 14) for g in groups_present], fontsize=8, rotation=20, ha="right")
        ax.set_title(str(gene), fontsize=10)
        ax.set_ylabel("log2 abundance" if idx % ncols == 0 else "")
        _apply_nature_axes(ax)
    for leftover in range(n, nrows * ncols):
        axes[leftover // ncols][leftover % ncols].axis("off")
    fig.suptitle("Rho GTPase–cytoskeleton–adhesion candidates across clusters",
                 fontsize=15, weight="bold", y=1.01)
    plt.tight_layout()
    file_path = _build_output_path(this_run_folder, "recipe_pispa_rho_gtpase_candidate_boxplot.png")
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(n),
        n_sig=int(n_sig),
        thresholds={"pval_cutoff": 0.05,
                    "gene_logFC_adjP": {g: [round(v["logFC"], 3), float(f"{v['adj_p']:.3g}")]
                                        for g, v in gene_lookup.items()}} if gene_lookup else None,
        filter_note=f"abundance source: {quant_source}; per-gene pairwise logFC/FDR reported in caption metadata only",
    )


def plot_recipe_pispa_migration_module_gradient(this_run_folder: str, *, data_BC, sampleinfo,
                                                protein_gene_map, input_paths, params):
    """Extension: migration curated-module score distributions + group-mean gradient."""
    modules = [str(m) for m in _recipe_param(params, "modules", [])]
    group_order = [str(g) for g in _recipe_param(params, "group_order", [])]
    scores = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_sample_scores.csv")
    group_summary = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv")
    modules_present = [m for m in modules if not scores.empty and m in scores.columns]
    if not modules_present and group_summary.empty:
        return _placeholder_plot(this_run_folder, "recipe_pispa_migration_module_gradient.png",
                                 "No curated module scores",
                                 "curated_module_sample_scores.csv and curated_module_group_summary.csv are both absent")
    group_col = "Cluster"
    group_labels: pd.Series = pd.Series(dtype="object")
    n_per_group: Dict[str, int] = {}
    if not scores.empty and group_col in scores.columns and modules_present:
        group_labels = _normalize_group_labels(scores[group_col])
        for group in pd.unique(group_labels):
            n_per_group[str(group)] = int((group_labels == group).sum())
    groups_present = _recipe_ordered_present(group_order, list(n_per_group) or group_summary.get("group", pd.Series(dtype=str)).astype(str).unique())
    _set_visual_theme("notebook")
    if modules_present and groups_present:
        ncols = min(3, len(modules_present))
        nrows = int(np.ceil(len(modules_present) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows + 0.6), squeeze=False)
        palette = _categorical_palette(groups_present)
        module_means: Dict[str, Dict[str, float]] = {}
        for idx, module in enumerate(modules_present):
            ax = axes[idx // ncols][idx % ncols]
            data_by_group = []
            for group in groups_present:
                values = pd.to_numeric(scores.loc[group_labels == group, module], errors="coerce").dropna()
                data_by_group.append(values.to_numpy())
            boxes = ax.boxplot(data_by_group, patch_artist=True, widths=0.55, showfliers=False,
                               medianprops=dict(color="#222222", linewidth=1.2))
            for patch, group in zip(boxes["boxes"], groups_present):
                patch.set_facecolor(palette[group])
                patch.set_alpha(0.5)
                patch.set_edgecolor("#444444")
            rng = np.random.default_rng(31 + idx)
            for gi, values in enumerate(data_by_group):
                if len(values) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(values))
                ax.scatter(np.full(len(values), gi + 1) + jitter, values, s=9, color="#333333",
                           alpha=0.55, linewidths=0, rasterized=True)
            means = [float(np.mean(v)) if len(v) else np.nan for v in data_by_group]
            ax.plot(np.arange(1, len(groups_present) + 1), means, color="#B2182B", linewidth=1.6,
                    marker="D", markersize=4.5, label="Group mean")
            module_means[str(module)] = {g: round(m, 4) for g, m in zip(groups_present, means) if not np.isnan(m)}
            ax.set_xticks(range(1, len(groups_present) + 1))
            ax.set_xticklabels([_wrap_label(g, 13) for g in groups_present], fontsize=8.5, rotation=20, ha="right")
            ax.set_title(_wrap_label(module, 26), fontsize=10)
            if idx % ncols == 0:
                ax.set_ylabel("Module score")
            _apply_nature_axes(ax)
        for leftover in range(len(modules_present), nrows * ncols):
            axes[leftover // ncols][leftover % ncols].axis("off")
        fig.suptitle("Migration module score gradient across clusters", fontsize=15, weight="bold", y=1.01)
        plt.tight_layout()
        file_path = _build_output_path(this_run_folder, "recipe_pispa_migration_module_gradient.png")
        deltas = {m: round(v[groups_present[0]] - v[groups_present[-1]], 4)
                  for m, v in module_means.items() if groups_present[0] in v and groups_present[-1] in v}
        return _recipe_save_info(
            file_path, fig,
            thresholds={"module_group_means": module_means,
                        f"delta_{groups_present[0]}_minus_{groups_present[-1]}": deltas} if module_means else None,
            filter_note="violin-free box+strip per group; red line connects group means; effect deltas in caption metadata",
        )
    # Fallback: group-mean curves only, from the group summary table.
    matrix = _recipe_module_group_matrix(group_summary, modules, groups_present)
    if matrix.empty:
        return _placeholder_plot(this_run_folder, "recipe_pispa_migration_module_gradient.png",
                                 "No module group means", "Module score tables did not cover the declared modules")
    file_path = _build_output_path(this_run_folder, "recipe_pispa_migration_module_gradient.png")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(matrix.shape[1])
    palette = sns.color_palette("colorblind", n_colors=len(matrix.index))
    for color, (module, row) in zip(palette, matrix.iterrows()):
        ax.plot(x, row.to_numpy(dtype=float), marker="o", linewidth=1.8, color=color, label=str(module))
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(c, 14) for c in matrix.columns], rotation=20, ha="right")
    ax.set_ylabel("Group mean module score")
    ax.set_title("Curated migration module group means")
    _apply_nature_axes(ax)
    _legend_outside(ax, title="Module")
    plt.tight_layout()
    return _recipe_save_info(file_path, fig, thresholds={"module_group_means": matrix.round(4).to_dict(orient="index")})


def plot_recipe_brain_state_axis_module_gradient(this_run_folder: str, *, data_BC, sampleinfo,
                                                 protein_gene_map, input_paths, params):
    """Brain Fig. 6a,b equivalent: RG→oRG→IPC-EN→EN marker/module state gradient."""
    state_order = [str(s) for s in _recipe_param(params, "state_order", [])]
    markers = [str(m) for m in _recipe_param(params, "state_markers", [])]
    modules = [str(m) for m in _recipe_param(params, "modules", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    states_present = [s for s in state_order if s in set(group_series)] or sorted(set(group_series))
    marker_mat = _recipe_gene_matrix(quant, gene_index, markers, sample_cols)
    marker_means = _recipe_group_mean_matrix(marker_mat, group_series, states_present) if not marker_mat.empty else pd.DataFrame()
    marker_z = _recipe_row_zscore(marker_means)
    module_matrix = _recipe_module_group_matrix(
        _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv"),
        modules, states_present)
    if marker_z.empty and module_matrix.empty:
        return _placeholder_plot(this_run_folder, "recipe_brain_state_axis_module_gradient.png",
                                 "No state-axis evidence",
                                 "State markers could not be matched to the matrix and no curated module summary was available")
    _set_visual_theme("notebook")
    panels = [name for name, frame in [("markers", marker_z), ("modules", module_matrix)] if not frame.empty]
    omitted_panels = [name for name in ("markers", "modules") if name not in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels) + 2.4, max(4.0, 0.42 * max(len(marker_z), len(module_matrix), 1) + 1.6)),
                             squeeze=False)
    ax = axes[0][0]
    column = 0
    if not marker_z.empty:
        sns.heatmap(marker_z, ax=ax, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                    vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Mean row Z-score"})
        ax.set_title("Stage markers along the state axis")
        ax.set_xlabel("Developmental state")
        ax.set_ylabel("")
        _apply_nature_heatmap_frame(ax)
        ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=35, ha="right", fontsize=8.5)
        column += 1
    if not module_matrix.empty:
        ax2 = axes[0][column] if len(panels) > 1 else axes[0][0]
        if len(panels) == 1:
            ax2.clear()
        x = np.arange(module_matrix.shape[1])
        palette = sns.color_palette("colorblind", n_colors=len(module_matrix.index))
        for color, (module, row) in zip(palette, module_matrix.iterrows()):
            ax2.plot(x, row.to_numpy(dtype=float), marker="o", linewidth=1.8, color=color, label=str(module))
        ax2.set_xticks(x)
        ax2.set_xticklabels([_wrap_label(c, 12) for c in module_matrix.columns], rotation=15, ha="right")
        ax2.set_ylabel("Group mean module score")
        ax2.set_title("Curated module gradient")
        _apply_nature_axes(ax2)
        _legend_outside(ax2, title="Module")
    fig.suptitle("RG→oRG→IPC-EN→EN developmental state axis", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    file_path = _build_output_path(this_run_folder, "recipe_brain_state_axis_module_gradient.png")
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(marker_z)),
        omitted_panels=omitted_panels,
        filter_note=f"abundance source: {quant_source}; ASD/NDD kept outside this figure as external annotations only",
    )


def plot_recipe_brain_marker_cluster_heatmap(this_run_folder: str, *, data_BC, sampleinfo,
                                             protein_gene_map, input_paths, params):
    """Extension: canonical cell-type marker × cluster mean-abundance heatmap."""
    markers = [str(m) for m in _recipe_param(params, "markers", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, markers, sample_cols)
    if mat.empty:
        return _placeholder_plot(this_run_folder, "recipe_brain_marker_cluster_heatmap.png",
                                 "No markers matched",
                                 "None of the declared cell-type markers were found in the quantified matrix")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    mean_df = _recipe_group_mean_matrix(mat, group_series)
    if mean_df.empty or mean_df.shape[1] < 2:
        return _placeholder_plot(this_run_folder, "recipe_brain_marker_cluster_heatmap.png",
                                 "Cluster axis unavailable",
                                 "Fewer than two clusters were available for the marker heatmap")
    z_df = _recipe_row_zscore(mean_df)
    file_path = _build_output_path(this_run_folder, "recipe_brain_marker_cluster_heatmap.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * z_df.shape[1] + 2.0), max(3.6, 0.45 * z_df.shape[0] + 1.6)))
    sns.heatmap(z_df, ax=ax, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Mean row Z-score"})
    ax.set_title("Cell-type markers × cluster mean abundance")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    _apply_nature_heatmap_frame(ax)
    _wrap_axis_ticklabels(ax, "x", width=16, rotation=30)
    plt.tight_layout()
    unmatched = [m for m in markers if m not in z_df.index]
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(z_df)),
        thresholds={"unmatched_markers": unmatched} if unmatched else None,
        filter_note=f"abundance source: {quant_source}",
    )


def plot_recipe_dvp_zonation_zone_axis_heatmap(this_run_folder: str, *, data_BC, sampleinfo,
                                               protein_gene_map, input_paths, params):
    """DVP Fig. 3c,d,e equivalent: zone-ordered z-score heatmap + marker gradients."""
    zone_order = [str(z) for z in _recipe_param(params, "zone_order", [])]
    declared_markers = [str(m) for m in _recipe_param(params, "markers", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    diff_df = _recipe_differential_table(input_paths, ["processed_proteins/differential_Central_vs_Portal.csv"])
    top_genes: List[str] = []
    ranking_source = ""
    if not diff_df.empty:
        top = diff_df.reindex(diff_df["logFC"].abs().sort_values(ascending=False).index)
        top_genes = [g for g in top["gene_label"].map(
            lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v)
        ).tolist() if g and g.upper() != "NAN"][:15]
        ranking_source = "differential_Central_vs_Portal top |logFC|"
    if len(top_genes) < 5:
        top_genes = declared_markers
        ranking_source = "declared zone markers"
    mat = _recipe_gene_matrix(quant, gene_index, top_genes, sample_cols)
    groups_present = [z for z in zone_order if z in set(group_series)] or sorted(set(group_series))
    mean_df = _recipe_group_mean_matrix(mat, group_series, groups_present) if not mat.empty else pd.DataFrame()
    if mean_df.empty or mean_df.shape[1] < 2:
        return _placeholder_plot(this_run_folder, "recipe_dvp_zonation_zone_axis_heatmap.png",
                                 "Zonation axis unavailable",
                                 "Zone proteins could not be matched to the matrix or fewer than two declared zones were present")
    z_df = _recipe_row_zscore(mean_df)
    marker_mat = _recipe_gene_matrix(quant, gene_index, declared_markers, sample_cols)
    marker_curves = _recipe_group_mean_matrix(marker_mat, group_series, groups_present) if not marker_mat.empty else pd.DataFrame()
    marker_z = _recipe_row_zscore(marker_curves)
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(4.2, 0.32 * len(z_df) + 2.0)),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    sns.heatmap(z_df, ax=axes[0], cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Mean row Z-score"})
    axes[0].set_title("Top zonation proteins")
    axes[0].set_xlabel("Zone")
    axes[0].set_ylabel("")
    _apply_nature_heatmap_frame(axes[0])
    _wrap_axis_ticklabels(axes[0], "x", width=14, rotation=0)
    if not marker_z.empty:
        x = np.arange(marker_z.shape[1])
        palette = sns.color_palette("colorblind", n_colors=len(marker_z.index))
        for color, (gene, row) in zip(palette, marker_z.iterrows()):
            axes[1].plot(x, row.to_numpy(dtype=float), marker="o", linewidth=1.8, color=color, label=str(gene))
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([_wrap_label(c, 14) for c in marker_z.columns], rotation=15, ha="right")
        axes[1].set_ylabel("Marker Z-score (zone means)")
        axes[1].set_title("Zone marker gradients")
        _apply_nature_axes(axes[1])
        _legend_outside(axes[1], title="Marker")
    else:
        axes[1].axis("off")
    fig.suptitle("Portal–Midlobular–Central zonation axis", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    file_path = _build_output_path(this_run_folder, "recipe_dvp_zonation_zone_axis_heatmap.png")
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(z_df)),
        omitted_panels=[] if not marker_z.empty else ["zone_marker_gradients"],
        filter_note=f"protein ranking: {ranking_source}; abundance source: {quant_source}",
    )


def plot_recipe_dvp_zonation_module_curves(this_run_folder: str, *, data_BC, sampleinfo,
                                           protein_gene_map, input_paths, params):
    """Extension: periportal-urea vs central-xenobiotic module curves on the zone axis."""
    modules = [str(m) for m in _recipe_param(params, "modules", [])]
    zone_order = [str(z) for z in _recipe_param(params, "zone_order", [])]
    summary = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv")
    scores_json = _recipe_input_json(input_paths, "evaluation_evidence/curated_module_scores.json")
    if summary.empty and isinstance(scores_json.get("rows"), list):
        try:
            summary = pd.DataFrame(scores_json["rows"])
        except Exception:
            summary = pd.DataFrame()
    matrix = _recipe_module_group_matrix(summary, modules, zone_order)
    if matrix.empty or matrix.shape[1] < 2:
        return _placeholder_plot(this_run_folder, "recipe_dvp_zonation_module_curves.png",
                                 "No module zone means",
                                 "curated module tables did not cover the declared modules on a usable zone axis")
    file_path = _build_output_path(this_run_folder, "recipe_dvp_zonation_module_curves.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(matrix.shape[1])
    palette = sns.color_palette("colorblind", n_colors=len(matrix.index))
    for color, (module, row) in zip(palette, matrix.iterrows()):
        ax.plot(x, row.to_numpy(dtype=float), marker="o", linewidth=2.0, color=color, label=str(module))
    ax.axhline(0.0, color="#999999", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(c, 14) for c in matrix.columns], rotation=15, ha="right")
    ax.set_ylabel("Group mean module score")
    ax.set_title("Urea-cycle vs xenobiotic module direction along zones")
    _apply_nature_axes(ax)
    _legend_outside(ax, title="Module")
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        thresholds={"module_zone_means": matrix.round(4).to_dict(orient="index")},
        filter_note="curated module scores only; zone axis is the ordinal Portal–Midlobular–Central proxy",
    )


def plot_recipe_bloodcell_hspc_state_hierarchy_dotplot(this_run_folder: str, *, data_BC, sampleinfo,
                                                       protein_gene_map, input_paths, params):
    """BloodCell Fig. 2f,g equivalent: HSPC state-hierarchy marker dotplot."""
    state_order = [str(s) for s in _recipe_param(params, "state_order", [])]
    markers = [str(m) for m in _recipe_param(params, "markers", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, markers, sample_cols)
    if mat.empty:
        return _placeholder_plot(this_run_folder, "recipe_bloodcell_hspc_state_hierarchy_dotplot.png",
                                 "No markers matched",
                                 "None of the declared HSPC markers were found in the quantified matrix")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    states_present = [s for s in state_order if s in set(group_series)] or sorted(set(group_series))
    mean_df = _recipe_group_mean_matrix(mat, group_series, states_present)
    if mean_df.empty or mean_df.shape[1] < 2:
        return _placeholder_plot(this_run_folder, "recipe_bloodcell_hspc_state_hierarchy_dotplot.png",
                                 "State axis unavailable",
                                 "Fewer than two declared HSPC states were present in the sample sheet")
    z_df = _recipe_row_zscore(mean_df)
    diff_df = _recipe_differential_table(input_paths, ["processed_proteins/differential_GMP_vs_HSC.csv"])
    gene_lookup = _recipe_logfc_lookup(diff_df, list(mean_df.index))
    file_path = _build_output_path(this_run_folder, "recipe_bloodcell_hspc_state_hierarchy_dotplot.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(1.5 * mean_df.shape[1] + 3.0, 1.0 * len(mean_df.index) + 2.0))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = plt.Normalize(vmin=-2.5, vmax=2.5)
    for yi, gene in enumerate(mean_df.index):
        row_mean = mean_df.loc[gene].to_numpy(dtype=float)
        row_z = z_df.loc[gene].to_numpy(dtype=float)
        span = float(np.nanmax(row_mean) - np.nanmin(row_mean))
        for xi, state in enumerate(mean_df.columns):
            size = 90.0 if not span else 90.0 * (row_mean[xi] - float(np.nanmin(row_mean))) / span + 55.0
            ax.scatter(xi, yi, s=size, color=cmap(norm(float(row_z[xi]))), edgecolors="#333333",
                       linewidths=0.7, zorder=3)
    ax.set_xticks(range(len(mean_df.columns)))
    ax.set_xticklabels([_wrap_label(c, 12) for c in mean_df.columns])
    ax.set_yticks(range(len(mean_df.index)))
    ax.set_yticklabels(mean_df.index)
    ax.set_xlabel("HSPC state")
    ax.set_title("HSPC state-hierarchy marker dotplot")
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.1)
    ax.invert_yaxis()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Mean row Z-score", shrink=0.85)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(mean_df)),
        thresholds={"pval_cutoff": 0.05,
                    "GMP_vs_HSC_logFC_adjP": {g: [round(v["logFC"], 3), float(f"{v['adj_p']:.3g}")]
                                              for g, v in gene_lookup.items()}} if gene_lookup else None,
        filter_note="dot size: within-marker range of state mean abundance; color: row z-score; GMP_vs_HSC direction per AGENTS.md",
    )


def plot_recipe_bloodcell_cluster_label_agreement(this_run_folder: str, *, data_BC, sampleinfo,
                                                  protein_gene_map, input_paths, params):
    """Extension: protein-cluster × FACS-label agreement heatmap."""
    cross = _recipe_input_csv(input_paths, "evaluation_evidence/group_cross_table_Cluster_by_Label.csv")
    if cross.empty:
        sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
        if not sampleinfo_eff.empty and {"Cluster", "Label"}.issubset(sampleinfo_eff.columns):
            cross = pd.crosstab(sampleinfo_eff["Cluster"].astype(str), sampleinfo_eff["Label"].astype(str))
    if cross.empty:
        return _placeholder_plot(this_run_folder, "recipe_bloodcell_cluster_label_agreement.png",
                                 "No cluster×label table",
                                 "group_cross_table_Cluster_by_Label.csv was absent and the sample sheet has no Label column")
    if "Cluster" in cross.columns:
        cross = cross.set_index("Cluster")
    cross = cross.apply(pd.to_numeric, errors="coerce").fillna(0)
    if cross.empty:
        return _placeholder_plot(this_run_folder, "recipe_bloodcell_cluster_label_agreement.png",
                                 "Empty cross table", "The cross table contained no usable rows")
    file_path = _build_output_path(this_run_folder, "recipe_bloodcell_cluster_label_agreement.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(max(6.5, 1.05 * cross.shape[1] + 2.5), max(3.8, 0.55 * cross.shape[0] + 1.8)))
    totals = cross.sum(axis=1).replace(0, np.nan)
    share = cross.div(totals, axis=0)
    sns.heatmap(share, ax=ax, cmap="Blues", vmin=0, vmax=1, rasterized=True,
                annot=cross.astype(int), fmt="d",
                cbar_kws={"label": "Row share"}, annot_kws={"fontsize": 8.5},
                linewidths=0.6, linecolor="white")
    ax.set_title("Protein cluster × FACS label agreement")
    ax.set_xlabel("FACS label")
    ax.set_ylabel("Protein cluster")
    _apply_nature_heatmap_frame(ax)
    ax.set_xticklabels([_wrap_label(t.get_text(), 20) for t in ax.get_xticklabels()], rotation=30, ha="right")
    plt.tight_layout()
    dominant_share = share.max(axis=1).fillna(0.0)
    return _recipe_save_info(
        file_path, fig,
        thresholds={"dominant_label_share_by_cluster": {str(k): round(float(v), 3) for k, v in dominant_share.items()}},
        filter_note="cell annotation = cell count; color = within-cluster share of that label",
    )


def plot_recipe_turnover_drug_dose_footprint(this_run_folder: str, *, data_BC, sampleinfo,
                                             protein_gene_map, input_paths, params):
    """Turnover Fig. 2b,d,f equivalent: overall abundance + low/high dose log2FC scatters."""
    drugs = _recipe_param(params, "drugs", {}) or {}
    control_group = str(_recipe_param(params, "control_group", "Control"))
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    if sample_cols:
        group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
        per_sample = quant[sample_cols].apply(pd.to_numeric, errors="coerce")
        overall = per_sample.mean(axis=0, skipna=True)
        overall_df = pd.DataFrame({"group": [str(group_series.get(s, UNANNOTATED_GROUP_LABEL)) for s in sample_cols],
                                   "value": overall.to_numpy(dtype=float)})
    else:
        overall_df = pd.DataFrame()
    group_order = [control_group]
    for drug, doses in drugs.items():
        for dose in doses:
            group_order.append(f"{drug}_{dose}")
    diff_tables = ([t for t in input_paths if str(t).startswith("processed_proteins/differential_")]
                   if isinstance(input_paths, dict) else [])
    diff_df = _recipe_differential_table(input_paths, diff_tables)
    _set_visual_theme("notebook")
    file_path = _build_output_path(this_run_folder, "recipe_turnover_drug_dose_footprint.png")
    scatter_drugs = []
    drug_scatter: Dict[str, Dict[str, pd.Series]] = {}
    for drug, doses in drugs.items():
        doses = [str(d) for d in doses]
        if len(doses) < 2:
            continue
        low_name = f"differential_{drug}_{doses[0]}_vs_{control_group}.csv"
        high_name = f"differential_{drug}_{doses[1]}_vs_{control_group}.csv"
        low = diff_df[diff_df["contrast"].astype(str).str.contains(f"{drug}_{doses[0]}_vs_{control_group}", regex=False)]
        high = diff_df[diff_df["contrast"].astype(str).str.contains(f"{drug}_{doses[1]}_vs_{control_group}", regex=False)]
        if low.empty or high.empty:
            continue
        merged = low[["protein_id", "gene_label", "logFC"]].merge(
            high[["protein_id", "logFC"]], on="protein_id", suffixes=("_low", "_high")).dropna()
        if merged.empty:
            continue
        drug_scatter[drug] = {"low": merged.set_index("protein_id")["logFC_low"],
                              "high": merged.set_index("protein_id")["logFC_high"],
                              "doses": doses}
        scatter_drugs.append(drug)
    if overall_df.empty and not scatter_drugs:
        return _placeholder_plot(this_run_folder, "recipe_turnover_drug_dose_footprint.png",
                                 "No drug footprint evidence",
                                 "Neither per-sample abundance nor the declared dose-contrast differential tables were available")
    n_panels = 1 + len(scatter_drugs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 + 4.6 * len(scatter_drugs), 4.9), squeeze=False)
    column = 0
    if not overall_df.empty:
        ax = axes[0][0]
        groups_present = _recipe_ordered_present(group_order, overall_df["group"].unique())
        palette = _categorical_palette(groups_present)
        positions: List[float] = []
        labels: List[str] = []
        pos = 0.0
        ticks = []
        for group in groups_present:
            values = overall_df.loc[overall_df["group"] == group, "value"].to_numpy(dtype=float)
            start = pos
            for value in values:
                ax.bar(pos, value, width=0.75, color=palette[group], edgecolor="none")
                pos += 1.0
            ticks.append((start + pos - 1) / 2 if len(values) else pos)
            labels.append(_wrap_label(group, 14))
            pos += 0.7
        for group, tick in zip(groups_present, ticks):
            values = overall_df.loc[overall_df["group"] == group, "value"].to_numpy(dtype=float)
            if len(values):
                ax.hlines(float(np.median(values)), tick - 1.4, tick + 1.4, color="#222222",
                          linewidth=1.4, linestyle="--", zorder=4)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Overall abundance level (mean log2 intensity)")
        ax.set_title("Overall abundance per sample")
        _apply_nature_axes(ax)
        column += 1
    for drug in scatter_drugs:
        ax = axes[0][column]
        merged_x = drug_scatter[drug]["low"]
        merged_y = drug_scatter[drug]["high"]
        ax.scatter(merged_x, merged_y, s=14, color="#4C78A8", alpha=0.55, linewidths=0, rasterized=True)
        lims = [float(np.nanmin([merged_x.min(), merged_y.min()])), float(np.nanmax([merged_x.max(), merged_y.max()]))]
        ax.plot(lims, lims, color="#888888", linewidth=1.0, linestyle="--")
        ax.axhline(0, color="#BBBBBB", linewidth=0.8)
        ax.axvline(0, color="#BBBBBB", linewidth=0.8)
        ax.set_xlabel(f"log2FC {drug}_{drug_scatter[drug]['doses'][0]} vs {control_group}")
        ax.set_ylabel(f"log2FC {drug}_{drug_scatter[drug]['doses'][1]} vs {control_group}")
        ax.set_title(f"{drug} dose response")
        _apply_nature_axes(ax)
        column += 1
    fig.suptitle("Drug-dose abundance footprint (abundance only, not turnover rate)",
                 fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    correlations = {}
    for drug in scatter_drugs:
        x = drug_scatter[drug]["low"].to_numpy(dtype=float)
        y = drug_scatter[drug]["high"].to_numpy(dtype=float)
        if len(x) > 2 and float(np.std(x)) and float(np.std(y)):
            correlations[drug] = round(float(np.corrcoef(x, y)[0, 1]), 3)
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(max([len(v["low"]) for v in drug_scatter.values()] + [1])) if drug_scatter else None,
        thresholds={"dose_logFC_pearson": correlations} if correlations else None,
        filter_note=f"abundance source: {quant_source}; scatter diagonal = identity; abundance footprint only (no turnover-rate inference)",
    )


def plot_recipe_turnover_drug_module_direction(this_run_folder: str, *, data_BC, sampleinfo,
                                               protein_gene_map, input_paths, params):
    """Extension: Bortezomib vs Cycloheximide module/enrichment direction synthesis."""
    drug_modules = _recipe_param(params, "drug_modules", {}) or {}
    control_group = str(_recipe_param(params, "control_group", "Control"))
    summary = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv")
    drug_names = [str(d) for d in drug_modules]
    if not drug_names:
        drug_names = ["Bortezomib", "Cycloheximide"]
    # Dose ordering: Control first, then per drug Low before High when the
    # group labels carry the dose suffix.
    group_order = [control_group]
    available_groups = (set(summary["group"].astype(str)) if not summary.empty and "group" in summary.columns
                        else set())
    for drug in drug_names:
        drug_groups = sorted((g for g in available_groups
                              if g.startswith(drug + "_") or g.startswith(drug + "-")),
                             key=lambda g: (0 if "low" in g.lower() else 1, g))
        group_order += drug_groups
    module_matrix = _recipe_module_group_matrix(summary, None, group_order)
    go_tables = ([t for t in input_paths if str(t).startswith("enrichment_results/GO_")]
                 if isinstance(input_paths, dict) else [])
    enrich = _recipe_enrichment_table(input_paths, go_tables)
    if module_matrix.empty and enrich.empty:
        return _placeholder_plot(this_run_folder, "recipe_turnover_drug_module_direction.png",
                                 "No module/enrichment direction evidence",
                                 "curated module summary and GO direction tables were both unavailable")
    file_path = _build_output_path(this_run_folder, "recipe_turnover_drug_module_direction.png")
    keyword_map = {
        "Bortezomib": ["proteasom", "ubiquitin", "proteostas", "protein fold"],
        "Cycloheximide": ["ribosom", "translat", "peptid"],
    }
    rows: List[Dict[str, Any]] = []
    if not enrich.empty:
        for drug in drug_names:
            sub = enrich[enrich["Description"].notna()]
            if sub.empty:
                continue
            keywords = keyword_map.get(str(drug)) or [str(m).replace("_", " ") for m in drug_modules.get(str(drug), [])]
            keyword_hits = sub[sub["Description"].astype(str).str.lower().apply(
                lambda text: any(k in text for k in keywords))]
            ordered = pd.concat([keyword_hits, sub]).drop_duplicates(
                subset=["Description", "direction", "namespace"])
            for direction, sign in (("upregulated", 1.0), ("downregulated", -1.0)):
                part = ordered[ordered["direction"] == direction].sort_values("p.adjust").head(3)
                for _, row in part.iterrows():
                    rows.append({"drug": str(drug), "term": str(row["Description"]),
                                 "score": sign * float(-np.log10(max(float(row["p.adjust"]), 1e-300)))})
    _set_visual_theme("notebook")
    panels = 1 + (1 if rows else 0)
    fig_height = max(4.4, 0.30 * max(len(module_matrix), len(rows), 1) + 1.8)
    fig, axes = plt.subplots(1, panels, figsize=(7.2 + 5.6 * (panels - 1), fig_height), squeeze=False)
    ax = axes[0][0]
    if not module_matrix.empty:
        z_matrix = _recipe_row_zscore(module_matrix)
        sns.heatmap(z_matrix, ax=ax, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                    vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Module row Z-score"})
        ax.set_title("Curated module direction by treatment")
        ax.set_xlabel("Treatment group")
        ax.set_ylabel("")
        ax.set_yticklabels([_wrap_label(t.get_text(), 20) for t in ax.get_yticklabels()],
                           rotation=0, fontsize=8.5)
        _apply_nature_heatmap_frame(ax)
        _wrap_axis_ticklabels(ax, "x", width=14, rotation=30)
    else:
        ax.axis("off")
        ax.set_title("Curated module direction by treatment (unavailable)")
    if rows:
        ax2 = axes[0][1]
        term_df = pd.DataFrame(rows).sort_values("score")
        y = np.arange(len(term_df))
        colors = ["#B2182B" if v >= 0 else "#2166AC" for v in term_df["score"]]
        ax2.barh(y, term_df["score"].to_numpy(dtype=float), color=colors, edgecolor="none")
        ax2.set_yticks(y)
        ax2.set_yticklabels([f"{d[:4]}: {(t if len(t) <= 46 else t[:45] + '…')}"
                             for d, t in zip(term_df["drug"], term_df["term"])],
                            fontsize=6.8)
        ax2.axvline(0, color="#666666", linewidth=0.9)
        ax2.set_xlabel("Signed −log10(adjusted P)  (up: red, down: blue)")
        ax2.set_title("Drug-perturbation term directions")
        _apply_nature_axes(ax2)
    fig.suptitle("Bortezomib vs Cycloheximide mechanism direction", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    omitted = []
    if module_matrix.empty:
        omitted.append("module_direction_heatmap")
    if not rows:
        omitted.append("drug_specific_terms")
    return _recipe_save_info(
        file_path, fig,
        omitted_panels=omitted,
        thresholds={"module_group_means": module_matrix.round(4).to_dict(orient="index")} if not module_matrix.empty else None,
        filter_note="module Z-scores are row-normalized; term scores are ORA-based signed significance, not GSEA NES",
    )


def _recipe_tier_label(name: str) -> str:
    text = str(name or "").lower()
    if "single_cell" in text or "single cell" in text or "single-cell" in text:
        return "single_cell"
    if "10cell" in text or "pool" in text:
        return "10cell_pool"
    return "other"


def plot_recipe_turnover_sc_lowinput_stratification(this_run_folder: str, *, data_BC, sampleinfo,
                                                    protein_gene_map, input_paths, params):
    """Extension: single-cell vs low-input stratification on the treatment PCA."""
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    tier_col = str(_recipe_param(params, "tier_col", "Label"))
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    if not sample_cols or sampleinfo_eff.empty:
        return _placeholder_plot(this_run_folder, "recipe_turnover_sc_lowinput_stratification.png",
                                 "No sample mapping",
                                 "SampleInfo and the quantified matrix could not be aligned on FileName")
    coords = _recipe_pca_coords(quant, sample_cols)
    if coords.empty:
        return _placeholder_plot(this_run_folder, "recipe_turnover_sc_lowinput_stratification.png",
                                 "PCA unavailable", "Too few usable proteins/samples for a PCA panel")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    dedup = sampleinfo_eff.drop_duplicates(subset=["FileName"]).set_index("FileName")
    tiers = []
    for sample in coords.index:
        raw = ""
        if tier_col in dedup.columns and sample in dedup.index:
            raw = dedup.loc[sample, tier_col]
        if not str(raw).strip() or str(raw).lower() == "nan":
            raw = sample
        tiers.append(_recipe_tier_label(raw))
    tier_series = pd.Series(tiers, index=coords.index)
    treatment = pd.Series([str(group_series.get(s, UNANNOTATED_GROUP_LABEL)) for s in coords.index], index=coords.index)
    file_path = _build_output_path(this_run_folder, "recipe_turnover_sc_lowinput_stratification.png")
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    palette = _categorical_palette(treatment.unique())
    tier_markers = {"single_cell": "o", "10cell_pool": "s", "other": "^"}
    for group in palette:
        for tier, marker in tier_markers.items():
            mask = (treatment == group) & (tier_series == tier)
            if not mask.any():
                continue
            ax.scatter(coords.loc[mask, "PC1"], coords.loc[mask, "PC2"], s=34, marker=marker,
                       color=palette[group], alpha=0.85, linewidths=0.6, edgecolors="white")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", color="#8C8C8C",
                      label="tier: single_cell (circle)"),
               Line2D([0], [0], marker="s", linestyle="", color="#8C8C8C",
                      label="tier: 10cell_pool (square)")]
    for group in palette:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=palette[group], label=str(group)))
    ax.legend(handles=handles, title="Treatment / tier", bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, frameon=True, fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA: treatment color, input-tier marker")
    _apply_nature_axes(ax)
    ax2 = axes[1]
    counts = pd.crosstab(treatment, tier_series)
    tier_order = [t for t in ["single_cell", "10cell_pool", "other"] if t in counts.columns]
    counts = counts[tier_order] if tier_order else counts
    bottom = np.zeros(len(counts.index))
    y_positions = np.arange(len(counts.index))
    for tier in counts.columns:
        values = counts[tier].to_numpy(dtype=float)
        ax2.barh(y_positions, values, left=bottom, color={"single_cell": "#4C78A8", "10cell_pool": "#F58518"}.get(tier, "#BAB0AC"),
                 edgecolor="white", linewidth=0.6, label=str(tier))
        for yi, (value, base) in enumerate(zip(values, bottom)):
            if value > 0:
                ax2.text(base + value / 2, yi, f"{int(value)}", ha="center", va="center", fontsize=8)
        bottom += values
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([_wrap_label(g, 18) for g in counts.index])
    ax2.set_xlabel("Samples")
    ax2.set_title("Tier composition per treatment")
    _apply_nature_axes(ax2)
    ax2.invert_yaxis()
    fig.suptitle("Single-cell vs low-input stratification", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    cross = _recipe_input_csv(input_paths, "evaluation_evidence/group_cross_table_Cluster_by_Label.csv")
    return _recipe_save_info(
        file_path, fig,
        thresholds={"tier_counts": counts.to_dict(orient="index")},
        filter_note=f"abundance source: {quant_source}; tier parsed from {tier_col}/FileName text; declared cross table rows: {len(cross)}",
    )


def plot_recipe_pscope_polarization_gradient(this_run_folder: str, *, data_BC, sampleinfo,
                                             protein_gene_map, input_paths, params):
    """pSCoPE Fig. 4a equivalent: polarization gradient on a signature-colored PCA."""
    signature_genes = [str(g) for g in _recipe_param(params, "signature_genes", [])]
    polarization_order = [str(c) for c in _recipe_param(params, "polarization_order", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, signature_genes, sample_cols)
    if mat.empty:
        return _placeholder_plot(this_run_folder, "recipe_pscope_polarization_gradient.png",
                                 "No signature genes matched",
                                 "None of the declared V-ATPase/lysosome signature genes were found in the quantified matrix")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    conditions_present = _recipe_ordered_present(polarization_order, list(group_series.unique()))
    per_gene_z = _recipe_row_zscore(mat)  # z per protein across samples
    signature_z = per_gene_z.mean(axis=0, skipna=True)
    coords = _recipe_pca_coords(quant, sample_cols)
    if coords is not None and not coords.empty:
        aligned = signature_z.reindex(coords.index).dropna()
    else:
        aligned = signature_z.dropna()
    if aligned.empty:
        return _placeholder_plot(this_run_folder, "recipe_pscope_polarization_gradient.png",
                                 "Signature projection unavailable", "Signature z-scores could not be aligned to samples")
    file_path = _build_output_path(this_run_folder, "recipe_pscope_polarization_gradient.png")
    _set_visual_theme("notebook")
    show_pca = coords is not None and not coords.empty
    fig, axes = plt.subplots(1, 2 if show_pca else 1,
                             figsize=(12.8 if show_pca else 6.6, 5.0), squeeze=False)
    if show_pca:
        ax = axes[0][0]
        vmin, vmax = float(np.min(aligned)), float(np.max(aligned))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        sc = ax.scatter(coords.loc[aligned.index, "PC1"], coords.loc[aligned.index, "PC2"],
                        c=aligned.to_numpy(dtype=float), cmap="viridis", s=34, linewidths=0.5,
                        edgecolors="white")
        markers = ["o", "s", "^", "D"]
        for ci, condition in enumerate(conditions_present):
            samples = [s for s in aligned.index if group_series.get(s) == condition]
            if not samples:
                continue
            ax.scatter(coords.loc[samples, "PC1"], coords.loc[samples, "PC2"], s=110, marker=markers[ci % len(markers)],
                       facecolors="none", edgecolors="#222222", linewidths=1.0)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker=markers[ci % len(markers)], linestyle="", markersize=9,
                          markerfacecolor="none", markeredgecolor="#222222", label=str(condition))
                   for ci, condition in enumerate(conditions_present)]
        ax.legend(handles=handles, title="Condition", loc="upper right", frameon=True, fontsize=9)
        fig.colorbar(sc, ax=ax, label="Signature median Z-score", shrink=0.85)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA colored by polarization signature")
        _apply_nature_axes(ax)
    ax2 = axes[0][1] if show_pca else axes[0][0]
    data_by_condition = []
    for condition in conditions_present:
        samples = [s for s in aligned.index if group_series.get(s) == condition]
        data_by_condition.append(aligned.loc[samples].to_numpy(dtype=float) if samples else np.array([]))
    boxes = ax2.boxplot(data_by_condition, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="#222222", linewidth=1.2))
    palette = _categorical_palette(conditions_present)
    for patch, condition in zip(boxes["boxes"], conditions_present):
        patch.set_facecolor(palette[condition])
        patch.set_alpha(0.5)
        patch.set_edgecolor("#444444")
    rng = np.random.default_rng(43)
    for gi, values in enumerate(data_by_condition):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax2.scatter(np.full(len(values), gi + 1) + jitter, values, s=10, color="#333333",
                    alpha=0.55, linewidths=0, rasterized=True)
    ax2.set_xticks(range(1, len(conditions_present) + 1))
    ax2.set_xticklabels([_wrap_label(c, 14) for c in conditions_present], rotation=15, ha="right")
    ax2.set_ylabel("Signature median Z-score")
    ax2.set_title("Polarization gradient by condition")
    _apply_nature_axes(ax2)
    fig.suptitle("Untreated → LPS_low → LPS_high polarization gradient", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    condition_means = {c: (round(float(np.mean(v)), 4) if len(v) else None)
                       for c, v in zip(conditions_present, data_by_condition)}
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(aligned)),
        thresholds={"condition_mean_signature_z": condition_means, "pval_cutoff": 0.05},
        filter_note=f"abundance source: {quant_source}; signature = mean z of matched V-ATPase/lysosome genes across samples",
    )


def plot_recipe_pscope_vatpase_covariation(this_run_folder: str, *, data_BC, sampleinfo,
                                           protein_gene_map, input_paths, params):
    """Extension: V-ATPase × phagosome/lysosome protein covariation per condition."""
    gene_sets = _recipe_param(params, "gene_sets", {}) or {}
    condition_order = [str(c) for c in _recipe_param(params, "condition_order", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    set_names = [str(k) for k in gene_sets]
    if len(set_names) < 2:
        return _placeholder_plot(this_run_folder, "recipe_pscope_vatpase_covariation.png",
                                 "Gene sets incomplete", "The recipe requires two declared protein sets for covariation")
    matrices = {}
    for name in set_names[:2]:
        mat = _recipe_gene_matrix(quant, gene_index, gene_sets[name], sample_cols)
        if mat.empty:
            return _placeholder_plot(this_run_folder, "recipe_pscope_vatpase_covariation.png",
                                     "Gene set unmatched", f"Set '{name}' had no matches in the quantified matrix")
        matrices[name] = _recipe_row_zscore(mat.T).T
    conditions = _recipe_ordered_present(condition_order, list(group_series.unique()))
    _set_visual_theme("notebook")
    ncols = min(3, max(len(conditions), 1))
    nrows = int(np.ceil(max(len(conditions), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows + 0.5), squeeze=False)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    max_r: Dict[str, float] = {}
    n_pairs: Dict[str, int] = {}
    rendered = 0
    for idx, condition in enumerate(conditions):
        ax = axes[idx // ncols][idx % ncols]
        samples = [s for s in sample_cols if group_series.get(s) == condition]
        set_a, set_b = matrices[set_names[0]], matrices[set_names[1]]
        if len(samples) >= 3:
            a = set_a[samples].to_numpy(dtype=float)
            b = set_b[samples].to_numpy(dtype=float)
            corr = np.corrcoef(a, b)[:a.shape[0], a.shape[0]:]
            corr_df = pd.DataFrame(corr, index=set_a.index, columns=set_b.index)
            corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            sns.heatmap(corr_df, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1, rasterized=True,
                        annot=corr_df.round(2).astype(str), fmt="", annot_kws={"fontsize": 7},
                        cbar_kws={"label": "Pearson r"}, linewidths=0.5, linecolor="white")
            max_r[condition] = round(float(np.nanmax(np.abs(corr_df.to_numpy(dtype=float)))), 3)
            n_pairs[condition] = int(corr_df.size)
            rendered += 1
        else:
            ax.axis("off")
        ax.set_title(f"{condition} (n={len(samples)})", fontsize=10)
        ax.set_xlabel(_wrap_label(set_names[1], 22))
        ax.set_ylabel(_wrap_label(set_names[0], 22))
        _apply_nature_heatmap_frame(ax)
        _wrap_axis_ticklabels(ax, "x", width=10, rotation=90)
        _wrap_axis_ticklabels(ax, "y", width=12)
    for leftover in range(len(conditions), nrows * ncols):
        axes[leftover // ncols][leftover % ncols].axis("off")
    fig.suptitle("V-ATPase × phagosome/lysosome covariation by condition", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    file_path = _build_output_path(this_run_folder, "recipe_pscope_vatpase_covariation.png")
    if not rendered:
        return _placeholder_plot(this_run_folder, "recipe_pscope_vatpase_covariation.png",
                                 "Conditions too small", "No condition had at least 3 samples for covariation")
    return _recipe_save_info(
        file_path, fig,
        thresholds={"max_abs_r_by_condition": max_r, "n_pairs_by_condition": n_pairs},
        filter_note=f"abundance source: {quant_source}; per-protein z within condition; Pearson r across samples",
    )


def plot_recipe_proteinleakage_compartment_leakage(this_run_folder: str, *, data_BC, sampleinfo,
                                                   protein_gene_map, input_paths, params):
    """ProteinLeakage Fig. 1e equivalent: compartment-selective leakage log2FC boxplot."""
    compartments = _recipe_param(params, "compartments", {}) or {}
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    diff_df = _recipe_differential_table(input_paths, ["processed_proteins/differential_Intact_vs_Permeable.csv"])
    gene_lookup = _recipe_logfc_lookup(diff_df, [g for genes in compartments.values() for g in genes])
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    fallback_logfc: Dict[str, float] = {}
    if sample_cols and not sampleinfo_eff.empty and "Type1" in sampleinfo_eff.columns:
        group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Type1")
        intact_cols = [s for s in sample_cols if group_series.get(s) == "Intact"]
        permeable_cols = [s for s in sample_cols if group_series.get(s) == "Permeable"]
        if intact_cols and permeable_cols:
            for genes in compartments.values():
                mat = _recipe_gene_matrix(quant, gene_index, genes, sample_cols)
                if mat.empty:
                    continue
                intact_mean = mat[intact_cols].mean(axis=1, skipna=True)
                permeable_mean = mat[permeable_cols].mean(axis=1, skipna=True)
                for gene, value in (intact_mean - permeable_mean).items():
                    fallback_logfc[str(gene).upper()] = float(value)
    rows = []
    for label, genes in compartments.items():
        for gene in genes:
            key = str(gene).strip().upper()
            if key in gene_lookup:
                rows.append({"compartment": str(label), "gene": str(gene),
                             "logFC": gene_lookup[key]["logFC"], "source": "limma"})
            elif key in fallback_logfc:
                rows.append({"compartment": str(label), "gene": str(gene),
                             "logFC": fallback_logfc[key], "source": "group_mean_diff"})
    if not rows:
        return _placeholder_plot(this_run_folder, "recipe_proteinleakage_compartment_leakage.png",
                                 "No compartment logFC values",
                                 "Compartment markers were neither in the differential table nor computable from group means")
    logfc_df = pd.DataFrame(rows)
    file_path = _build_output_path(this_run_folder, "recipe_proteinleakage_compartment_leakage.png")
    _set_visual_theme("notebook")
    module_summary = _recipe_input_csv(input_paths, "proteins_leakage/leakage_module_summary.csv")
    has_modules = (not module_summary.empty and {"module", "mean_logFC_group_a_minus_group_b"}.issubset(module_summary.columns))
    fig, axes = plt.subplots(1, 2 if has_modules else 1,
                             figsize=(11.8 if has_modules else 6.4, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 0.9]} if has_modules else None,
                             squeeze=False)
    ax = axes[0][0]
    compartments_present = list(dict.fromkeys(logfc_df["compartment"].tolist()))
    data_by_compartment = [logfc_df.loc[logfc_df["compartment"] == c, "logFC"].to_numpy(dtype=float)
                           for c in compartments_present]
    boxes = ax.boxplot(data_by_compartment, patch_artist=True, widths=0.5,
                       medianprops=dict(color="#222222", linewidth=1.3))
    palette = _categorical_palette(compartments_present)
    for patch, name in zip(boxes["boxes"], compartments_present):
        patch.set_facecolor(palette[name])
        patch.set_alpha(0.5)
        patch.set_edgecolor("#444444")
    rng = np.random.default_rng(5)
    for ci, values in enumerate(data_by_compartment):
        jitter = rng.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(np.full(len(values), ci + 1) + jitter, values, s=30, color="#333333",
                   alpha=0.75, linewidths=0.6, zorder=3)
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(1, len(compartments_present) + 1))
    ax.set_xticklabels([_wrap_label(c, 22) for c in compartments_present])
    ax.set_ylabel("log2FC (Intact − Permeable)")
    ax.set_title("Compartment-selective leakage")
    _apply_nature_axes(ax)
    if has_modules:
        ax2 = axes[0][1]
        mod = module_summary.dropna(subset=["mean_logFC_group_a_minus_group_b"]).copy()
        mod["value"] = pd.to_numeric(mod["mean_logFC_group_a_minus_group_b"], errors="coerce")
        mod = mod.dropna(subset=["value"]).sort_values("value")
        y = np.arange(len(mod))
        colors = ["#B2182B" if v >= 0 else "#2166AC" for v in mod["value"]]
        ax2.barh(y, mod["value"].to_numpy(dtype=float), color=colors, edgecolor="none")
        ax2.set_yticks(y)
        ax2.set_yticklabels([_wrap_label(str(m), 30) for m in mod["module"]], fontsize=8)
        ax2.axvline(0, color="#666666", linewidth=0.9)
        ax2.set_xlabel("Module mean log2FC (Intact − Permeable)")
        ax2.set_title("Curated leakage modules")
        _apply_nature_axes(ax2)
    fig.suptitle("Cytosol/nucleus depletion vs mito/membrane retention", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    medians = {c: round(float(np.median(v)), 4) for c, v in zip(compartments_present, data_by_compartment) if len(v)}
    n_fallback = int((logfc_df["source"] == "group_mean_diff").sum())
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(logfc_df)),
        thresholds={"compartment_median_logFC": medians},
        filter_note=f"abundance source: {quant_source}; log2FC>0 = higher in Intact; {n_fallback} value(s) from group-mean differences when limma rows were unavailable",
    )


def plot_recipe_proteinleakage_status_confounder_dotplot(this_run_folder: str, *, data_BC, sampleinfo,
                                                         protein_gene_map, input_paths, params):
    """Extension: permeable fraction by cell type × preservation state."""
    cluster_col = str(_recipe_param(params, "cluster_col", "Cluster"))
    type1_col = str(_recipe_param(params, "type1_col", "Type1"))
    type2_col = str(_recipe_param(params, "type2_col", "Type2"))
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    if sampleinfo_eff.empty or not {cluster_col, type1_col}.issubset(sampleinfo_eff.columns):
        return _placeholder_plot(this_run_folder, "recipe_proteinleakage_status_confounder_dotplot.png",
                                 "No status sample sheet",
                                 "SampleInfo with cell-type and leakage-status columns was unavailable")
    df = sampleinfo_eff.copy()
    df["cell_type"] = _normalize_group_labels(df[cluster_col]).astype(str)
    df["status"] = _normalize_group_labels(df[type1_col]).astype(str)
    permeable_label = next((c for c in df["status"].unique() if "permeable" in c.lower()), "Permeable")
    rows = []
    has_type2 = type2_col in df.columns
    if has_type2:
        df["state"] = _normalize_group_labels(df[type2_col]).astype(str)
        for (cell_type, state), sub in df.groupby(["cell_type", "state"]):
            rows.append({"cell_type": cell_type, "state": str(state), "n": int(len(sub)),
                         "fraction": float((sub["status"] == permeable_label).mean())})
    else:
        for cell_type, sub in df.groupby("cell_type"):
            rows.append({"cell_type": cell_type, "state": "all", "n": int(len(sub)),
                         "fraction": float((sub["status"] == permeable_label).mean())})
    frac_df = pd.DataFrame(rows)
    if frac_df.empty:
        return _placeholder_plot(this_run_folder, "recipe_proteinleakage_status_confounder_dotplot.png",
                                 "No groups available", "The sample sheet had no usable cell-type/status groups")
    cross = _recipe_input_csv(input_paths, "evaluation_evidence/group_cross_table_Type1_by_Type2.csv")
    file_path = _build_output_path(this_run_folder, "recipe_proteinleakage_status_confounder_dotplot.png")
    _set_visual_theme("notebook")
    cell_order = _recipe_ordered_present(None, frac_df["cell_type"].unique())
    state_palette = _categorical_palette(frac_df["state"].unique())
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(cell_order) + 2.5), 4.8))
    for si, state in enumerate(sorted(frac_df["state"].unique())):
        sub = frac_df[frac_df["state"] == state]
        x_positions = [cell_order.index(c) for c in sub["cell_type"]]
        ax.scatter(x_positions, sub["fraction"], s=40 + 2.2 * sub["n"].to_numpy(dtype=float),
                   color=state_palette[state], marker=markers[si % len(markers)],
                   edgecolors="#333333", linewidths=0.8, alpha=0.9,
                   label=f"{state} (n={int(sub['n'].sum())})")
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels([_wrap_label(c, 16) for c in cell_order], rotation=25, ha="right")
    ax.set_ylabel(f"Fraction {permeable_label}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Permeability by cell type and preservation state")
    ax.axhline(df["status"].eq(permeable_label).mean(), color="#888888", linewidth=1.0, linestyle="--")
    _apply_nature_axes(ax)
    from matplotlib.lines import Line2D
    states_sorted = sorted(frac_df["state"].unique())
    marker_list = ["o", "s", "^", "D", "v"]
    handles = [Line2D([0], [0], marker=marker_list[i % len(marker_list)], linestyle="", markersize=9,
                      color=state_palette[state],
                      label=f"{state} (n={int(frac_df.loc[frac_df['state'] == state, 'n'].sum())})")
               for i, state in enumerate(states_sorted)]
    ax.legend(handles=handles, title="Preservation", bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, frameon=True, fontsize=9)
    plt.tight_layout()
    fractions = {f"{r.cell_type}|{r.state}": round(float(r.fraction), 3) for r in frac_df.itertuples()}
    return _recipe_save_info(
        file_path, fig,
        thresholds={"permeable_fraction": fractions, "n_samples": int(len(df))},
        filter_note=(f"dot size ∝ group n; dashed line = overall {permeable_label} fraction; "
                     f"declared Type1×Type2 cross table rows: {len(cross)}"),
    )


def plot_recipe_proteinleakage_leakage_correlation_heatmap(this_run_folder: str, *, data_BC, sampleinfo,
                                                           protein_gene_map, input_paths, params):
    """Extension: cross-cell-type leakage fold-change correlation heatmap."""
    summary = _recipe_input_json(input_paths, "proteins_leakage/leakage_summary.json")

    def _find_correlation(node: Any) -> Any:
        if isinstance(node, dict):
            candidate = node.get("correlation_matrix")
            if isinstance(candidate, dict) and candidate:
                return candidate
            for value in node.values():
                found = _find_correlation(value)
                if found:
                    return found
        return None

    matrix_data = _find_correlation(summary) if isinstance(summary, dict) else None
    if not isinstance(matrix_data, dict) or not matrix_data:
        return _placeholder_plot(this_run_folder, "recipe_proteinleakage_leakage_correlation_heatmap.png",
                                 "No correlation matrix",
                                 "proteins_leakage/leakage_summary.json did not contain a correlation_matrix")
    try:
        corr = pd.DataFrame(matrix_data).astype(float)
    except Exception:
        return _placeholder_plot(this_run_folder, "recipe_proteinleakage_leakage_correlation_heatmap.png",
                                 "Unreadable correlation matrix", "The correlation_matrix payload could not be coerced to a matrix")
    file_path = _build_output_path(this_run_folder, "recipe_proteinleakage_leakage_correlation_heatmap.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(max(5.6, 0.75 * corr.shape[1] + 2.2), max(4.6, 0.62 * corr.shape[0] + 1.8)))
    sns.heatmap(corr, ax=ax, cmap="Blues", vmin=0, vmax=1, rasterized=True,
                annot=corr.round(2).astype(str), fmt="", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Pearson r of leakage log2FC"}, linewidths=0.6, linecolor="white")
    ax.set_title("Cross-cell-type leakage correlation")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Cell type")
    _apply_nature_heatmap_frame(ax)
    ax.set_xticklabels([_wrap_label(t.get_text(), 14) for t in ax.get_xticklabels()], rotation=30, ha="right")
    plt.tight_layout()
    values = corr.to_numpy(dtype=float)
    off_diag = values[~np.eye(values.shape[0], dtype=bool)]
    high_pairs = summary.get("high_correlation_pairs") if isinstance(summary, dict) else None
    return _recipe_save_info(
        file_path, fig,
        thresholds={"max_off_diagonal_r": round(float(np.max(off_diag)), 3) if off_diag.size else None,
                    "high_correlation_pairs": high_pairs if isinstance(high_pairs, list) else None},
        filter_note="matrix reused verbatim from the run's leakage analysis (stain-free QC evidence)",
    )


def plot_recipe_scpro_klrg1_context_matrix(this_run_folder: str, *, data_BC, sampleinfo,
                                           protein_gene_map, input_paths, params):
    """SCPro Fig. 6f equivalent: KLRG1+ vs KLRG1− direction matrix across T-cell contexts."""
    contexts = _recipe_param(params, "contexts", []) or []
    focus_genes = [str(g) for g in _recipe_param(params, "focus_genes", [])]
    pval_cut = float(_recipe_param(params, "pval_cutoff", 0.05))
    columns: Dict[str, pd.DataFrame] = {}
    for context in contexts:
        label = str(context.get("label", ""))
        contrast = str(context.get("contrast", ""))
        table = f"processed_proteins/differential_{contrast}.csv"
        diff_df = _recipe_differential_table(input_paths, [table])
        if diff_df.empty:
            continue
        columns[label or contrast] = diff_df
    if not columns or not focus_genes:
        return _placeholder_plot(this_run_folder, "recipe_scpro_klrg1_context_matrix.png",
                                 "No context differential tables",
                                 "None of the declared KLRG1 contrast tables were available")
    rows = []
    for context_label, diff_df in columns.items():
        lookup = _recipe_logfc_lookup(diff_df, focus_genes)
        for gene in focus_genes:
            info = lookup.get(str(gene).upper())
            if info is None:
                continue
            rows.append({"gene": str(gene), "context": context_label, "logFC": info["logFC"],
                         "adj_p": info["adj_p"]})
    if not rows:
        return _placeholder_plot(this_run_folder, "recipe_scpro_klrg1_context_matrix.png",
                                 "No focus genes matched",
                                 "None of the declared focus genes were found in the context differential tables")
    long_df = pd.DataFrame(rows)
    matrix = long_df.pivot_table(index="gene", columns="context", values="logFC", aggfunc="mean")
    sig = long_df.pivot_table(index="gene", columns="context", values="adj_p", aggfunc="min") < pval_cut
    context_order = [str(c.get("label", "")) for c in contexts if str(c.get("label", "")) in matrix.columns]
    context_order += [c for c in matrix.columns if c not in context_order]
    matrix = matrix[context_order]
    sig = sig.reindex(index=matrix.index, columns=matrix.columns).fillna(False)
    file_path = _build_output_path(this_run_folder, "recipe_scpro_klrg1_context_matrix.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(max(5.6, 1.5 * matrix.shape[1] + 2.2), max(3.8, 0.55 * matrix.shape[0] + 1.8)))
    _recipe_direction_heatmap(ax, matrix, cbar_label="log2FC (KLRG1+ − KLRG1−)", sig_mask=sig)
    ax.set_title("KLRG1 program direction across T-cell contexts")
    ax.set_xlabel("Cell context")
    ax.set_ylabel("")
    _wrap_axis_ticklabels(ax, "x", width=14, rotation=0)
    plt.tight_layout()
    shared = int((sig.sum(axis=1) == matrix.shape[1]).sum())
    signs = np.sign(matrix.to_numpy(dtype=float))
    divergent = int((signs != signs[:, [0]]).any(axis=1).sum()) if matrix.shape[1] else 0
    n_sig = {c: int(sig[c].sum()) for c in matrix.columns}
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(matrix.shape[0]),
        n_sig=n_sig,
        thresholds={"pval_cutoff": pval_cut,
                    "shared_significant_genes": shared,
                    "direction_divergent_genes_vs_first_context": divergent},
        filter_note="ring = FDR below cutoff in that context; avoid generalizing one context's KLRG1 program to all T cells",
    )


def plot_recipe_scpro_treg_klrg1_enrichment(this_run_folder: str, *, data_BC, sampleinfo,
                                            protein_gene_map, input_paths, params):
    """Extension: Treg KLRG1+ vs KLRG1− GO/KEGG/Reactome enrichment bar."""
    contrast = str(_recipe_param(params, "contrast", ""))
    if not contrast:
        return _placeholder_plot(this_run_folder, "recipe_scpro_treg_klrg1_enrichment.png",
                                 "No contrast declared", "The recipe params carry no contrast name")
    tables = ([t for t in input_paths
               if str(t).startswith("enrichment_results/") and contrast in str(t)]
              if isinstance(input_paths, dict) else [])
    enrich = _recipe_enrichment_table(input_paths, tables)
    if enrich.empty:
        return _placeholder_plot(this_run_folder, "recipe_scpro_treg_klrg1_enrichment.png",
                                 "No enrichment tables", f"No GO/KEGG/Reactome tables matched contrast {contrast}")
    top_n = int(_recipe_param(params, "top_n_terms", 8))
    enrich = enrich.sort_values("p.adjust").groupby(["namespace", "direction"], group_keys=False).head(top_n)
    enrich = enrich.assign(score=enrich.apply(
        lambda row: (-np.log10(max(float(row["p.adjust"]), 1e-300)))
        * (-1.0 if str(row["direction"]) == "downregulated" else 1.0), axis=1))
    enrich = enrich.sort_values("score")
    file_path = _build_output_path(this_run_folder, "recipe_scpro_treg_klrg1_enrichment.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.36 * len(enrich) + 1.6)))
    y = np.arange(len(enrich))
    colors = ["#B2182B" if v >= 0 else "#2166AC" for v in enrich["score"]]
    ax.barh(y, enrich["score"].to_numpy(dtype=float), color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{str(n)}] {_wrap_label(str(d), 40)}" for n, d in zip(enrich["namespace"], enrich["Description"])],
                       fontsize=8)
    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("Signed −log10(adjusted P)  (up: red, down: blue)")
    ax.set_title(f"Treg KLRG1+ vs KLRG1− enrichment signature")
    _apply_nature_axes(ax)
    plt.tight_layout()
    n_up = int((enrich["direction"] == "upregulated").sum())
    n_down = int((enrich["direction"] == "downregulated").sum())
    return _recipe_save_info(
        file_path, fig,
        n_up=n_up,
        n_down=n_down,
        thresholds={"n_terms": int(len(enrich)),
                    "top_term": str(enrich.iloc[-1]["Description"]) if len(enrich) else None},
        filter_note="ORA-based signed significance for the highest-weight core contrast (functional readout added on top of limma evidence)",
    )


def plot_recipe_scpro_treg_effector_group_means(this_run_folder: str, *, data_BC, sampleinfo,
                                                protein_gene_map, input_paths, params):
    """Extension: Treg KLRG1+ immunosuppression program group means + detection."""
    genes = [str(g) for g in _recipe_param(params, "genes", [])]
    module = str(_recipe_param(params, "module", ""))
    group_order = [str(g) for g in _recipe_param(params, "group_order", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, genes, sample_cols)
    summary = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv")
    module_matrix = _recipe_module_group_matrix(summary, [module] if module else None, group_order)
    if mat.empty and module_matrix.empty:
        return _placeholder_plot(this_run_folder, "recipe_scpro_treg_effector_group_means.png",
                                 "No gene/module evidence",
                                 "Declared genes matched nothing in the matrix and the curated module summary was unavailable")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    mean_df = pd.DataFrame()
    detect_df = pd.DataFrame()
    if not mat.empty:
        groups_present = _recipe_ordered_present(group_order, list(group_series.unique()))
        mean_df = _recipe_group_mean_matrix(mat, group_series, groups_present)
        detect_df = mean_df.copy()
        for group in mean_df.columns:
            samples = [s for s in sample_cols if group_series.get(s) == group]
            block = mat[samples].apply(pd.to_numeric, errors="coerce") if samples else pd.DataFrame()
            detect_df[group] = (block.replace(0, np.nan).notna().mean(axis=1)).reindex(mean_df.index) if not block.empty else np.nan
    file_path = _build_output_path(this_run_folder, "recipe_scpro_treg_effector_group_means.png")
    _set_visual_theme("notebook")
    panels = []
    if not mean_df.empty:
        panels.append("gene_group_means")
    if not module_matrix.empty:
        panels.append("module_group_means")
    if not panels:
        return _placeholder_plot(this_run_folder, "recipe_scpro_treg_effector_group_means.png",
                                 "No group means", "Grouped means could not be computed for the declared genes or module")
    fig, axes = plt.subplots(1, len(panels) + (1 if not mean_df.empty else 0),
                             figsize=(6.4 * len(panels) + (3.4 if not mean_df.empty else 0), max(4.0, 0.5 * max(len(mean_df), 1) + 1.8)),
                             squeeze=False)
    column = 0
    if not mean_df.empty:
        z_df = _recipe_row_zscore(mean_df)
        sns.heatmap(z_df, ax=axes[0][column], cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                    vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Group mean row Z-score"})
        axes[0][column].set_title("Candidate program group means")
        axes[0][column].set_xlabel("Group")
        axes[0][column].set_ylabel("")
        _apply_nature_heatmap_frame(axes[0][column])
        _wrap_axis_ticklabels(axes[0][column], "x", width=18, rotation=25)
        column += 1
        sns.heatmap(detect_df * 100, ax=axes[0][column], cmap="YlGnBu", vmin=0, vmax=100,
                    rasterized=True, cbar_kws={"label": "Detection rate (%)"})
        axes[0][column].set_title("Detection rate")
        axes[0][column].set_xlabel("Group")
        axes[0][column].set_ylabel("")
        axes[0][column].set_yticklabels([])
        _apply_nature_heatmap_frame(axes[0][column])
        _wrap_axis_ticklabels(axes[0][column], "x", width=18, rotation=25)
        column += 1
    if not module_matrix.empty:
        ax2 = axes[0][column] if column < axes[0].shape[0] else None
        if ax2 is None:
            ax2 = fig.add_subplot(1, column + 1, column + 1)
        x = np.arange(module_matrix.shape[1])
        ax2.bar(x, module_matrix.iloc[0].to_numpy(dtype=float), color="#4C78A8", width=0.6)
        ax2.axhline(0, color="#666666", linewidth=0.9)
        ax2.set_xticks(x)
        ax2.set_xticklabels([_wrap_label(c, 16) for c in module_matrix.columns], rotation=25, ha="right")
        ax2.set_ylabel("Group mean module score")
        ax2.set_title(_wrap_label(module, 30))
        _apply_nature_axes(ax2)
    fig.suptitle("Treg KLRG1+ immunosuppressive/effector program", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(mean_df)) if not mean_df.empty else None,
        thresholds={"module_group_means": module_matrix.round(4).to_dict(orient="index")} if not module_matrix.empty else None,
        filter_note=f"abundance source: {quant_source}; KLRG1+ direction read within the Treg context (no cross-context extrapolation)",
    )


def plot_recipe_nociceptor_subtype_treatment_matrix(this_run_folder: str, *, data_BC, sampleinfo,
                                                    protein_gene_map, input_paths, params):
    """Nociceptor Fig. 4e-g equivalent: subtype × treatment effect matrix + module deltas."""
    subtypes = [str(s) for s in _recipe_param(params, "subtypes", [])]
    focus_genes = [str(g) for g in _recipe_param(params, "focus_genes", [])]
    modules = [str(m) for m in _recipe_param(params, "modules", [])]
    pval_cut = float(_recipe_param(params, "pval_cutoff", 0.05))
    rows = []
    for subtype in subtypes:
        table = f"processed_proteins/differential_{subtype}_Inflamed_vs_{subtype}_Control.csv"
        diff_df = _recipe_differential_table(input_paths, [table])
        lookup = _recipe_logfc_lookup(diff_df, focus_genes)
        for gene in focus_genes:
            info = lookup.get(str(gene).upper())
            if info is None:
                continue
            rows.append({"gene": str(gene), "subtype": subtype, "logFC": info["logFC"], "adj_p": info["adj_p"]})
    summary = _recipe_input_csv(input_paths, "evaluation_evidence/curated_module_group_summary.csv")
    module_rows = []
    if not summary.empty and not summary.empty and {"module", "group", "mean_score"}.issubset(summary.columns):
        for module in modules:
            for subtype in subtypes:
                inflamed = summary[(summary["module"].astype(str) == module)
                                   & (summary["group"].astype(str) == f"{subtype}_Inflamed")]
                control = summary[(summary["module"].astype(str) == module)
                                  & (summary["group"].astype(str) == f"{subtype}_Control")]
                if inflamed.empty or control.empty:
                    continue
                try:
                    delta = float(pd.to_numeric(inflamed["mean_score"], errors="coerce").mean()
                                  - pd.to_numeric(control["mean_score"], errors="coerce").mean())
                except Exception:
                    continue
                module_rows.append({"module": module, "subtype": subtype, "delta": delta})
    if not rows and not module_rows:
        return _placeholder_plot(this_run_folder, "recipe_nociceptor_subtype_treatment_matrix.png",
                                 "No subtype treatment evidence",
                                 "Within-subtype differential tables and module summaries carried no usable rows")
    file_path = _build_output_path(this_run_folder, "recipe_nociceptor_subtype_treatment_matrix.png")
    _set_visual_theme("notebook")
    if rows and module_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, max(4.0, 0.75 * len(focus_genes) + 2.0)),
                                 gridspec_kw={"width_ratios": [1.05, 1.0]}, squeeze=False)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7.4, max(4.0, 0.75 * max(len(focus_genes), len(modules)) + 2.0)),
                                 squeeze=False)
    if rows:
        long_df = pd.DataFrame(rows)
        matrix = long_df.pivot_table(index="gene", columns="subtype", values="logFC", aggfunc="mean")
        sig = long_df.pivot_table(index="gene", columns="subtype", values="adj_p", aggfunc="min") < pval_cut
        subtype_order = [s for s in subtypes if s in matrix.columns] + [c for c in matrix.columns if c not in subtypes]
        matrix = matrix[subtype_order]
        sig = sig.reindex(index=matrix.index, columns=matrix.columns).fillna(False)
        _recipe_direction_heatmap(axes[0][0], matrix, cbar_label="log2FC (Inflamed − Control)", sig_mask=sig)
        axes[0][0].set_title("Candidate proteins: subtype-internal treatment effect")
        axes[0][0].set_xlabel("Subtype")
        axes[0][0].set_ylabel("")
        _wrap_axis_ticklabels(axes[0][0], "x", width=12, rotation=0)
    if module_rows:
        ax = axes[0][1] if rows else axes[0][0]
        mod_df = pd.DataFrame(module_rows).pivot_table(index="module", columns="subtype", values="delta", aggfunc="mean")
        mod_df = mod_df.reindex(columns=[s for s in subtypes if s in mod_df.columns] + [c for c in mod_df.columns if c not in subtypes])
        x = np.arange(mod_df.shape[1])
        width = 0.8 / max(len(mod_df.index), 1)
        palette = sns.color_palette("colorblind", n_colors=len(mod_df.index))
        for mi, (module, row) in enumerate(mod_df.iterrows()):
            ax.bar(x + (mi - (len(mod_df.index) - 1) / 2) * width, row.to_numpy(dtype=float),
                   width=width * 0.92, color=palette[mi], label=str(module))
        ax.axhline(0, color="#666666", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(mod_df.columns)
        ax.set_ylabel("Δ module score (Inflamed − Control)")
        ax.set_title("Curated module response per subtype")
        _apply_nature_axes(ax)
        _legend_outside(ax, title="Module")
    fig.suptitle("Subtype × inflammation response matrix", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    sig_counts: Dict[str, int] = {}
    if rows:
        long_df = pd.DataFrame(rows)
        sig_counts = {s: int(((long_df["subtype"] == s) & (long_df["adj_p"] < pval_cut)).sum()) for s in subtypes}
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(rows)),
        n_sig=sig_counts or None,
        thresholds={"pval_cutoff": pval_cut,
                    "module_delta": {(r["module"], r["subtype"]): round(r["delta"], 4) for r in module_rows}},
        filter_note="weak within-subtype FDR is expected; the figure reports direction plus curated-module support without over-claiming",
    )


def plot_recipe_nociceptor_subtype_marker_heatmap(this_run_folder: str, *, data_BC, sampleinfo,
                                                  protein_gene_map, input_paths, params):
    """Extension: baseline subtype z-score protein heatmap (hierarchical row order)."""
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    baseline_tables = ([t for t in input_paths if str(t).startswith("processed_proteins/differential_")
                        and str(t).endswith("_Control.csv") and "Inflamed" not in str(t)]
                       if isinstance(input_paths, dict) else [])
    diff_df = _recipe_differential_table(input_paths, baseline_tables)
    top_genes: List[str] = []
    if not diff_df.empty:
        ranked = diff_df.reindex(diff_df["logFC"].abs().sort_values(ascending=False).index)
        seen = set()
        for gene in ranked["gene_label"].astype(str):
            key = gene.upper()
            if key and key != "NAN" and key not in seen:
                seen.add(key)
                top_genes.append(gene)
            if len(top_genes) >= 30:
                break
    if not top_genes:
        return _placeholder_plot(this_run_folder, "recipe_nociceptor_subtype_marker_heatmap.png",
                                 "No baseline contrast proteins",
                                 "Baseline subtype differential tables were unavailable or empty")
    mat = _recipe_gene_matrix(quant, gene_index, top_genes, sample_cols)
    if mat.empty:
        return _placeholder_plot(this_run_folder, "recipe_nociceptor_subtype_marker_heatmap.png",
                                 "Proteins unmatched", "Baseline top proteins could not be matched to the quantified matrix")
    groups = list(group_series.unique())
    mean_df = _recipe_group_mean_matrix(mat, group_series, groups)
    if mean_df.empty or mean_df.shape[1] < 2:
        return _placeholder_plot(this_run_folder, "recipe_nociceptor_subtype_marker_heatmap.png",
                                 "Subtype axis unavailable", "Fewer than two baseline subtypes were available")
    z_df = _recipe_row_zscore(mean_df)
    row_order = list(z_df.index)
    try:
        linkage_matrix = linkage(z_df.to_numpy(dtype=float), method="average", metric="euclidean")
        row_order = [str(z_df.index[i]) for i in leaves_list(linkage_matrix)]
    except Exception:
        pass
    z_df = z_df.loc[row_order]
    file_path = _build_output_path(this_run_folder, "recipe_nociceptor_subtype_marker_heatmap.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(max(5.4, 1.15 * z_df.shape[1] + 2.2), max(5.5, 0.24 * len(z_df) + 1.8)))
    sns.heatmap(z_df, ax=ax, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Mean row Z-score"})
    ax.set_title("Baseline subtype z-score protein clustering")
    ax.set_xlabel("Baseline subtype")
    ax.set_ylabel("")
    _apply_nature_heatmap_frame(ax)
    _wrap_axis_ticklabels(ax, "x", width=12, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(z_df)),
        filter_note=f"abundance source: {quant_source}; proteins ranked by pooled |logFC| across declared baseline contrasts; average-linkage row order",
    )


def plot_recipe_nociceptor_baseline_signature_compare(this_run_folder: str, *, data_BC, sampleinfo,
                                                      protein_gene_map, input_paths, params):
    """Extension: baseline subtype up-regulated enrichment signatures side by side."""
    tables = ([t for t in input_paths if str(t).startswith("enrichment_results/GO_") and "upregulated" in str(t)]
              if isinstance(input_paths, dict) else [])
    enrich = _recipe_enrichment_table(input_paths, tables)
    if enrich.empty:
        return _placeholder_plot(this_run_folder, "recipe_nociceptor_baseline_signature_compare.png",
                                 "No baseline enrichment tables",
                                 "None of the declared baseline GO upregulated tables were available")
    top_n = int(_recipe_param(params, "top_n_terms", 8))
    contrasts = list(dict.fromkeys(enrich["contrast"].astype(str).tolist()))
    ncols = min(3, len(contrasts))
    nrows = int(np.ceil(len(contrasts) / ncols))
    file_path = _build_output_path(this_run_folder, "recipe_nociceptor_baseline_signature_compare.png")
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 0.42 * top_n * nrows + 2.2), squeeze=False)
    top_terms: Dict[str, str] = {}
    for idx, contrast in enumerate(contrasts):
        ax = axes[idx // ncols][idx % ncols]
        sub = enrich[enrich["contrast"].astype(str) == contrast].sort_values("p.adjust").head(top_n)
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.assign(score=[-np.log10(max(float(p), 1e-300)) for p in sub["p.adjust"]]).sort_values("score")
        ax.barh(np.arange(len(sub)), sub["score"].to_numpy(dtype=float), color="#54A24B", edgecolor="none")
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels([_wrap_label(t, 36) for t in sub["Description"]], fontsize=7.5)
        ax.set_xlabel("−log10(adjusted P)")
        ax.set_title(_wrap_label(contrast.replace("_vs_", " vs "), 30), fontsize=10)
        _apply_nature_axes(ax)
        top_terms[contrast] = str(sub.iloc[-1]["Description"])
    for leftover in range(len(contrasts), nrows * ncols):
        axes[leftover // ncols][leftover % ncols].axis("off")
    fig.suptitle("Baseline subtype up-regulated signatures", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        n_contrasts=int(len(contrasts)),
        thresholds={"top_term_by_contrast": top_terms},
        filter_note="ORA-based signature comparison on the declared baseline contrasts; verify TrkA metabolism vs IB4/mechano C-fiber direction in captions",
    )


def plot_recipe_carr_lps_reactome_signature(this_run_folder: str, *, data_BC, sampleinfo,
                                            protein_gene_map, input_paths, params):
    """Carr Fig. 3e equivalent: Reactome LPS_vs_DMSO signature bar."""
    contrast = str(_recipe_param(params, "contrast", "LPS_vs_DMSO"))
    top_n = int(_recipe_param(params, "top_n_terms", 12))
    tables = ([t for t in input_paths
               if str(t).startswith("enrichment_results/Reactome_") and contrast in str(t)]
              if isinstance(input_paths, dict) else [])
    enrich = _recipe_enrichment_table(input_paths, tables)
    if enrich.empty:
        return _placeholder_plot(this_run_folder, "recipe_carr_lps_reactome_signature.png",
                                 "No Reactome table", f"No Reactome enrichment table matched contrast {contrast}")
    enrich = enrich.sort_values("p.adjust").head(top_n).copy()
    enrich["score"] = [-np.log10(max(float(p), 1e-300)) for p in enrich["p.adjust"]]
    direction_from_json = _recipe_reactome_direction_from_json(input_paths, contrast, enrich)
    enrich = enrich.assign(score=enrich["score"] * direction_from_json.reindex(enrich.index).fillna(1.0))
    enrich = enrich.sort_values("score")
    file_path = _build_output_path(this_run_folder, "recipe_carr_lps_reactome_signature.png")
    _set_visual_theme("notebook")
    fig, ax = plt.subplots(figsize=(9.8, max(4.2, 0.38 * len(enrich) + 1.6)))
    y = np.arange(len(enrich))
    colors = ["#B2182B" if v >= 0 else "#2166AC" for v in enrich["score"]]
    ax.barh(y, enrich["score"].to_numpy(dtype=float), color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap_label(str(d), 46) for d in enrich["Description"]], fontsize=8)
    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("Signed −log10(adjusted P)  (up: red, down: blue)")
    ax.set_title(f"Reactome signature: {contrast.replace('_', ' ')}")
    _apply_nature_axes(ax)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        n_up=int((enrich["score"] >= 0).sum()),
        n_down=int((enrich["score"] < 0).sum()),
        thresholds={"n_terms": int(len(enrich)),
                    "top_term": str(enrich.iloc[-1]["Description"]) if len(enrich) else None,
                    "pval_cutoff": 0.05},
        filter_note="signed direction cross-checked against the go_kegg_reactome_results.json gene→term mapping; ORA significance, not NES",
    )


def _recipe_reactome_direction_from_json(input_paths: Dict[str, str], contrast: str, enrich: pd.DataFrame) -> pd.Series:
    """+1/−1 per term from up/down gene→term mappings in go_kegg_reactome_results.json."""
    data = _recipe_input_json(input_paths, "enrichment_results/go_kegg_reactome_results.json")
    up_counts: Dict[str, int] = {}
    down_counts: Dict[str, int] = {}

    def _collect(node: Any, direction: str) -> None:
        if isinstance(node, dict):
            for term_key, term_value in node.items():
                description = str(term_value) if not isinstance(term_value, dict) else str(term_value)
                if not description:
                    continue
                target = up_counts if direction == "upregulated" else down_counts
                target[description] = target.get(description, 0) + 1

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if contrast in node:
            branch = node[contrast]
            if isinstance(branch, dict):
                _collect(branch.get("upregulated", {}), "upregulated")
                _collect(branch.get("downregulated", {}), "downregulated")
            return
        for value in node.values():
            _walk(value)

    _walk(data)
    if not up_counts and not down_counts:
        return pd.Series(dtype="float64")
    signs = []
    for description in enrich["Description"].astype(str):
        n_up = up_counts.get(description, 0)
        n_down = down_counts.get(description, 0)
        signs.append(1.0 if n_up >= n_down else -1.0)
    return pd.Series(signs, index=enrich.index)


def plot_recipe_carr_batch_treatment_boundary(this_run_folder: str, *, data_BC, sampleinfo,
                                              protein_gene_map, input_paths, params):
    """Carr Supp Fig. 5 equivalent: batch vs treatment effect before/after correction."""
    treatment_col = str(_recipe_param(params, "treatment_col", "Cluster"))
    batch_col = str(_recipe_param(params, "batch_col", "Batch"))
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    if sampleinfo_eff.empty or batch_col not in sampleinfo_eff.columns:
        return _placeholder_plot(this_run_folder, "recipe_carr_batch_treatment_boundary.png",
                                 "No batch annotation", "SampleInfo without a Batch column cannot quantify batch vs treatment effects")
    stages = [("Filtered", "processed_proteins/ProteinQuant_Filtered.csv"),
              ("Normalized", "processed_proteins/ProQuant_Normalized.csv"),
              ("ComBat", "processed_proteins/ProteinQuant_ComBat.csv")]
    r_squared: Dict[str, Dict[str, float]] = {}
    matrices: Dict[str, pd.DataFrame] = {}
    for label, table in stages:
        quant, _ = _recipe_quant_matrix(input_paths, pd.DataFrame(), table)
        sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
        if not sample_cols or len(sample_cols) < 4:
            continue
        matrices[label] = quant[sample_cols]
        block = quant[sample_cols].apply(pd.to_numeric, errors="coerce")
        batch_labels = _recipe_group_series(sampleinfo_eff, sample_cols, batch_col)
        treat_labels = _recipe_group_series(sampleinfo_eff, sample_cols, treatment_col)
        values = block.to_numpy(dtype=float)

        def _eta_squared(labels: pd.Series) -> float:
            """Median per-protein one-way ANOVA η² across all proteins."""
            groups: Dict[str, List[int]] = {}
            for gi, sample in enumerate(sample_cols):
                group_label = labels.get(sample)
                if group_label == group_label:  # skip NaN labels
                    groups.setdefault(str(group_label), []).append(gi)
            if len(groups) < 2:
                return np.nan
            etas: List[float] = []
            for row in values:
                grand = np.nanmean(row)
                if np.isnan(grand):
                    continue
                ss_total = float(np.nansum((row - grand) ** 2))
                if not ss_total:
                    continue
                ss_between = 0.0
                for cols in groups.values():
                    vals = row[cols]
                    vals = vals[~np.isnan(vals)]
                    if len(vals):
                        ss_between += len(vals) * (float(np.mean(vals)) - grand) ** 2
                etas.append(ss_between / ss_total)
            return float(np.nanmedian(etas)) if etas else np.nan

        r_squared[label] = {
            "batch": round(_eta_squared(batch_labels), 4),
            "treatment": round(_eta_squared(treat_labels), 4),
        }
    if not r_squared:
        return _placeholder_plot(this_run_folder, "recipe_carr_batch_treatment_boundary.png",
                                 "No aligned matrices", "None of the declared quantification stages could be aligned with the sample sheet")
    file_path = _build_output_path(this_run_folder, "recipe_carr_batch_treatment_boundary.png")
    _set_visual_theme("notebook")
    fig = plt.figure(figsize=(13.6, 11.6))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    stage_names = [s for s, _ in stages if s in r_squared]
    x = np.arange(len(stage_names))
    width = 0.36
    ax.bar(x - width / 2, [r_squared[s]["batch"] for s in stage_names], width=width,
           color="#9D755D", label="Batch effect (median η²)")
    ax.bar(x + width / 2, [r_squared[s]["treatment"] for s in stage_names], width=width,
           color="#4C78A8", label="Treatment effect (median η²)")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel("Variance explained (η², top-variance proteins)")
    ax.set_title("Batch vs treatment effect across processing stages")
    _apply_nature_axes(ax)
    _legend_outside(ax)
    pca_specs = [("Filtered", batch_col, "Filtered · colored by batch"),
                 ("Filtered", treatment_col, "Filtered · colored by treatment"),
                 ("ComBat", batch_col, "ComBat · colored by batch"),
                 ("ComBat", treatment_col, "ComBat · colored by treatment")]
    pca_slots = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for idx, ((stage, color_col, title), (row, col)) in enumerate(zip(pca_specs, pca_slots)):
        ax = fig.add_subplot(gs[row, col])
        if stage not in matrices:
            ax.axis("off")
            continue
        coords = _recipe_pca_coords(matrices[stage], list(matrices[stage].columns), max_proteins=1200)
        if coords.empty:
            ax.axis("off")
            continue
        labels = _recipe_group_series(sampleinfo_eff, list(coords.index), color_col)
        palette = _categorical_palette(labels.unique())
        for group in palette:
            samples = [s for s in coords.index if labels.get(s) == group]
            if samples:
                ax.scatter(coords.loc[samples, "PC1"], coords.loc[samples, "PC2"], s=22,
                           color=palette[group], alpha=0.8, linewidths=0, label=str(group))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        _apply_nature_axes(ax)
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels_, fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("LPS effect vs batch boundary (correction before/after)", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    combat_noop = False
    if "ComBat" in matrices and "Normalized" in matrices:
        try:
            combat_noop = bool(np.allclose(matrices["ComBat"].to_numpy(dtype=float),
                                           matrices["Normalized"].to_numpy(dtype=float),
                                           rtol=1e-6, atol=1e-6, equal_nan=True))
        except Exception:
            combat_noop = False
    noop_note = (" ComBat output equals the normalized matrix in this run (batch correction was a no-op); "
                 "batch structure therefore remains in the final matrix." if combat_noop else "")
    return _recipe_save_info(
        file_path, fig,
        thresholds={"eta_squared_by_stage": r_squared, "combat_equals_normalized": combat_noop},
        filter_note="η² = median per-protein one-way ANOVA variance explained across all proteins per stage;"
                    " PCA panels use top-variance proteins." + noop_note,
    )


def plot_recipe_carr_per_cell_depth_bar(this_run_folder: str, *, data_BC, sampleinfo,
                                        protein_gene_map, input_paths, params):
    """Carr Fig. 3b equivalent: per-sample protein detection depth by treatment."""
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    if not sample_cols or sampleinfo_eff.empty:
        return _placeholder_plot(this_run_folder, "recipe_carr_per_cell_depth_bar.png",
                                 "No aligned samples", "SampleInfo and the quantified matrix could not be aligned on FileName")
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    block = quant[sample_cols].apply(pd.to_numeric, errors="coerce")
    detected = (block > 0).sum(axis=0)
    depth_df = pd.DataFrame({"group": [str(group_series.get(s, UNANNOTATED_GROUP_LABEL)) for s in sample_cols],
                             "detected": detected.to_numpy(dtype=float)}, index=sample_cols)
    groups_present = _recipe_ordered_present(None, depth_df["group"].unique())
    palette = _categorical_palette(groups_present)
    file_path = _build_output_path(this_run_folder, "recipe_carr_per_cell_depth_bar.png")
    _set_visual_theme("notebook")
    fig_width = min(18.0, max(7.5, 0.10 * len(depth_df) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ticks = []
    pos = 0
    stats: Dict[str, Dict[str, float]] = {}
    for group in groups_present:
        sub = depth_df[depth_df["group"] == group].sort_values("detected")
        start = pos
        for sample, row in sub.iterrows():
            ax.bar(pos, row["detected"], width=0.78, color=palette[group], edgecolor="none")
            pos += 1
        ticks.append((start + pos - 1) / 2 if len(sub) else pos)
        median = float(sub["detected"].median()) if len(sub) else np.nan
        mad = float((sub["detected"] - median).abs().median()) if len(sub) else np.nan
        stats[group] = {"median": round(median, 1) if not np.isnan(median) else None,
                        "mad": round(mad, 1) if not np.isnan(mad) else None,
                        "n": int(len(sub))}
        if not np.isnan(median):
            ax.hlines(median, start - 0.45, pos - 0.55, color="#222222", linewidth=1.4, linestyle="--", zorder=4)
        pos += 0.9
    ax.set_xticks(ticks)
    ax.set_xticklabels([_wrap_label(g, 14) for g in groups_present], rotation=20, ha="right")
    ax.set_ylabel("Detected protein groups per sample")
    ax.set_title("Per-sample detection depth by treatment")
    _apply_nature_axes(ax)
    plt.tight_layout()
    return _recipe_save_info(
        file_path, fig,
        thresholds={"per_group": stats},
        filter_note=f"abundance source: {quant_source}; bar = one sample; dashed line = group median; detection = nonzero intensity",
    )


def plot_recipe_ipsc_pluripotency_gradient(this_run_folder: str, *, data_BC, sampleinfo,
                                           protein_gene_map, input_paths, params):
    """iPSC Fig. 6b,c equivalent: pluripotency-colored PCA + marker group boxes."""
    pluripotency_markers = [str(g) for g in _recipe_param(params, "pluripotency_markers", [])]
    lineage_markers = [str(g) for g in _recipe_param(params, "lineage_markers", [])]
    group_order = [str(g) for g in _recipe_param(params, "group_order", [])]
    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    mat = _recipe_gene_matrix(quant, gene_index, pluripotency_markers + lineage_markers, sample_cols)
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    if mat.empty or sampleinfo_eff.empty:
        return _placeholder_plot(this_run_folder, "recipe_ipsc_pluripotency_gradient.png",
                                 "No markers matched",
                                 "Declared pluripotency/lineage markers were absent from the quantified matrix")
    pluripotency_present = [m for m in pluripotency_markers if m in mat.index]
    per_gene_z = _recipe_row_zscore(mat)  # z per protein across samples
    pluripotency_z = (per_gene_z.loc[pluripotency_present].mean(axis=0, skipna=True)
                      if pluripotency_present else pd.Series(dtype="float64"))
    coords = _recipe_pca_coords(quant, sample_cols)
    file_path = _build_output_path(this_run_folder, "recipe_ipsc_pluripotency_gradient.png")
    groups_present = _recipe_ordered_present(group_order, list(group_series.unique()))
    _set_visual_theme("notebook")
    marker_count = len(mat.index)
    ncols = min(3, max(marker_count, 1))
    nrows = int(np.ceil(max(marker_count, 1) / ncols))
    fig = plt.figure(figsize=(5.6 + 3.1 * ncols, 4.6 + 2.9 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, hspace=0.55, wspace=0.34)
    if not coords.empty and len(pluripotency_z.dropna()) >= 3:
        ax = fig.add_subplot(gs[0, :])
        aligned = pluripotency_z.reindex(coords.index).dropna()
        vmin, vmax = float(aligned.min()), float(aligned.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        sc = ax.scatter(coords.loc[aligned.index, "PC1"], coords.loc[aligned.index, "PC2"],
                        c=aligned.to_numpy(dtype=float), cmap="plasma", s=36, linewidths=0.5, edgecolors="white")
        fig.colorbar(sc, ax=ax, label="Pluripotency marker mean Z-score", shrink=0.85)
        markers = ["o", "s", "^", "D"]
        for gi, group in enumerate(groups_present):
            samples = [s for s in aligned.index if group_series.get(s) == group]
            if samples:
                ax.scatter(coords.loc[samples, "PC1"], coords.loc[samples, "PC2"], s=120,
                           marker=markers[gi % len(markers)], facecolors="none", edgecolors="#111111",
                           linewidths=1.1)
        from matplotlib.lines import Line2D
        group_handles = [Line2D([0], [0], marker=markers[gi % len(markers)], linestyle="", markersize=9,
                                markerfacecolor="none", markeredgecolor="#111111", label=str(group))
                         for gi, group in enumerate(groups_present)]
        ax.legend(handles=group_handles, title="Group", loc="upper right", frameon=True, fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA colored by pluripotency signature")
        _apply_nature_axes(ax)
    palette = _categorical_palette(groups_present)
    ttest_results: Dict[str, Dict[str, float]] = {}
    for mi, gene in enumerate(mat.index):
        row = gs[1 + mi // ncols, mi % ncols] if marker_count > 0 else None
        if row is None:
            break
        ax = fig.add_subplot(row)
        data_by_group = []
        for group in groups_present:
            samples = [s for s in sample_cols if group_series.get(s) == group]
            values = pd.to_numeric(mat.loc[gene, samples], errors="coerce").dropna() if samples else pd.Series(dtype="float64")
            data_by_group.append(values.to_numpy(dtype=float))
        boxes = ax.boxplot(data_by_group, patch_artist=True, widths=0.55, showfliers=False,
                           medianprops=dict(color="#222222", linewidth=1.1))
        for patch, group in zip(boxes["boxes"], groups_present):
            patch.set_facecolor(palette[group])
            patch.set_alpha(0.5)
            patch.set_edgecolor("#444444")
        rng = np.random.default_rng(71 + mi)
        for gi, values in enumerate(data_by_group):
            if len(values) == 0:
                continue
            jitter = rng.uniform(-0.1, 0.1, size=len(values))
            ax.scatter(np.full(len(values), gi + 1) + jitter, values, s=8, color="#333333",
                       alpha=0.55, linewidths=0, rasterized=True)
        if len(data_by_group) >= 2 and min(len(v) for v in data_by_group) >= 2:
            statistic, p_value = ttest_ind(data_by_group[0], data_by_group[1], equal_var=False)
            ttest_results[str(gene)] = {"p": float(f"{p_value:.3g}"),
                                        "mean_first_group": round(float(np.mean(data_by_group[0])), 3),
                                        "mean_last_group": round(float(np.mean(data_by_group[-1])), 3)}
        ax.set_xticks(range(1, len(groups_present) + 1))
        ax.set_xticklabels([_wrap_label(g, 10) for g in groups_present], fontsize=7.5, rotation=20, ha="right")
        ax.set_title(str(gene), fontsize=9)
        ax.tick_params(axis="y", labelsize=7.5)
        _apply_nature_axes(ax)
    for leftover in range(marker_count, nrows * ncols):
        try:
            fig.add_subplot(gs[1 + leftover // ncols, leftover % ncols]).axis("off")
        except Exception:
            break
    fig.suptitle("iPSC → EB pluripotency decline and lineage rise", fontsize=15, weight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(marker_count),
        thresholds={"pval_cutoff": 0.05, "welch_ttest": ttest_results} if ttest_results else None,
        filter_note=f"abundance source: {quant_source}; two-sided Welch t-test between first/last group reported in caption metadata",
    )


def plot_recipe_ipsc_eb_heterogeneity_clusters(this_run_folder: str, *, data_BC, sampleinfo,
                                               protein_gene_map, input_paths, params):
    """Extension: within-EB heterogeneity clustering + per-cluster GO annotation."""
    from scipy.cluster.hierarchy import fcluster

    sampleinfo_eff = _recipe_effective_sampleinfo(sampleinfo, input_paths)
    quant, quant_source = _recipe_quant_matrix(input_paths, data_BC)
    gene_index = _recipe_gene_index(quant, protein_gene_map)
    sample_cols = _recipe_sample_columns(quant, sampleinfo_eff)
    diff_df = _recipe_differential_table(input_paths, ["processed_proteins/differential_EB_vs_iPSCs.csv"])
    if diff_df.empty:
        return _placeholder_plot(this_run_folder, "recipe_ipsc_eb_heterogeneity_clusters.png",
                                 "No differential table", "differential_EB_vs_iPSCs.csv was unavailable")
    pval_cut = float(_recipe_param(params, "pval_cutoff", 0.05))
    significant = diff_df[diff_df["adj.P.Val"] < pval_cut] if "adj.P.Val" in diff_df.columns else diff_df.iloc[0:0]
    if len(significant) < 10:
        significant = diff_df.sort_values("adj.P.Val").head(50)
    genes = [g for g in significant["gene_label"].map(
        lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v)
    ).tolist() if g and g.upper() != "NAN"][:200]
    group_series = _recipe_group_series(sampleinfo_eff, sample_cols, "Cluster")
    eb_samples = [s for s in sample_cols if group_series.get(s) == "EB"]
    if not eb_samples:
        target_group = _recipe_param(params, "target_group", "")
        eb_samples = [s for s in sample_cols if group_series.get(s) == str(target_group)] if target_group else sample_cols
    mat = _recipe_gene_matrix(quant, gene_index, genes, eb_samples)
    if mat.empty or mat.shape[1] < 3:
        return _placeholder_plot(this_run_folder, "recipe_ipsc_eb_heterogeneity_clusters.png",
                                 "EB samples unavailable", "No EB sample columns could be aligned with the significant protein set")
    z_df = _recipe_row_zscore(mat)
    n_clusters = int(_recipe_param(params, "n_clusters", 4))
    cluster_labels: np.ndarray = np.zeros(len(z_df), dtype=int)
    try:
        linkage_matrix = linkage(z_df.to_numpy(dtype=float), method="average", metric="euclidean")
        cluster_labels = fcluster(linkage_matrix, t=min(n_clusters, len(z_df)), criterion="maxclust") - 1
        order = np.lexsort((np.arange(len(z_df)), cluster_labels))
        z_df = z_df.iloc[order]
        cluster_labels = cluster_labels[order]
    except Exception:
        pass
    go_tables = ([t for t in input_paths if str(t).startswith("enrichment_results/GO_EB_vs_iPSCs")]
                 if isinstance(input_paths, dict) else [])
    enrich = _recipe_enrichment_table(input_paths, go_tables)
    file_path = _build_output_path(this_run_folder, "recipe_ipsc_eb_heterogeneity_clusters.png")
    _set_visual_theme("notebook")
    fig_height = min(15.0, max(6.0, 0.2 * len(z_df) + 2.0))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.0, fig_height),
                                  gridspec_kw={"width_ratios": [1.6, 1.0]})
    sns.heatmap(z_df, ax=ax, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0,
                vmin=-2.5, vmax=2.5, rasterized=True, cbar_kws={"label": "Row Z-score"},
                yticklabels=len(z_df) <= 60, xticklabels=True)
    ax.set_title("Within-EB clustering of regulated proteins")
    ax.set_xlabel("EB sample")
    ax.set_ylabel("")
    _apply_nature_heatmap_frame(ax)
    cluster_palette = sns.color_palette("colorblind", n_colors=max(int(cluster_labels.max()) + 1, 1))
    for ci in range(int(cluster_labels.max()) + 1):
        indices = np.where(cluster_labels == ci)[0]
        if len(indices):
            ax.axhspan(indices.min(), indices.max() + 1, color=cluster_palette[ci], alpha=0.08, zorder=0)
    cluster_ids = list(range(int(cluster_labels.max()) + 1))
    cluster_top_genes: Dict[str, List[str]] = {}
    annotation_rows: List[Dict[str, Any]] = []
    has_term_gene_lists = (not enrich.empty and "geneID" in enrich.columns)
    if has_term_gene_lists:
        for ci in cluster_ids:
            cluster_genes = {str(z_df.index[i]).upper() for i in np.where(cluster_labels == ci)[0]}
            if not cluster_genes:
                continue
            best_overlap, best_term, best_direction = 0, "", ""
            for _, row in enrich.iterrows():
                term_genes = {g.strip().upper() for g in str(row.get("geneID", "")).replace("/", ";").split(";") if g.strip()}
                overlap = len(term_genes & cluster_genes)
                if overlap > best_overlap:
                    best_overlap, best_term, best_direction = overlap, str(row["Description"]), str(row["direction"])
            if best_term:
                annotation_rows.append({"cluster": f"C{ci + 1} (n={int((cluster_labels == ci).sum())})",
                                        "term": best_term, "overlap": best_overlap, "direction": best_direction})
    # Right panel: per-cluster centroid activity across EB samples + top marker
    # genes. Functional GO annotation is only drawn when the run's enrichment
    # tables actually carry per-term gene lists (geneID column).
    ax2.clear()
    x_samples = np.arange(1, z_df.shape[1] + 1)
    for ci in cluster_ids:
        indices = np.where(cluster_labels == ci)[0]
        if not len(indices):
            continue
        block = z_df.iloc[indices]
        centroid = block.mean(axis=0).to_numpy(dtype=float)
        gene_means = block.mean(axis=1)
        top_markers = [str(g) for g in gene_means.sort_values(ascending=False).head(4).index]
        cluster_top_genes[f"C{ci + 1}"] = top_markers
        ax2.plot(x_samples, centroid, color=cluster_palette[ci], linewidth=1.6,
                 label=f"C{ci + 1} (n={len(indices)}): {', '.join(top_markers[:3])}")
    ax2.axhline(0, color="#999999", linewidth=0.9, linestyle="--")
    ax2.set_xlabel("EB sample (heatmap column order)")
    ax2.set_ylabel("Cluster centroid row Z-score")
    ax2.set_title("Per-cluster activity and markers", fontsize=11, loc="left")
    ax2.tick_params(axis="x", labelbottom=False)
    _apply_nature_axes(ax2)
    ax2.legend(fontsize=6.8, frameon=False, loc="upper right")
    if annotation_rows:
        ax2.clear()
        ax2.axis("off")
        table_text = "\n".join(
            f"{r['cluster']}: {r['term']} (n={r['overlap']}, {r['direction'].replace('downregulated', 'down').replace('upregulated', 'up')})"
            for r in annotation_rows)
        ax2.text(0.0, 0.98, table_text, va="top", ha="left", fontsize=8.2, family="monospace", color="#222222",
                 transform=ax2.transAxes)
        ax2.set_title("Top overlapping GO term per cluster", fontsize=11, loc="left")
    fig.suptitle("EB internal heterogeneity", fontsize=15, weight="bold", y=1.01)
    plt.tight_layout()
    cluster_sizes = {f"C{ci + 1}": int((cluster_labels == ci).sum()) for ci in range(int(cluster_labels.max()) + 1)}
    go_note = ("per-cluster GO overlap drawn from enrichment geneID lists" if annotation_rows else
               "run's GO tables carry no per-term gene lists (no geneID column); panel shows per-cluster markers instead")
    return _recipe_save_info(
        file_path, fig,
        n_proteins=int(len(z_df)),
        thresholds={"n_clusters": int(cluster_labels.max() + 1), "cluster_sizes": cluster_sizes,
                    "pval_cutoff": pval_cut,
                    "cluster_top_markers": cluster_top_genes,
                    "cluster_top_go": {r["cluster"]: r["term"] for r in annotation_rows}} if annotation_rows else {"n_clusters": int(cluster_labels.max() + 1), "cluster_sizes": cluster_sizes, "pval_cutoff": pval_cut, "cluster_top_markers": cluster_top_genes},
        filter_note=f"abundance source: {quant_source}; row Z-score of significant proteins across EB samples; average-linkage clusters; {go_note}",
    )


@tool
def visualize(
        plot_set: Any = "full",
        formats: List[str] | None = None,
        top_n_proteins: int = 50,
        top_n_terms: int = 15,
        max_labels: int = 12,
        include_enrichment: bool = True,
        *args, **kwargs
    ):
    """
    Generate static visual summaries for the current proteomics analysis.
    """

    requested_plot_set = plot_set
    if isinstance(plot_set, (list, tuple, set)):
        plot_set_items = [str(item).strip().lower() for item in plot_set if str(item).strip()]
    else:
        plot_set_items = [str(plot_set).strip().lower()] if str(plot_set).strip() else ["full"]
    supported_plot_sets = {"full", "mechanism", "all", "standard", "basic"}
    if not any(item in supported_plot_sets for item in plot_set_items):
        plot_set_mode = "full"
        unsupported_plot_set_items = plot_set_items
    elif any(item in {"full", "all", "mechanism"} for item in plot_set_items):
        plot_set_mode = "full"
        unsupported_plot_set_items = [item for item in plot_set_items if item not in supported_plot_sets]
    else:
        plot_set_mode = "standard"
        unsupported_plot_set_items = [item for item in plot_set_items if item not in supported_plot_sets]

    formats = formats or ["png"]
    formats = [str(fmt).lower() for fmt in formats]
    if "png" not in formats:
        formats.append("png")
    # Step 05: optional per-recipe parameter overrides keyed by recipe_id.
    figure_recipe_params = kwargs.pop("figure_recipe_params", None)

    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    processed_proteins_folder = _build_output_dir(this_run_folder, 'processed_proteins')
    enrichment_folder = _build_output_dir(this_run_folder, 'enrichment_results')
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')
    protein_gene_map_path = os.path.join(processed_proteins_folder, 'Protein_Gene_Map.csv')

    if not _path_exists(protein_quant_combat_path) or not _path_exists(protein_gene_map_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    data_BC = _read_csv_checked(protein_quant_combat_path, description="ComBat protein quantification file")
    protein_gene_map = _read_csv_checked(protein_gene_map_path, description="protein-gene map file")
    sampleinfo = _read_csv_checked(sampleinfo_path, description="sampleinfo file")

    group_col = "Cluster"
    if group_col not in sampleinfo.columns:
        processed_proteins_folder = os.path.join(resolve_path('this_run_folder_path'), 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = _read_csv_checked(sampleinfo_with_cluster_path, description="sampleinfo_with_clusters file")
        else:
            run_umap.func()
            if os.path.exists(sampleinfo_with_cluster_path):
                sampleinfo = _read_csv_checked(sampleinfo_with_cluster_path, description="sampleinfo_with_clusters file")
            else:
                raise ValueError(f"Neither '{group_col}' column found in sampleinfo nor sampleinfo_with_clusters.csv generated. Please check the UMAP clustering step.")

    differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv")]
    enrichment_results_list = [fn for fn in os.listdir(enrichment_folder) if fn.endswith(".csv")] if os.path.exists(enrichment_folder) else []
    if len(differential_proteins_path_list) == 0:
        print("No differential protein files found in processed_proteins folder. Try to run limma first.")
        _ = run_pairwise_limma.func()
    if len(differential_proteins_path_list) == 0:
        raise FileNotFoundError("No differential protein files found in processed_proteins folder after running limma.")

    top_n_proteins = max(6, min(int(top_n_proteins), 30))
    top_n_terms = max(3, min(int(top_n_terms), 15))
    max_labels = max(6, min(int(max_labels), 12))
    max_contrast_figures = 12 if len(sampleinfo) > 1000 else 24

    results = {}
    figure_records: List[Dict[str, Any]] = []
    contrast_col = 'contrast'
    design_diff = _get_analysis_design().get("differential", {})
    viz_pval_cut = float(design_diff.get("p_thresh", 0.05))
    viz_logfc_cut = float(design_diff.get("logfc_thresh", 1.0))
    visualize_params = {
        "plot_set_requested": requested_plot_set,
        "plot_set_effective": plot_set_mode,
        "unsupported_plot_set_items": unsupported_plot_set_items,
        "formats": formats,
        "top_n_proteins": int(top_n_proteins),
        "top_n_terms": int(top_n_terms),
        "max_labels": int(max_labels),
        "include_enrichment": bool(include_enrichment),
        "pval_cut": viz_pval_cut,
        "logfc_cut": viz_logfc_cut,
    }

    _build_output_dir(this_run_folder, 'visualize_results')

    # Step 04: requested formats beyond PNG (pdf/svg) are exported alongside the
    # always-generated PNG. The registry resets on the next visualize() call, so
    # a stale value cannot leak into later runs.
    _set_active_vector_formats(formats)

    heatmap_out = plot_heatmap(this_run_folder, data_BC=data_BC, sampleinfo=sampleinfo, protein_gene_map=protein_gene_map, out_file="heatmap.png")
    _add_figure_record(figure_records, heatmap_out, "Sample structure", "Differential protein heatmap using top informative features.", [protein_quant_combat_path, sampleinfo_path, protein_gene_map_path], visualize_params, "heatmap")
    pca_out = plot_pca(this_run_folder, data_BC=data_BC, sampleinfo=sampleinfo, out_file="pca_plot.png")
    _add_figure_record(figure_records, pca_out, "Sample structure", "PCA sample structure plot with group centers and ellipses.", [protein_quant_combat_path, sampleinfo_path], visualize_params, "pca")
    umap_out = plot_umap(this_run_folder, data_BC=data_BC, sampleinfo=sampleinfo, out_file="umap_plot.png")
    _add_figure_record(figure_records, umap_out, "Sample structure", "UMAP sample structure plot with group centers and ellipses.", [protein_quant_combat_path, sampleinfo_path], visualize_params, "umap")
    differential_file_paths = [os.path.join(processed_proteins_folder, fn) for fn in differential_proteins_path_list]
    key_protein_out = plot_key_protein_overview(
        this_run_folder,
        differential_file_paths=differential_file_paths,
        out_file="key_protein_overview.png",
        pval_cut=viz_pval_cut,
        top_n_per_contrast=max(2, min(int(max_labels), 2))
    )
    _add_figure_record(figure_records, key_protein_out, "Differential evidence", "Top significant proteins across contrasts.", differential_file_paths, visualize_params, "key_protein_overview")
    differential_summary_barplot_out = plot_differential_summary_barplot(
        this_run_folder,
        differential_file_paths=differential_file_paths,
        out_file="differential_summary_barplot.png",
        pval_cut=viz_pval_cut,
        logfc_cut=viz_logfc_cut
    )
    _add_figure_record(figure_records, differential_summary_barplot_out, "Differential evidence", "Up/down significant protein counts for each contrast.", differential_file_paths, visualize_params, "differential_summary_barplot")
    protein_contrast_bubble_out = plot_protein_contrast_bubble(
        this_run_folder,
        differential_file_paths=differential_file_paths,
        out_file="protein_contrast_bubble.png",
        pval_cut=viz_pval_cut,
        top_n_per_contrast=6,
        max_proteins=30
    )
    _add_figure_record(figure_records, protein_contrast_bubble_out, "Differential evidence", "Bubble map of key proteins across contrasts.", differential_file_paths, visualize_params, "protein_contrast_bubble")
    volcano_count = 0
    for file in differential_proteins_path_list:
        file_path = os.path.join(processed_proteins_folder, file)
        df = _read_csv_checked(file_path, description="differential protein file")
        for contrast, subdf in df.groupby(contrast_col):
            if volcano_count >= max_contrast_figures:
                continue
            safe_contrast = contrast.replace(" ", "")  # ✅ 统一安全命名
            volcano_out_file = f"{safe_contrast}_volcano_plot.png"
            volcano_out = plot_volcano(
                this_run_folder,
                df=subdf,
                contrast=contrast,
                out_file=volcano_out_file,
                pval_cut=viz_pval_cut,
                logfc_cut=viz_logfc_cut,
            )
            _add_figure_record(figure_records, volcano_out, "Differential evidence", f"Volcano plot for contrast `{contrast}`.", [file_path], visualize_params, "volcano")
            volcano_count += 1

            # cluster_out_file = f"{safe_contrast}_cluster_plot.png"
            # cluster_out = plot_cluster(this_run_folder, df=subdf, out_file=cluster_out_file)

            results[safe_contrast] = {
                "volcano_plot": volcano_out,
                # "cluster_plot": cluster_out,
            }

    enrichment_file_paths = [os.path.join(enrichment_folder, fn) for fn in enrichment_results_list]
    enrichment_summary = _load_enrichment_summary(enrichment_file_paths, top_n_terms=int(top_n_terms)) if include_enrichment else pd.DataFrame()
    enrichment_bubbles = []
    if include_enrichment:
        for file in enrichment_results_list:
            file_path = os.path.join(enrichment_folder, file)
            df = _read_csv_checked(file_path, description="enrichment result file")
            bubble = plot_enrichment_bubble(this_run_folder, enrich_df=df, out_file=file.replace(".csv", "_enrichment_bubble.png"), top_n=int(top_n_terms))
            enrichment_bubbles.append(bubble)
            _add_figure_record(figure_records, bubble, "Enrichment evidence", f"Enrichment bubble plot derived from `{file}`.", [file_path], visualize_params, "enrichment_bubble")
            # plot_enrichment_bar(this_run_folder, enrich_df=df, out_file=file.replace(".csv", "_enrichment_bubble.png"), top_n=15)

    mechanism_outputs: Dict[str, Any] = {}
    if plot_set_mode in {"full", "mechanism", "all"}:
        dep_df = _collect_differential_records(differential_file_paths, pval_cut=viz_pval_cut, logfc_cut=viz_logfc_cut)

        qc_out = plot_qc_sample_overview(this_run_folder, data_BC=data_BC, sampleinfo=sampleinfo)
        mechanism_outputs["qc_sample_overview"] = qc_out
        _add_figure_record(figure_records, qc_out, "QC", "Sample-level detected proteins, missingness, group sizes, and correlation structure.", [protein_quant_combat_path, sampleinfo_path], visualize_params, "qc_sample_overview")

        top_means_out = plot_mechanism_top_protein_group_means(
            this_run_folder,
            dep_df=dep_df,
            data_BC=data_BC,
            sampleinfo=sampleinfo,
            top_n_proteins=int(top_n_proteins),
        )
        mechanism_outputs["mechanism_top_protein_group_means"] = top_means_out
        _add_figure_record(figure_records, top_means_out, "Mechanism evidence", "Top differential proteins summarized by group mean abundance and detection rate.", differential_file_paths + [protein_quant_combat_path, sampleinfo_path], visualize_params, "mechanism_top_protein_group_means")

        overlap_out = plot_mechanism_dep_overlap_upset(this_run_folder, dep_df=dep_df)
        mechanism_outputs["mechanism_dep_overlap_upset"] = overlap_out
        _add_figure_record(figure_records, overlap_out, "Mechanism evidence", "UpSet-like summary of shared and contrast-specific differential proteins.", differential_file_paths, visualize_params, "mechanism_dep_overlap_upset")

        contrast_outputs = {}
        for contrast in dep_df["contrast"].dropna().astype(str).drop_duplicates().tolist()[:max_contrast_figures]:
            contrast_out = plot_mechanism_contrast_evidence(
                this_run_folder,
                contrast=contrast,
                dep_df=dep_df,
                enrichment_df=enrichment_summary,
                data_BC=data_BC,
                sampleinfo=sampleinfo,
                top_n_proteins=max(6, min(int(max_labels), 20)),
                max_labels=int(max_labels),
            )
            contrast_outputs[str(contrast)] = contrast_out
            _add_figure_record(figure_records, contrast_out, "Mechanism evidence", f"Integrated evidence panel for contrast `{contrast}`.", differential_file_paths + enrichment_file_paths + [protein_quant_combat_path, sampleinfo_path], visualize_params, "mechanism_contrast_evidence")
        mechanism_outputs["mechanism_contrast_evidence"] = contrast_outputs

        enrichment_dotplot = plot_mechanism_enrichment_dotplot(
            this_run_folder,
            enrichment_df=enrichment_summary,
            top_n_terms=int(top_n_terms),
        )
        mechanism_outputs["mechanism_enrichment_dotplot"] = enrichment_dotplot
        _add_figure_record(figure_records, enrichment_dotplot, "Enrichment evidence", "Cross-contrast enrichment dotplot combining GO/KEGG/Reactome results.", enrichment_file_paths, visualize_params, "mechanism_enrichment_dotplot")

        pathway_heatmaps = plot_mechanism_pathway_gene_heatmaps(
            this_run_folder,
            enrichment_df=enrichment_summary,
            data_BC=data_BC,
            sampleinfo=sampleinfo,
            protein_gene_map=protein_gene_map,
            top_n_terms=max(3, min(int(top_n_terms), 8)),
            max_genes=max(12, min(int(top_n_proteins), 40)),
        )
        mechanism_outputs["mechanism_pathway_gene_heatmaps"] = pathway_heatmaps
        for heatmap in pathway_heatmaps:
            _add_figure_record(figure_records, heatmap, "Enrichment evidence", f"Pathway gene heatmap for `{heatmap.get('description', '')}`.", enrichment_file_paths + [protein_quant_combat_path, sampleinfo_path, protein_gene_map_path], visualize_params, "mechanism_pathway_gene_heatmap")

    # Step 05: dataset-specific recipe figures (original-equivalent + extension)
    # are appended after the generic figure set, before the catalog is written.
    recipe_summary = _render_dataset_figure_recipes(
        this_run_folder,
        data_BC=data_BC,
        sampleinfo=sampleinfo,
        protein_gene_map=protein_gene_map,
        figure_records=figure_records,
        visualize_params=visualize_params,
        figure_recipe_params=figure_recipe_params,
    )

    catalog = _write_figure_catalog(this_run_folder, figure_records, visualize_params)
    _set_active_vector_formats([])

    results["_summary"] = {
        "heatmap": heatmap_out,
        "pca": pca_out,
        "umap": umap_out,
        "key_protein_overview": key_protein_out,
        "differential_summary_barplot": differential_summary_barplot_out,
        "protein_contrast_bubble": protein_contrast_bubble_out,
        "mechanism_outputs": mechanism_outputs,
        "figure_recipes": recipe_summary,
        "figure_manifest": catalog["figure_manifest"],
        "figure_index": catalog["figure_index"],
        "n_figures": int(len(figure_records)),
        "n_enrichment_bubble_plots": int(len(enrichment_bubbles)),
        "n_volcano_plots": len(results)
    }
    return results




def _legacy_plot_volcano(
        this_run_folder: str,
        df: pd.DataFrame,
        contrast: str,
        out_file: str = "volcano_plot.png",
        logfc_cut: float = 1.0,
        pval_cut: float = 0.05,
        top_n: int = 40
    ):

    file_path = _build_output_path(this_run_folder, out_file)

    dep = df.copy()
    dep = dep[dep["contrast"] == contrast].copy()

    dep["gene_label"] = dep["PG.Genes"].astype(str).str.split(";").str[0]
    dep["neglog10p"] = -np.log10(dep["adj.P.Val"] + 1e-10)

    dep["sig"] = "No change"
    dep.loc[(dep["logFC"] > logfc_cut) & (dep["adj.P.Val"] < pval_cut), "sig"] = "Up"
    dep.loc[(dep["logFC"] < -logfc_cut) & (dep["adj.P.Val"] < pval_cut), "sig"] = "Down"

    # 综合评分（更容易选出真正显著的点）
    dep["score"] = abs(dep["logFC"]) * dep["neglog10p"]

    up_top = (
        dep[dep["sig"] == "Up"]
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    down_top = (
        dep[dep["sig"] == "Down"]
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    colors = {
        "Up": "#ff3355",
        "Down": "#339dff",
        "No change": "darkgrey"
    }

    plt.figure(figsize=(6, 6), dpi=300)

    plt.scatter(
        dep["logFC"],
        dep["neglog10p"],
        c=dep["sig"].map(colors),
        s=18,
        alpha=0.6
    )

    plt.axvline(logfc_cut, linestyle="--", linewidth=1)
    plt.axvline(-logfc_cut, linestyle="--", linewidth=1)
    plt.axhline(-np.log10(pval_cut), linestyle="--", linewidth=1)

    plt.xlim(dep["logFC"].min(), dep["logFC"].max())
    plt.ylim(0, dep["neglog10p"].max() * 1.05)

    texts = []

    # 左侧标签
    for _, row in down_top.iterrows():
        texts.append(
            plt.text(
                row["logFC"],
                row["neglog10p"],
                row["gene_label"],
                fontsize=8,
                ha="right"
            )
        )

    # 右侧标签
    for _, row in up_top.iterrows():
        texts.append(
            plt.text(
                row["logFC"],
                row["neglog10p"],
                row["gene_label"],
                fontsize=8,
                ha="left"
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            lw=0.6,
            shrinkA=8,
            shrinkB=8
        ),
        expand_points=(1.5, 1.8),
        expand_text=(1.5, 1.8)
    )

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(adj.P.Val)")
    plt.title(contrast)

    plt.tight_layout()
    _save_figure(file_path, dpi=300)

    return {
        "file_path": file_path,
        "LogFC threshold": logfc_cut,
        "adj.P.Val threshold": pval_cut
    }

def _legacy_plot_heatmap(
        this_run_folder: str,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        protein_gene_map: pd.DataFrame,
        out_file: str = "heatmap.png",
        top_n: int = 100
):

    file_path = _build_output_path(this_run_folder, out_file)

    # =========================
    # 1 构建表达矩阵
    # =========================

    data_Anal = data_BC[sampleinfo["FileName"]].copy()
    data_Anal["PG"] = data_BC["PG.ProteinGroups"]

    data_Anal = (
        data_Anal
        .groupby("PG")
        .mean()
    )

    sample_groups = sampleinfo.set_index("FileName")["Cluster"]
    group_names = sample_groups.unique()

    # =========================
    # 2 差异分析 (pairwise)
    # =========================

    dep_list = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):

            g1 = group_names[i]
            g2 = group_names[j]

            s1 = sample_groups[sample_groups == g1].index
            s2 = sample_groups[sample_groups == g2].index

            mat1 = data_Anal[s1]
            mat2 = data_Anal[s2]

            logfc = mat1.mean(axis=1) - mat2.mean(axis=1)

            pvals = []
            for k in range(len(data_Anal)):
                _, p = ttest_ind(mat1.iloc[k], mat2.iloc[k], nan_policy="omit")
                pvals.append(p)

            res = pd.DataFrame({
                "protein": data_Anal.index,
                "contrast": f"{g1} vs {g2}",
                "logFC": logfc,
                "adj.P.Val": pvals
            })

            res["score"] = abs(res["logFC"]) * -np.log10(res["adj.P.Val"])

            dep_list.append(res)

    dep = pd.concat(dep_list)

    # =========================
    # 3 添加基因名
    # =========================

    dep = dep.merge(
        protein_gene_map,
        left_on="protein",
        right_on="PG.ProteinGroups",
        how="left"
    )

    # =========================
    # 4 选择 top 差异蛋白
    # =========================

    dep = dep.sort_values("score", ascending=False)
    proteins = dep["PG.ProteinGroups"].dropna().unique()[:top_n]

    data_dep = data_Anal.loc[proteins]

    # =========================
    # 5 Z-score 标准化
    # =========================

    data_dep = data_dep.sub(data_dep.mean(axis=1), axis=0)
    data_dep = data_dep.div(data_dep.std(axis=1), axis=0)

    data_dep = data_dep.fillna(0)

    # =========================
    # 6 样本分组颜色
    # =========================

    palette = sns.color_palette("Set2", len(group_names))
    group_color_map = dict(zip(group_names, palette))
    col_colors = sample_groups.map(group_color_map)

    # =========================
    # 7 画热图
    # =========================

    g = sns.clustermap(
        data_dep,
        cmap="vlag",
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=False,
        xticklabels=False,
        yticklabels=False,
        figsize=(7, 7)
    )

    # =========================
    # 8 添加图例
    # =========================

    for label in group_names:
        g.ax_col_dendrogram.bar(
            0, 0,
            color=group_color_map[label],
            label=label,
            linewidth=0
        )

    g.ax_col_dendrogram.legend(
        title="Group",
        loc="center",
        ncol=len(group_names)
    )

    # =========================
    # 9 标题标注对比类别
    # =========================

    contrasts = dep["contrast"].unique()
    # title_text = "Differential Protein Heatmap\nComparisons: " + ", ".join(contrasts)

    # plt.title(title_text, fontsize=12)

    _apply_nature_heatmap_frame(g.ax_heatmap)
    _save_figure(file_path, close_obj=g.fig, dpi=300)

    return {
        "file_path": file_path,
        "comparisons": list(contrasts)
    }


def plot_cluster(
        this_run_folder: str,
        df: pd.DataFrame,
        top_n: int = 20,
        out_file: str = "clustering.png"):
    """绘制层次聚类树状图"""

    out_file = os.path.basename(out_file)
    file_path = _build_output_path(this_run_folder, out_file)
    genes_topn = df.sort_values("P.Value").head(top_n)['PG.Genes'].values
    expr = {g: np.random.normal(df.loc[df['PG.Genes'] == g, 'AveExpr'].values[0], 0.5, size=6) for g in genes_topn}
    expr_df = pd.DataFrame(expr, index=[f"Sample{i}" for i in range(6)])

    link = linkage(expr_df.T, method='average', metric='correlation')
    plt.figure(figsize=(8, 6))
    dendrogram(link, labels=expr_df.columns, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Proteins")
    plt.tight_layout()

    # 保存结果
    _save_figure(file_path, dpi=300)
    return {
        "file_path": file_path,
    }


def draw_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width,
        height,
        angle=theta,
        **kwargs
    )

    ax.add_patch(ellipse)


def _legacy_plot_pca(
        this_run_folder: str,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        out_file: str = "pca_plot.png",
):

    file_path = _build_output_path(this_run_folder, out_file)

    # expression matrix
    data_Anal = data_BC[sampleinfo["FileName"]].copy()
    data_Anal["PG"] = data_BC["PG.ProteinGroups"]

    data_Anal = data_Anal.groupby("PG").mean()
    data_Anal = data_Anal.fillna(0)

    X = data_Anal.T

    # standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"],
        index=X.index
    )

    pca_df["Cluster"] = sampleinfo.set_index("FileName").loc[pca_df.index, "Cluster"]

    var_exp = pca.explained_variance_ratio_ * 100

    groups = pca_df["Cluster"].unique()
    palette = sns.color_palette("Set1", len(groups))
    color_map = dict(zip(groups, palette))

    fig, ax = plt.subplots(figsize=(6,6))

    texts = []

    for g in groups:

        sub = pca_df[pca_df["Cluster"] == g]

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=60,
            color=color_map[g],
            edgecolor="black",
            alpha=0.9,
            label=g
        )

        # 椭圆
        draw_confidence_ellipse(
            sub["PC1"],
            sub["PC2"],
            ax,
            facecolor=color_map[g],
            alpha=0.25,
            edgecolor=color_map[g],
            linewidth=2
        )

        # 组中心
        cx = sub["PC1"].mean()
        cy = sub["PC2"].mean()

        ax.scatter(
            cx,
            cy,
            s=150,
            marker="X",
            color=color_map[g],
            edgecolor="black",
            linewidth=1.5,
            zorder=5
        )

        # # 样本标签
        # for idx, row in sub.iterrows():

        #     texts.append(
        #         ax.text(
        #             row["PC1"],
        #             row["PC2"],
        #             idx,
        #             fontsize=8
        #         )
        #     )

    # adjust_text(
    #     texts,
    #     arrowprops=dict(
    #         arrowstyle="-",
    #         color="black",
    #         lw=0.6,
    #         shrinkA=5,
    #         shrinkB=5
    #     )
    # )

    ax.set_xlabel(f"PC1 {var_exp[0]:.1f}%")
    ax.set_ylabel(f"PC2 {var_exp[1]:.1f}%")

    ax.legend(title="Group")

    _apply_nature_axes(ax)

    plt.tight_layout()
    _save_figure(file_path, dpi=300)

    return {
        "file_path": file_path,
        "PC1_variance": round(var_exp[0],2),
        "PC2_variance": round(var_exp[1],2)
    }



def plot_umap(
        this_run_folder: str,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        out_file: str = "umap_plot.png",
    ):
    """Improved UMAP visualization for sample-level proteomics data."""

    file_path = _build_output_path(this_run_folder, out_file)

    if "FileName" not in sampleinfo.columns or "Cluster" not in sampleinfo.columns:
        raise ValueError("sampleinfo must contain 'FileName' and 'Cluster' columns")

    sampleinfo_plot = sampleinfo.copy()
    sample_cols = [col for col in sampleinfo_plot["FileName"] if col in data_BC.columns]
    if not sample_cols:
        raise ValueError("No sample columns from sampleinfo['FileName'] were found in data_BC")

    sampleinfo_plot = (
        sampleinfo_plot[sampleinfo_plot["FileName"].isin(sample_cols)]
        .drop_duplicates(subset=["FileName"])
        .set_index("FileName")
        .loc[sample_cols]
        .reset_index()
    )

    label_col = next(
        (
            candidate for candidate in ["Label", "Type", "Group", "Condition"]
            if candidate in sampleinfo_plot.columns
        ),
        "Cluster"
    )

    data_anal = data_BC[sample_cols].copy()
    data_anal["PG"] = data_BC["PG.ProteinGroups"]
    data_anal = data_anal.groupby("PG").mean()
    X = data_anal.T.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = X_scaled.shape[0]
    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.35,
        metric="cosine",
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame({
        "UMAP_1": embedding[:, 0],
        "UMAP_2": embedding[:, 1],
        "FileName": sampleinfo_plot["FileName"].values,
        "Cluster": _normalize_group_labels(sampleinfo_plot["Cluster"]).values,
        "Label": _normalize_group_labels(sampleinfo_plot[label_col]).values,
    })

    cluster_order = _group_label_order(umap_df["Cluster"])
    n_clusters = len(cluster_order)
    palette = _categorical_palette(cluster_order)

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(10.5, 8))

    sns.scatterplot(
        data=umap_df,
        x="UMAP_1",
        y="UMAP_2",
        hue="Cluster",
        hue_order=cluster_order,
        palette=palette,
        s=110,
        alpha=0.9,
        linewidth=0.8,
        edgecolor="white",
        ax=ax
    )

    cluster_centers = []
    for cluster in cluster_order:
        sub = umap_df[umap_df["Cluster"] == cluster]
        color = palette[cluster]
        if len(sub) >= 3:
            try:
                draw_confidence_ellipse(
                    sub["UMAP_1"].values,
                    sub["UMAP_2"].values,
                    ax=ax,
                    n_std=1.8,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.12,
                    linewidth=1.5
                )
            except Exception:
                pass

        center_x = sub["UMAP_1"].mean()
        center_y = sub["UMAP_2"].mean()
        cluster_centers.append((center_x, center_y, cluster))
        ax.scatter(
            center_x,
            center_y,
            s=260,
            marker="X",
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=5
        )

    texts = []
    for center_x, center_y, cluster in cluster_centers:
        texts.append(
            ax.text(
                center_x,
                center_y,
                cluster,
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=palette[cluster],
                    linewidth=1,
                    alpha=0.95
                ),
                zorder=6
            )
        )
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))

    ax.set_title(f"UMAP of Samples by {label_col}", fontsize=18, pad=14, weight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    _apply_nature_axes(ax)

    legend = ax.legend(
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True
    )
    if legend is not None:
        legend.get_title().set_fontsize(11)
        for text in legend.get_texts():
            text.set_fontsize(10)

    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    return {
        "file_path": file_path,
        "n_samples": int(n_samples),
        "n_clusters": int(n_clusters),
        "label_column": label_col
    }


def _prepare_sample_plot_data(
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
):
    if "FileName" not in sampleinfo.columns or "Cluster" not in sampleinfo.columns:
        raise ValueError("sampleinfo must contain 'FileName' and 'Cluster' columns")

    sampleinfo_plot = sampleinfo.copy()
    sample_cols = [col for col in sampleinfo_plot["FileName"] if col in data_BC.columns]
    if not sample_cols:
        raise ValueError("No sample columns from sampleinfo['FileName'] were found in data_BC")

    sampleinfo_plot = (
        sampleinfo_plot[sampleinfo_plot["FileName"].isin(sample_cols)]
        .drop_duplicates(subset=["FileName"])
        .set_index("FileName")
        .loc[sample_cols]
        .reset_index()
    )

    label_col = next(
        (
            candidate for candidate in ["Label", "Type", "Group", "Condition"]
            if candidate in sampleinfo_plot.columns
        ),
        "Cluster"
    )

    # Step 04: grouping columns must never keep raw NaN/blank cells that would
    # become literal "nan" tick labels after astype(str).
    sampleinfo_plot["Cluster"] = _normalize_group_labels(sampleinfo_plot["Cluster"])
    if label_col in sampleinfo_plot.columns:
        sampleinfo_plot[label_col] = _normalize_group_labels(sampleinfo_plot[label_col])

    data_anal = data_BC[sample_cols].copy()
    data_anal["PG"] = data_BC["PG.ProteinGroups"]
    data_anal = data_anal.groupby("PG").mean().fillna(0)

    return data_anal, sampleinfo_plot, sample_cols, label_col


def plot_key_protein_overview(
        this_run_folder: str,
        differential_file_paths: List[str],
        out_file: str = "key_protein_overview.png",
        pval_cut: float = 0.05,
        top_n_per_contrast: int = 2
    ):
    """Summarize significant proteins across multiple differential comparison files."""

    file_path = _build_output_path(this_run_folder, out_file)

    all_dep = []
    for file_path_in in differential_file_paths:
        if not _path_exists(file_path_in):
            continue
        dep = _read_csv_checked(file_path_in, description="differential protein file")
        if dep.empty:
            continue
        required_cols = {"contrast", "logFC", "adj.P.Val"}
        if not required_cols.issubset(dep.columns):
            continue

        dep = dep.copy()
        dep["source_file"] = os.path.basename(file_path_in)
        dep["adj.P.Val"] = pd.to_numeric(dep["adj.P.Val"], errors="coerce")
        dep["logFC"] = pd.to_numeric(dep["logFC"], errors="coerce")
        dep = dep.dropna(subset=["logFC", "adj.P.Val"])
        dep = dep[dep["adj.P.Val"] < pval_cut].copy()
        if dep.empty:
            continue

        gene_source = "PG.Genes" if "PG.Genes" in dep.columns else "protein"
        dep["protein_label"] = dep[gene_source].fillna(dep.get("protein", "")).astype(str).str.split(";").str[0]
        dep["neglog10p"] = -np.log10(dep["adj.P.Val"].clip(lower=1e-300))
        dep["score"] = dep["logFC"].abs() * dep["neglog10p"]
        dep["direction"] = np.where(dep["logFC"] >= 0, "Up", "Down")
        all_dep.append(dep)

    if all_dep:
        summary_df = pd.concat(all_dep, ignore_index=True)
    else:
        summary_df = pd.DataFrame(columns=["contrast", "logFC", "adj.P.Val", "neglog10p", "score", "protein_label", "direction"])

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(max(9.5, len(differential_file_paths) * 1.25), 8.2))

    if summary_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No proteins with adj.P.Val < {pval_cut}",
            ha="center",
            va="center",
            fontsize=15
        )
        ax.set_axis_off()
        plt.tight_layout()
        _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
        return {
            "file_path": file_path,
            "n_significant": 0,
            "n_contrasts": 0,
            "pval_cutoff": float(pval_cut),
            "top_n_per_contrast": int(top_n_per_contrast),
            "thresholds": {
                "adj_pval": float(pval_cut),
                "top_n_per_contrast": int(top_n_per_contrast),
            },
            "filter_note": f"Filtered by adj.P.Val < {pval_cut}; labeled top {top_n_per_contrast} proteins per contrast",
            "empty": True,
        }

    contrast_order = summary_df.groupby("contrast")["score"].max().sort_values(ascending=False).index.tolist()
    summary_df["contrast"] = pd.Categorical(summary_df["contrast"], categories=contrast_order, ordered=True)
    summary_df["contrast_num"] = summary_df["contrast"].cat.codes.astype(float)

    rng = np.random.default_rng(42)
    summary_df["x_jitter"] = summary_df["contrast_num"] + rng.uniform(-0.18, 0.18, size=len(summary_df))

    colors = {"Up": "#d1495b", "Down": "#2c7fb8"}
    for direction in ["Down", "Up"]:
        sub = summary_df[summary_df["direction"] == direction]
        ax.scatter(
            sub["x_jitter"],
            sub["logFC"],
            s=np.clip(sub["neglog10p"] * 18, 24, 190),
            c=colors[direction],
            alpha=0.72,
            edgecolors="white",
            linewidth=0.7,
            label=f"{direction} (n={len(sub)})"
        )

    contrast_centers = summary_df.groupby("contrast", observed=False)["logFC"].median()
    for idx, contrast in enumerate(contrast_order):
        if contrast in contrast_centers.index and pd.notna(contrast_centers.loc[contrast]):
            ax.hlines(
                contrast_centers.loc[contrast],
                idx - 0.22,
                idx + 0.22,
                colors="#444444",
                linewidth=2,
                zorder=3
            )

    label_df = (
        summary_df.sort_values(["contrast", "score"], ascending=[True, False])
        .groupby("contrast", observed=False)
        .head(top_n_per_contrast)
        .copy()
    )
    texts = []
    for _, row in label_df.iterrows():
        texts.append(
            ax.text(
                row["x_jitter"],
                row["logFC"],
                row["protein_label"],
                fontsize=7.2,
                ha="left" if row["logFC"] >= 0 else "right",
                va="bottom",
                color="#222222"
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5),
            expand_points=(1.6, 1.7),
            expand_text=(1.5, 1.6),
            force_text=(0.5, 0.8),
            lim=300
        )

    ax.axhline(0, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(contrast_order)))
    ax.set_xticklabels(contrast_order, rotation=35, ha="right")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("log2 Fold Change")
    ax.set_title("Key Significant Proteins Across Differential Comparisons", fontsize=17, pad=12, weight="bold")
    _apply_nature_axes(ax)
    ax.legend(
        title="Direction",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0, 0.84, 0.97))
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    filter_note = f"Filtered by adj.P.Val < {pval_cut}; labeled top {top_n_per_contrast} proteins per contrast"
    return {
        "file_path": file_path,
        "n_significant": int(len(summary_df)),
        "n_contrasts": int(summary_df["contrast"].nunique()),
        "pval_cutoff": float(pval_cut),
        "top_n_per_contrast": int(top_n_per_contrast),
        "thresholds": {
            "adj_pval": float(pval_cut),
            "top_n_per_contrast": int(top_n_per_contrast),
        },
        "filter_note": filter_note,
    }


def plot_differential_summary_barplot(
        this_run_folder: str,
        differential_file_paths: List[str],
        out_file: str = "differential_summary_barplot.png",
        pval_cut: float = 0.05,
        logfc_cut: float = 1.0
    ):
    """Summarize the number of up/down regulated proteins for each contrast."""

    file_path = _build_output_path(this_run_folder, out_file)

    summary_rows = []
    for file_path_in in differential_file_paths:
        if not _path_exists(file_path_in):
            continue
        dep = _read_csv_checked(file_path_in, description="differential protein file")
        if dep.empty or not {"contrast", "logFC", "adj.P.Val"}.issubset(dep.columns):
            continue

        dep = dep.copy()
        dep["adj.P.Val"] = pd.to_numeric(dep["adj.P.Val"], errors="coerce")
        dep["logFC"] = pd.to_numeric(dep["logFC"], errors="coerce")
        dep = dep.dropna(subset=["logFC", "adj.P.Val"])
        if dep.empty:
            continue

        contrast = str(dep["contrast"].iloc[0])
        n_up = int(((dep["adj.P.Val"] < pval_cut) & (dep["logFC"] >= logfc_cut)).sum())
        n_down = int(((dep["adj.P.Val"] < pval_cut) & (dep["logFC"] <= -logfc_cut)).sum())
        summary_rows.append({"contrast": contrast, "direction": "Up", "count": n_up})
        summary_rows.append({"contrast": contrast, "direction": "Down", "count": -n_down})

    summary_df = pd.DataFrame(summary_rows)

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(max(8.5, len(differential_file_paths) * 1.15), 7.2))

    if summary_df.empty:
        ax.text(0.5, 0.5, "No differential summary data available", ha="center", va="center", fontsize=15)
        ax.set_axis_off()
        plt.tight_layout()
        _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
        return {
            "file_path": file_path,
            "n_contrasts": 0,
            "pval_cutoff": float(pval_cut),
            "logfc_cutoff": float(logfc_cut),
            "n_sig": 0,
            "n_up": 0,
            "n_down": 0,
            "thresholds": {"adj_pval": float(pval_cut), "abs_logfc": float(logfc_cut)},
            "filter_note": f"Thresholds: adj.P.Val < {pval_cut}, |logFC| >= {logfc_cut}",
            "empty": True,
        }

    contrast_order = (
        summary_df.assign(abs_count=lambda x: x["count"].abs())
        .groupby("contrast")["abs_count"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    plot_df = summary_df.copy()
    plot_df["contrast"] = pd.Categorical(plot_df["contrast"], categories=contrast_order, ordered=True)
    plot_df = plot_df.sort_values(["contrast", "direction"])

    colors = {"Up": "#d1495b", "Down": "#2c7fb8"}
    for direction in ["Up", "Down"]:
        sub = plot_df[plot_df["direction"] == direction]
        ax.bar(sub["contrast"].astype(str), sub["count"], color=colors[direction], alpha=0.88, label=direction, width=0.72)

    max_abs = max(1, plot_df["count"].abs().max())
    ax.axhline(0, color="#555555", linewidth=1.1)
    ax.set_ylim(-max_abs * 1.18, max_abs * 1.18)
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Number of significant proteins")
    ax.set_title("Differential Protein Counts by Contrast", fontsize=17, pad=12, weight="bold")
    ax.set_xticklabels(contrast_order, rotation=35, ha="right")
    _apply_nature_axes(ax)
    ax.legend(title="Direction", frameon=True)

    for _, row in plot_df.iterrows():
        y = row["count"]
        va = "bottom" if y >= 0 else "top"
        offset = max_abs * 0.03 if y >= 0 else -max_abs * 0.03
        ax.text(row["contrast"], y + offset, str(abs(int(y))), ha="center", va=va, fontsize=9, color="#222222")

    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    up_counts = plot_df.loc[plot_df["direction"] == "Up", "count"].clip(lower=0)
    down_counts = (-plot_df.loc[plot_df["direction"] == "Down", "count"]).clip(lower=0)
    n_up_total = int(up_counts.sum())
    n_down_total = int(down_counts.sum())
    return {
        "file_path": file_path,
        "n_contrasts": int(len(contrast_order)),
        "pval_cutoff": float(pval_cut),
        "logfc_cutoff": float(logfc_cut),
        "n_sig": n_up_total + n_down_total,
        "n_up": n_up_total,
        "n_down": n_down_total,
        "thresholds": {"adj_pval": float(pval_cut), "abs_logfc": float(logfc_cut)},
        "filter_note": f"Thresholds: adj.P.Val < {pval_cut}, |logFC| >= {logfc_cut}",
    }


def plot_protein_contrast_bubble(
        this_run_folder: str,
        differential_file_paths: List[str],
        out_file: str = "protein_contrast_bubble.png",
        pval_cut: float = 0.05,
        top_n_per_contrast: int = 8,
        max_proteins: int = 32
    ):
    """Show key proteins across contrasts as a bubble chart."""

    file_path = _build_output_path(this_run_folder, out_file)

    all_dep = []
    for file_path_in in differential_file_paths:
        if not _path_exists(file_path_in):
            continue
        dep = _read_csv_checked(file_path_in, description="differential protein file")
        if dep.empty or not {"contrast", "logFC", "adj.P.Val"}.issubset(dep.columns):
            continue

        dep = dep.copy()
        dep["adj.P.Val"] = pd.to_numeric(dep["adj.P.Val"], errors="coerce")
        dep["logFC"] = pd.to_numeric(dep["logFC"], errors="coerce")
        dep = dep.dropna(subset=["logFC", "adj.P.Val"])
        dep = dep[dep["adj.P.Val"] < pval_cut].copy()
        if dep.empty:
            continue

        gene_source = "PG.Genes" if "PG.Genes" in dep.columns else "protein"
        dep["protein_label"] = dep[gene_source].fillna(dep.get("protein", "")).astype(str).str.split(";").str[0]
        dep["neglog10p"] = -np.log10(dep["adj.P.Val"].clip(lower=1e-300))
        dep["score"] = dep["logFC"].abs() * dep["neglog10p"]
        all_dep.append(dep)

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(11, 8.8))

    if all_dep:
        bubble_df = pd.concat(all_dep, ignore_index=True)
    else:
        bubble_df = pd.DataFrame(columns=["contrast", "protein_label", "logFC", "neglog10p", "score"])

    if bubble_df.empty:
        ax.text(0.5, 0.5, f"No proteins with adj.P.Val < {pval_cut}", ha="center", va="center", fontsize=15)
        ax.set_axis_off()
        plt.tight_layout()
        _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
        return {
            "file_path": file_path,
            "n_proteins": 0,
            "n_contrasts": 0,
            "n_sig": 0,
            "pval_cutoff": float(pval_cut),
            "top_n_per_contrast": int(top_n_per_contrast),
            "max_proteins": int(max_proteins),
            "thresholds": {
                "adj_pval": float(pval_cut),
                "top_n_per_contrast": int(top_n_per_contrast),
                "max_proteins": int(max_proteins),
            },
            "filter_note": f"Filtered by adj.P.Val < {pval_cut}; top {top_n_per_contrast} proteins per contrast",
            "empty": True,
        }

    bubble_df = (
        bubble_df.sort_values(["contrast", "score"], ascending=[True, False])
        .groupby("contrast", observed=False)
        .head(top_n_per_contrast)
        .copy()
    )

    protein_order = (
        bubble_df.groupby("protein_label")["score"]
        .max()
        .sort_values(ascending=False)
        .head(max_proteins)
        .index
        .tolist()
    )
    bubble_df = bubble_df[bubble_df["protein_label"].isin(protein_order)].copy()
    contrast_order = (
        bubble_df.groupby("contrast")["score"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    x_map = {contrast: i for i, contrast in enumerate(contrast_order)}
    y_map = {protein: i for i, protein in enumerate(reversed(protein_order))}
    bubble_df["x"] = bubble_df["contrast"].map(x_map)
    bubble_df["y"] = bubble_df["protein_label"].map(y_map)

    scatter = ax.scatter(
        bubble_df["x"],
        bubble_df["y"],
        s=np.clip(bubble_df["neglog10p"] * 42, 35, 300),
        c=bubble_df["logFC"],
        cmap="coolwarm",
        vmin=-bubble_df["logFC"].abs().max(),
        vmax=bubble_df["logFC"].abs().max(),
        alpha=0.88,
        edgecolors="white",
        linewidth=0.7
    )

    ax.set_xticks(range(len(contrast_order)))
    ax.set_xticklabels(contrast_order, rotation=35, ha="right")
    ax.set_yticks(range(len(y_map)))
    ax.set_yticklabels([protein for protein, _ in sorted(y_map.items(), key=lambda x: x[1])])
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Key protein")
    ax.set_title("Key Proteins Across Contrasts", fontsize=17, pad=12, weight="bold")
    _apply_nature_axes(ax)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("log2 Fold Change")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    filter_note = f"Filtered by adj.P.Val < {pval_cut}; top {top_n_per_contrast} proteins per contrast"
    return {
        "file_path": file_path,
        "n_proteins": int(len(protein_order)),
        "n_contrasts": int(len(contrast_order)),
        "pval_cutoff": float(pval_cut),
        "n_sig": int(sum(len(dep) for dep in all_dep)),
        "top_n_per_contrast": int(top_n_per_contrast),
        "max_proteins": int(max_proteins),
        "thresholds": {
            "adj_pval": float(pval_cut),
            "top_n_per_contrast": int(top_n_per_contrast),
            "max_proteins": int(max_proteins),
        },
        "filter_note": filter_note,
    }


def plot_volcano(
        this_run_folder: str,
        df: pd.DataFrame,
        contrast: str,
        out_file: str = "volcano_plot.png",
        logfc_cut: float = 1.0,
        pval_cut: float = 0.05,
        top_n: int = 1
    ):

    file_path = _build_output_path(this_run_folder, out_file)

    dep = df.copy()
    if "contrast" in dep.columns:
        dep = dep[dep["contrast"] == contrast].copy()
    if dep.empty:
        raise ValueError(f"No data found for contrast: {contrast}")

    dep["PG.Genes"] = dep["PG.Genes"].fillna(dep.get("protein", ""))
    dep["gene_label"] = dep["PG.Genes"].astype(str).str.split(";").str[0]
    dep["adj.P.Val"] = pd.to_numeric(dep["adj.P.Val"], errors="coerce").fillna(1.0).clip(lower=1e-300)
    dep["logFC"] = pd.to_numeric(dep["logFC"], errors="coerce")
    dep = dep.dropna(subset=["logFC"]).copy()
    dep["neglog10p"] = -np.log10(dep["adj.P.Val"])

    dep["sig"] = "Not significant"
    dep.loc[(dep["logFC"] >= logfc_cut) & (dep["adj.P.Val"] < pval_cut), "sig"] = "Upregulated"
    dep.loc[(dep["logFC"] <= -logfc_cut) & (dep["adj.P.Val"] < pval_cut), "sig"] = "Downregulated"
    dep["score"] = dep["neglog10p"] * dep["logFC"].abs()

    up_top = dep[dep["sig"] == "Upregulated"].sort_values("score", ascending=False).head(top_n)
    down_top = dep[dep["sig"] == "Downregulated"].sort_values("score", ascending=False).head(top_n)

    colors = {
        "Upregulated": "#d1495b",
        "Downregulated": "#2c7fb8",
        "Not significant": "#b8b8b8"
    }

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(9.2, 7.4))

    for category in ["Not significant", "Downregulated", "Upregulated"]:
        sub = dep[dep["sig"] == category]
        ax.scatter(
            sub["logFC"],
            sub["neglog10p"],
            s=22 if category == "Not significant" else 34,
            c=colors[category],
            alpha=0.55 if category == "Not significant" else 0.82,
            edgecolors="none",
            label=f"{category} (n={len(sub)})",
            rasterized=True
        )

    ax.axvline(logfc_cut, linestyle="--", linewidth=1.2, color="#666666")
    ax.axvline(-logfc_cut, linestyle="--", linewidth=1.2, color="#666666")
    ax.axhline(-np.log10(pval_cut), linestyle="--", linewidth=1.2, color="#666666")

    x_abs = max(abs(dep["logFC"].min()), abs(dep["logFC"].max()))
    x_lim = max(2.5, x_abs * 1.2)
    y_lim = max(3, dep["neglog10p"].replace([np.inf, -np.inf], np.nan).max() * 1.15)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, y_lim)

    texts = []
    for _, row in pd.concat([down_top, up_top], axis=0).drop_duplicates(subset=["gene_label"]).iterrows():
        ha = "right" if row["logFC"] < 0 else "left"
        texts.append(
            ax.text(
                row["logFC"],
                row["neglog10p"],
                row["gene_label"],
                fontsize=7.5,
                ha=ha,
                va="bottom",
                color="#222222"
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, shrinkA=4, shrinkB=2),
            expand_points=(1.65, 1.75),
            expand_text=(1.55, 1.65),
            force_text=(0.45, 0.7),
            lim=300
        )

    n_up = int((dep["sig"] == "Upregulated").sum())
    n_down = int((dep["sig"] == "Downregulated").sum())
    ax.set_title(f"Volcano Plot: {contrast}", fontsize=17, pad=12, weight="bold")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 Adjusted P-value")
    _apply_nature_axes(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
        title="Category"
    )

    plt.tight_layout(rect=(0, 0, 0.80, 1))
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    return {
        "file_path": file_path,
        "LogFC threshold": logfc_cut,
        "adj.P.Val threshold": pval_cut,
        "n_up": n_up,
        "n_down": n_down,
        "n_sig": int(n_up + n_down),
        "thresholds": {
            "adj_pval": float(pval_cut),
            "abs_logfc": float(logfc_cut),
            "top_n_labels": int(top_n),
        },
        "filter_note": f"Thresholds: adj.P.Val < {pval_cut}, |logFC| >= {logfc_cut}; Up: {n_up}, Down: {n_down}",
    }


def plot_heatmap(
        this_run_folder: str,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        protein_gene_map: pd.DataFrame,
        out_file: str = "heatmap.png",
        top_n: int = 80
):

    file_path = _build_output_path(this_run_folder, out_file)

    data_anal, sampleinfo_plot, sample_cols, label_col = _prepare_sample_plot_data(data_BC, sampleinfo)
    sample_groups = _normalize_group_labels(sampleinfo_plot.set_index("FileName")["Cluster"])
    group_names = sample_groups.unique().tolist()

    dep_list = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = group_names[i]
            g2 = group_names[j]

            s1 = sample_groups[sample_groups == g1].index.tolist()
            s2 = sample_groups[sample_groups == g2].index.tolist()
            if len(s1) < 2 or len(s2) < 2:
                continue

            mat1 = data_anal[s1]
            mat2 = data_anal[s2]
            logfc = mat1.mean(axis=1) - mat2.mean(axis=1)
            pvals = [
                ttest_ind(mat1.iloc[k], mat2.iloc[k], nan_policy="omit", equal_var=False)[1]
                for k in range(len(data_anal))
            ]

            res = pd.DataFrame({
                "protein": data_anal.index,
                "contrast": f"{g1} vs {g2}",
                "logFC": logfc,
                "adj.P.Val": np.asarray(pvals).clip(min=1e-300)
            })
            res["score"] = res["logFC"].abs() * -np.log10(res["adj.P.Val"])
            dep_list.append(res)

    if dep_list:
        dep = pd.concat(dep_list, ignore_index=True)
    else:
        dep = pd.DataFrame({
            "protein": data_anal.index,
            "contrast": "all samples",
            "score": data_anal.var(axis=1).values
        })

    gene_map = protein_gene_map.copy()
    gene_map["gene_label"] = gene_map["PG.Genes"].fillna("").astype(str).str.split(";").str[0]
    dep = dep.merge(
        gene_map[["PG.ProteinGroups", "gene_label"]],
        left_on="protein",
        right_on="PG.ProteinGroups",
        how="left"
    )
    dep["feature_label"] = dep["gene_label"].replace("", np.nan).fillna(dep["protein"].astype(str))

    top_features = dep.sort_values("score", ascending=False)["protein"].dropna().drop_duplicates().head(top_n).tolist()
    if not top_features:
        top_features = data_anal.var(axis=1).sort_values(ascending=False).head(top_n).index.tolist()

    data_dep = data_anal.loc[top_features, sample_cols].copy()
    row_labels = (
        dep.drop_duplicates(subset=["protein"])
        .set_index("protein")
        .reindex(top_features)["feature_label"]
    )
    row_labels = row_labels.where(row_labels.notna(), pd.Series(top_features, index=top_features, dtype="object").astype(str))
    data_dep.index = row_labels

    row_std = data_dep.std(axis=1).replace(0, np.nan)
    data_dep = data_dep.sub(data_dep.mean(axis=1), axis=0).div(row_std, axis=0).fillna(0)
    data_dep = data_dep.clip(-2.5, 2.5)

    cluster_order = _group_label_order(group_names)
    sample_order = sampleinfo_plot.copy()
    sample_order["Cluster"] = _normalize_group_labels(sample_order["Cluster"])
    sample_order["Cluster"] = pd.Categorical(sample_order["Cluster"], categories=cluster_order, ordered=True)
    if label_col in sample_order.columns:
        sample_order[label_col] = _normalize_group_labels(sample_order[label_col])
        sample_order = sample_order.sort_values(["Cluster", label_col, "FileName"])
    else:
        sample_order = sample_order.sort_values(["Cluster", "FileName"])
    ordered_samples = sample_order["FileName"].tolist()
    data_dep = data_dep[ordered_samples]

    cluster_color_map = _categorical_palette(cluster_order)
    col_colors = sample_order["Cluster"].astype(str).map(cluster_color_map).tolist()

    sns.set_theme(style="white", context="talk")
    show_yticks = len(data_dep.index) <= 60
    g = sns.clustermap(
        data_dep,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=False,
        xticklabels=False,
        yticklabels=show_yticks,
        linewidths=0,
        rasterized=True,
        figsize=(10.5, 12),
        cbar_kws={"label": "Row Z-score"},
        dendrogram_ratio=(0.16, 0.08),
        colors_ratio=(0.03, 0.03)
    )

    g.fig.suptitle("Differential Protein Heatmap", y=1.02, fontsize=18, weight="bold")
    g.ax_heatmap.set_xlabel(f"Samples grouped by {label_col}", fontsize=11)
    g.ax_heatmap.set_ylabel("Proteins", fontsize=11)
    _apply_nature_heatmap_frame(g.ax_heatmap)
    if show_yticks:
        g.ax_heatmap.tick_params(axis="y", labelsize=8)

    for label in cluster_order:
        g.ax_col_dendrogram.bar(0, 0, color=cluster_color_map[label], label=label, linewidth=0)
    g.ax_col_dendrogram.legend(
        title="Cluster",
        loc="center",
        ncol=min(len(cluster_order), 4),
        frameon=True,
        fontsize=9
    )

    _save_figure(file_path, fig=g.fig, dpi=300, bbox_inches="tight")

    return {
        "file_path": file_path,
        "comparisons": sorted(dep["contrast"].astype(str).unique().tolist()),
        "n_features": int(len(top_features)),
        "label_column": label_col
    }


def plot_pca(
        this_run_folder: str,
        data_BC: pd.DataFrame,
        sampleinfo: pd.DataFrame,
        out_file: str = "pca_plot.png",
):

    file_path = _build_output_path(this_run_folder, out_file)

    data_anal, sampleinfo_plot, sample_cols, label_col = _prepare_sample_plot_data(data_BC, sampleinfo)
    X = data_anal[sample_cols].T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"],
        index=sample_cols
    ).reset_index(names="FileName")
    pca_df = pca_df.merge(sampleinfo_plot, on="FileName", how="left")
    pca_df["Cluster"] = _normalize_group_labels(pca_df["Cluster"])

    cluster_order = _group_label_order(pca_df["Cluster"])
    color_map = _categorical_palette(cluster_order)

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(8.4, 7.4))

    texts = []
    for cluster in cluster_order:
        sub = pca_df[pca_df["Cluster"] == cluster]
        color = color_map[cluster]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=95,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.9,
            label=cluster
        )

        if len(sub) >= 3:
            try:
                draw_confidence_ellipse(
                    sub["PC1"].values,
                    sub["PC2"].values,
                    ax,
                    n_std=1.8,
                    facecolor=color,
                    alpha=0.15,
                    edgecolor=color,
                    linewidth=1.5
                )
            except Exception:
                pass

        cx = sub["PC1"].mean()
        cy = sub["PC2"].mean()
        ax.scatter(
            cx,
            cy,
            s=220,
            marker="X",
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=5
        )
        texts.append(
            ax.text(
                cx,
                cy,
                str(cluster),
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.95),
                zorder=6
            )
        )

    if len(pca_df) <= 20:
        for _, row in pca_df.iterrows():
            texts.append(
                ax.text(
                    row["PC1"],
                    row["PC2"],
                    str(row[label_col]),
                    fontsize=8,
                    color="#333333"
                )
            )
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))

    ax.set_title(f"PCA of Samples by {label_col}", fontsize=17, pad=12, weight="bold")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    _apply_nature_axes(ax)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)

    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    return {
        "file_path": file_path,
        "PC1_variance": round(var_exp[0], 2),
        "PC2_variance": round(var_exp[1], 2),
        "label_column": label_col,
        "n_samples": int(len(pca_df))
    }


def plot_enrichment_bubble(
        this_run_folder: str,
        enrich_df: pd.DataFrame,
        out_file: str,
        top_n: int = 15
    ):
    """
    绘制 GO / KEGG 富集气泡图
    """

    file_path = _build_output_path(this_run_folder, out_file)

    df = enrich_df.copy()
    if df.empty or "Description" not in df.columns:
        return _placeholder_plot(this_run_folder, out_file, "No enrichment rows", "No enrichment terms were available for this plot.")

    if "p.adjust" in df.columns:
        df["p.adjust"] = pd.to_numeric(df["p.adjust"], errors="coerce").fillna(1.0).clip(lower=1e-300)
    else:
        df["p.adjust"] = 1.0
    if "Count" in df.columns:
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(1)
    else:
        df["Count"] = 1
    if "RichFactor" not in df.columns:
        df["RichFactor"] = df.get("GeneRatio", "").apply(_parse_ratio)
    df["RichFactor"] = pd.to_numeric(df["RichFactor"], errors="coerce").fillna(0)
    if "Cluster" not in df.columns:
        df["Cluster"] = "Enrichment"
    df["score"] = -np.log10(df["p.adjust"])

    df = (
        df.sort_values("p.adjust")
        .groupby("Cluster")
        .head(top_n)
    )
    if df.empty:
        return _placeholder_plot(this_run_folder, out_file, "No enrichment rows", "No enrichment terms passed filtering.")

    df = df.sort_values("RichFactor")

    _set_visual_theme("notebook")

    fig_h = max(5.5, min(12, 0.32 * len(df) + 2))
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    scatter = ax.scatter(
        df["RichFactor"],
        [_wrap_label(x, 42) for x in df["Description"]],
        s=df["Count"] * 20,
        c=df["score"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black"
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("-log10(adjusted p-value)")

    ax.set_xlabel("Rich Factor")
    ax.set_ylabel("")
    ax.set_title("Enrichment Bubble Plot")
    _apply_nature_axes(ax)

    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")

    return {
        "file_path": file_path
    }


def plot_enrichment_bar(
        this_run_folder: str,
        enrich_df: pd.DataFrame,
        out_file: str,
        top_n: int = 10
    ):
    """
    绘制 GO / KEGG 富集条形图
    """

    file_path = _build_output_path(this_run_folder, out_file)

    df = enrich_df.copy()

    # 计算 score
    df["score"] = -np.log10(df["p.adjust"])

    # 每个cluster选top_n
    df = (
        df.sort_values("p.adjust")
        .groupby("Cluster")
        .head(top_n)
    )

    # 为了绘图顺序
    df = df.sort_values("score")

    _set_visual_theme("talk")

    plt.figure(figsize=(7, 5))

    ax = sns.barplot(
        data=df,
        y="Description",
        x="score",
        hue="Cluster",
        dodge=False,
        palette="dark"
    )

    ax.set_xlabel("-log10(adjusted p-value)", fontsize=12)
    ax.set_ylabel("")

    plt.legend(title="Cluster")

    sns.despine()

    plt.tight_layout()
    _save_figure(file_path, dpi=300)

    return {
        "file_path": file_path
    }


def _load_enrichment_summary(enrichment_file_paths: List[str], top_n_terms: int = 15) -> pd.DataFrame:
    rows = []
    for file_path in enrichment_file_paths:
        if not os.path.exists(file_path):
            continue
        df = _read_csv_checked(file_path, description="enrichment result file")
        if df.empty or "Description" not in df.columns:
            continue
        meta = _parse_enrichment_file_name(os.path.basename(file_path))
        df = df.copy()
        if "p.adjust" in df.columns:
            df["p.adjust"] = pd.to_numeric(df["p.adjust"], errors="coerce").fillna(1.0).clip(lower=1e-300)
        else:
            df["p.adjust"] = 1.0
        if "Count" in df.columns:
            df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(1)
        else:
            df["Count"] = 1
        if "RichFactor" not in df.columns:
            df["RichFactor"] = df.get("GeneRatio", "").apply(_parse_ratio)
        df["RichFactor"] = pd.to_numeric(df["RichFactor"], errors="coerce").fillna(0)
        df["score"] = -np.log10(df["p.adjust"])
        df["namespace"] = meta["namespace"]
        df["contrast"] = df.get("Cluster", meta["contrast"])
        df["contrast"] = df["contrast"].fillna(meta["contrast"]).astype(str)
        df["direction"] = meta["direction"]
        df["source_file"] = os.path.basename(file_path)
        rows.append(df.sort_values("p.adjust").head(top_n_terms))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _qc_overview_sample_frame(data_BC: pd.DataFrame, sampleinfo: pd.DataFrame):
    """Shared data preparation for the QC sample-overview figure (unit-testable).

    Returns ``(matrix, sample_meta, cluster_order, color_map, label_col)`` with
    grouping labels already normalized (missing values → 'Unannotated').
    """
    data_anal, sampleinfo_plot, sample_cols, label_col = _prepare_sample_plot_data(data_BC, sampleinfo)
    matrix = data_anal[sample_cols].apply(pd.to_numeric, errors="coerce")
    sample_meta = sampleinfo_plot.set_index("FileName").loc[sample_cols].reset_index()
    sample_meta["Cluster"] = _normalize_group_labels(sample_meta["Cluster"])
    sample_meta["detected_proteins"] = matrix.replace(0, np.nan).notna().sum(axis=0).values
    sample_meta["missing_rate"] = 1 - sample_meta["detected_proteins"] / max(1, len(matrix))
    cluster_order = _group_label_order(sample_meta["Cluster"])
    color_map = _categorical_palette(cluster_order)
    sample_meta["color"] = sample_meta["Cluster"].map(color_map)
    sample_meta = sample_meta.sort_values(["Cluster", "FileName"]).reset_index(drop=True)
    return matrix, sample_meta, cluster_order, color_map, label_col


def _draw_qc_detected_panel(ax, sample_meta: pd.DataFrame, large_n: bool) -> None:
    ax.set_title("Detected Proteins Per Sample")
    ax.set_ylabel("Detected proteins")
    if large_n:
        # Per-sample bars degenerate at large n: draw a sorted profile line with
        # Q1/median/Q3 reference lines and a quartile band instead.
        values = sample_meta["detected_proteins"].to_numpy(dtype=float)
        sorted_vals = np.sort(values)[::-1]
        ax.plot(np.arange(1, len(sorted_vals) + 1), sorted_vals, color=VISUAL_PALETTE[0], linewidth=1.4)
        q1, med, q3 = np.percentile(sorted_vals, [25, 50, 75])
        ax.fill_between([1, len(sorted_vals)], q1, q3, color=VISUAL_PALETTE[0], alpha=0.10, linewidth=0)
        ax.axhline(med, color=VISUAL_PALETTE[0], linestyle="-", linewidth=1.0)
        ax.axhline(q1, color=VISUAL_PALETTE[0], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(q3, color=VISUAL_PALETTE[0], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0, len(sorted_vals) * 1.01)
        ax.set_xlabel("Samples (sorted by detected proteins)")
        ax.text(0.99, 0.03, f"Q1={q1:.0f}   median={med:.0f}   Q3={q3:.0f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#333333")
    else:
        ax.bar(range(len(sample_meta)), sample_meta["detected_proteins"], color=sample_meta["color"], width=0.85)
        ax.set_xticks([])
    _apply_nature_axes(ax)


def _draw_qc_missing_panel(ax, sample_meta: pd.DataFrame, cluster_order: List[str],
                           color_map: Dict[str, tuple], large_n: bool) -> None:
    sns.boxplot(data=sample_meta, x="Cluster", y="missing_rate", order=cluster_order,
                palette=color_map, ax=ax, fliersize=0)
    if large_n:
        # Deterministic per-cluster subsample keeps the scatter readable at large n.
        budget = max(40, int(LARGE_N_PANEL_THRESHOLD / max(1, len(cluster_order))))
        parts = [
            group.sample(n=min(len(group), budget), random_state=12345)
            for _, group in sample_meta.groupby("Cluster", observed=False)
        ]
        strip_df = pd.concat(parts, ignore_index=True)
        sns.stripplot(data=strip_df, x="Cluster", y="missing_rate", order=cluster_order,
                      color="#222222", size=2.2, alpha=0.35, jitter=0.22, ax=ax)
    else:
        sns.stripplot(data=sample_meta, x="Cluster", y="missing_rate", order=cluster_order,
                      color="#222222", size=3, alpha=0.55, ax=ax)
    ax.set_title("Missing Rate By Cluster")
    ax.set_ylabel("Missing rate")
    ax.set_xlabel("")
    _wrap_axis_ticklabels(ax, "x", width=16, rotation=25)
    _apply_nature_axes(ax)


def _draw_qc_counts_panel(ax, sample_meta: pd.DataFrame, cluster_order: List[str],
                          color_map: Dict[str, tuple]) -> None:
    counts = sample_meta["Cluster"].value_counts().reindex(cluster_order).fillna(0)
    ax.bar(counts.index, counts.values, color=[color_map[x] for x in counts.index])
    ax.set_title("Sample Count By Cluster")
    ax.set_ylabel("Samples")
    ax.set_xlabel("")
    _wrap_axis_ticklabels(ax, "x", width=16, rotation=25)
    _apply_nature_axes(ax)


def _draw_qc_correlation_panel(ax, matrix: pd.DataFrame, sample_meta: pd.DataFrame) -> None:
    corr = matrix[sample_meta["FileName"].tolist()].corr(method="spearman").fillna(0)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        rasterized=True,
        cbar_kws={"label": "Spearman r"},
    )
    ax.set_title("Sample Correlation")
    _apply_nature_heatmap_frame(ax)


def plot_qc_sample_overview(this_run_folder: str, data_BC: pd.DataFrame, sampleinfo: pd.DataFrame,
                            out_file: str = "qc_sample_overview.png") -> Dict[str, Any]:
    file_path = _build_output_path(this_run_folder, out_file)
    matrix, sample_meta, cluster_order, color_map, label_col = _qc_overview_sample_frame(data_BC, sampleinfo)
    n_samples = int(len(sample_meta))
    large_n = n_samples > LARGE_N_PANEL_THRESHOLD

    # Step 04: panels without matching data are dropped from the layout instead
    # of being rendered as gray-text placeholders.
    drawers = [
        ("detected_proteins", lambda ax: _draw_qc_detected_panel(ax, sample_meta, large_n)),
        ("missing_rate", lambda ax: _draw_qc_missing_panel(ax, sample_meta, cluster_order, color_map, large_n)),
        ("sample_counts", lambda ax: _draw_qc_counts_panel(ax, sample_meta, cluster_order, color_map)),
    ]
    if n_samples >= 2:
        drawers.append(("sample_correlation", lambda ax: _draw_qc_correlation_panel(ax, matrix, sample_meta)))
    available_keys = {key for key, _ in drawers}
    omitted_panels = [key for key in ("detected_proteins", "missing_rate", "sample_counts", "sample_correlation")
                      if key not in available_keys]

    _set_visual_theme("notebook")
    ncols = 2 if len(drawers) > 1 else 1
    nrows = (len(drawers) + ncols - 1) // ncols
    fig = plt.figure(figsize=(15, 5.6 * nrows + 0.6))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=([1.0, 1.05] * nrows)[:nrows])
    for idx, (_name, draw) in enumerate(drawers):
        row, col = divmod(idx, ncols)
        spans_row = (row == nrows - 1) and ncols > 1 and (len(drawers) % ncols == 1)
        ax = fig.add_subplot(gs[row, :] if spans_row else gs[row, col])
        draw(ax)

    fig.suptitle("QC Sample Overview", fontsize=18, weight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    return {
        "file_path": file_path,
        "n_samples": n_samples,
        "n_proteins": int(len(matrix)),
        "label_column": label_col,
        "n_unannotated": int((sample_meta["Cluster"] == UNANNOTATED_GROUP_LABEL).sum()),
        "large_n_mode": bool(large_n),
        "omitted_panels": omitted_panels,
    }


def plot_mechanism_enrichment_dotplot(this_run_folder: str, enrichment_df: pd.DataFrame,
                                      out_file: str = "mechanism_enrichment_dotplot.png",
                                      top_n_terms: int = 15) -> Dict[str, Any]:
    if enrichment_df is None or enrichment_df.empty:
        return _placeholder_plot(this_run_folder, out_file, "No enrichment results", "Run run_enrichment() before drawing enrichment evidence.")
    file_path = _build_output_path(this_run_folder, out_file)
    df = enrichment_df.copy()
    df = (
        df.sort_values("p.adjust")
        .groupby(["namespace", "contrast", "direction"], observed=False)
        .head(top_n_terms)
        .copy()
    )
    df["term_label"] = df["namespace"].astype(str) + ": " + df["Description"].astype(str)
    top_terms = df.groupby("term_label")["score"].max().sort_values(ascending=False).head(max(12, top_n_terms * 3)).index
    df = df[df["term_label"].isin(top_terms)].copy()
    df["x_label"] = df["contrast"].astype(str) + "\n" + df["direction"].astype(str).str.replace("regulated", "", regex=False)
    x_order = df.groupby("x_label")["score"].max().sort_values(ascending=False).index.tolist()
    y_order = df.groupby("term_label")["score"].max().sort_values(ascending=True).index.tolist()
    x_map = {x: i for i, x in enumerate(x_order)}
    y_map = {y: i for i, y in enumerate(y_order)}
    df["x"] = df["x_label"].map(x_map)
    df["y"] = df["term_label"].map(y_map)

    _set_visual_theme("notebook")
    fig_w = max(10, min(20, 1.0 * len(x_order) + 5))
    fig_h = max(7, min(18, 0.34 * len(y_order) + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    scatter = ax.scatter(
        df["x"],
        df["y"],
        s=np.clip(df["Count"] * 22, 35, 360),
        c=df["score"],
        cmap="viridis",
        alpha=0.86,
        edgecolors="white",
        linewidth=0.7,
    )
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([_wrap_label(x, 18) for x in x_order], rotation=35, ha="right")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels([_wrap_label(y, 42) for y in y_order])
    ax.set_xlabel("Contrast and direction")
    ax.set_ylabel("")
    ax.set_title("Mechanism Enrichment Dotplot")
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("-log10(FDR)")
    _apply_nature_axes(ax)
    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    return {"file_path": file_path, "n_terms": int(df["term_label"].nunique()), "n_panels": int(len(x_order))}


def _protein_ids_for_genes(data_BC: pd.DataFrame, protein_gene_map: pd.DataFrame, genes: List[str]) -> List[str]:
    if not genes or protein_gene_map is None or protein_gene_map.empty:
        return []
    gene_set = {str(g).strip().upper() for g in genes if str(g).strip()}
    if not gene_set or not {"PG.ProteinGroups", "PG.Genes"}.issubset(protein_gene_map.columns):
        return []
    matched = []
    for _, row in protein_gene_map.iterrows():
        row_genes = {g.upper() for g in offline_enrich.split_gene_tokens(row.get("PG.Genes", ""))}
        if gene_set & row_genes:
            matched.append(str(row["PG.ProteinGroups"]))
    data_ids = set(data_BC["PG.ProteinGroups"].astype(str)) if "PG.ProteinGroups" in data_BC.columns else set()
    return [p for p in matched if p in data_ids]


def _term_gene_list(row: pd.Series) -> List[str]:
    for col in ["geneID", "genes", "Genes"]:
        if col in row and pd.notna(row[col]):
            return offline_enrich.split_gene_tokens(row[col])
    return []


def plot_mechanism_pathway_gene_heatmaps(this_run_folder: str, enrichment_df: pd.DataFrame, data_BC: pd.DataFrame,
                                         sampleinfo: pd.DataFrame, protein_gene_map: pd.DataFrame,
                                         top_n_terms: int = 6, max_genes: int = 30) -> List[Dict[str, Any]]:
    if enrichment_df is None or enrichment_df.empty:
        return []
    outputs = []
    top_terms = enrichment_df.sort_values("p.adjust").head(top_n_terms)
    data_anal, sampleinfo_plot, sample_cols, _ = _prepare_sample_plot_data(data_BC, sampleinfo)
    sample_groups = _normalize_group_labels(sampleinfo_plot.set_index("FileName").loc[sample_cols, "Cluster"])
    cluster_order = _group_label_order(sample_groups)

    for idx, row in top_terms.iterrows():
        genes = _term_gene_list(row)[:max_genes]
        protein_ids = _protein_ids_for_genes(data_BC, protein_gene_map, genes)
        protein_ids = [p for p in protein_ids if p in data_anal.index][:max_genes]
        if len(protein_ids) < 2:
            continue
        means = {}
        for cluster in cluster_order:
            samples = sample_groups[sample_groups == cluster].index.tolist()
            means[cluster] = data_anal.loc[protein_ids, samples].mean(axis=1)
        mean_df = pd.DataFrame(means)
        labels = []
        for protein_id in protein_ids:
            label = protein_id
            if "PG.ProteinGroups" in protein_gene_map.columns and "PG.Genes" in protein_gene_map.columns:
                match = protein_gene_map[protein_gene_map["PG.ProteinGroups"].astype(str) == str(protein_id)]
                if not match.empty:
                    label = str(match.iloc[0].get("PG.Genes", protein_id)).split(";")[0] or str(protein_id)
            labels.append(label)
        z = mean_df.sub(mean_df.mean(axis=1), axis=0).div(mean_df.std(axis=1).replace(0, np.nan), axis=0).fillna(0).clip(-2.5, 2.5)
        z.index = labels
        namespace = _safe_filename(row.get("namespace", "Enrichment"), 24)
        contrast = _safe_filename(row.get("contrast", "contrast"), 70)
        direction = _safe_filename(row.get("direction", "direction"), 24)
        out_file = f"mechanism_pathway_gene_heatmap_{contrast}_{namespace}_{direction}_{idx}.png"
        file_path = _build_output_path(this_run_folder, out_file)

        _set_visual_theme("notebook")
        fig_h = max(5, min(12, 0.32 * len(z) + 2.2))
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        sns.heatmap(
            z,
            ax=ax,
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            center=0,
            vmin=-2.5,
            vmax=2.5,
            rasterized=True,
            cbar_kws={"label": "Group mean row Z-score"},
        )
        ax.set_title(_wrap_label(str(row.get("Description", "Pathway gene heatmap")), 55), pad=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Matched pathway genes")
        _wrap_axis_ticklabels(ax, "x", width=16, rotation=25)
        _apply_nature_heatmap_frame(ax)
        plt.tight_layout()
        _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
        outputs.append({
            "file_path": file_path,
            "description": str(row.get("Description", "")),
            "n_genes": int(len(protein_ids)),
            "namespace": str(row.get("namespace", "")),
            "contrast": str(row.get("contrast", "")),
        })
    return outputs


def plot_mechanism_top_protein_group_means(this_run_folder: str, dep_df: pd.DataFrame, data_BC: pd.DataFrame,
                                           sampleinfo: pd.DataFrame,
                                           out_file: str = "mechanism_top_protein_group_means.png",
                                           top_n_proteins: int = 50) -> Dict[str, Any]:
    if dep_df is None or dep_df.empty:
        return _placeholder_plot(this_run_folder, out_file, "No differential proteins", "Run differential analysis before drawing group-mean evidence.")
    ranked = dep_df[dep_df["is_significant"]].copy()
    if ranked.empty:
        ranked = dep_df.copy()
    ranked = ranked.sort_values("score", ascending=False).drop_duplicates("protein_id").head(top_n_proteins)
    protein_ids = ranked["protein_id"].astype(str).tolist()
    mean_df, detect_df = _group_summary_for_proteins(data_BC, sampleinfo, protein_ids)
    if mean_df.empty:
        return _placeholder_plot(this_run_folder, out_file, "No matched protein matrix rows", "Differential protein IDs could not be matched to ProteinQuant_ComBat.")
    labels = ranked.set_index("protein_id").reindex(mean_df.index)["protein_label"].fillna(mean_df.index.to_series()).astype(str)
    mean_z = mean_df.sub(mean_df.mean(axis=1), axis=0).div(mean_df.std(axis=1).replace(0, np.nan), axis=0).fillna(0).clip(-2.5, 2.5)
    mean_z.index = labels
    detect_df.index = labels
    fig_h = max(7, min(18, 0.28 * len(mean_z) + 3))
    file_path = _build_output_path(this_run_folder, out_file)
    _set_visual_theme("notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, fig_h), gridspec_kw={"width_ratios": [1.0, 0.85]})
    sns.heatmap(
        mean_z,
        ax=axes[0],
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        vmin=-2.5,
        vmax=2.5,
        rasterized=True,
        cbar_kws={"label": "Mean row Z-score"},
    )
    axes[0].set_title("Group Mean Abundance")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Top proteins")
    _apply_nature_heatmap_frame(axes[0])
    sns.heatmap(
        detect_df * 100,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        rasterized=True,
        cbar_kws={"label": "Detection rate (%)"},
    )
    axes[1].set_title("Detection Rate")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    _apply_nature_heatmap_frame(axes[1])
    for ax in axes:
        _wrap_axis_ticklabels(ax, "x", width=16, rotation=25)
    fig.suptitle("Top Differential Protein Group Means", fontsize=17, weight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    return {"file_path": file_path, "n_proteins": int(len(mean_df)), "n_clusters": int(len(mean_df.columns))}


def plot_mechanism_dep_overlap_upset(this_run_folder: str, dep_df: pd.DataFrame,
                                     out_file: str = "mechanism_dep_overlap_upset.png",
                                     max_sets: int = 8, max_intersections: int = 24) -> Dict[str, Any]:
    if dep_df is None or dep_df.empty:
        return _placeholder_plot(this_run_folder, out_file, "No differential proteins", "Run differential analysis before drawing DEP overlap.")
    sig = dep_df[dep_df["is_significant"]].copy()
    if sig.empty:
        return _placeholder_plot(this_run_folder, out_file, "No significant DEP sets", "No proteins passed the selected FDR/logFC thresholds.")
    sets: Dict[str, set] = {}
    for (contrast, direction), sub in sig.groupby(["contrast", "direction"], observed=False):
        label = f"{contrast} {direction}"
        genes = set(sub["protein_label"].dropna().astype(str))
        if genes:
            sets[label] = genes
    if not sets:
        return _placeholder_plot(this_run_folder, out_file, "No DEP sets", "No non-empty up/down protein sets were available.")
    set_names = sorted(sets, key=lambda x: len(sets[x]), reverse=True)[:max_sets]
    sets = {name: sets[name] for name in set_names}
    intersections = []
    names = list(sets)
    for r in range(1, min(4, len(names)) + 1):
        for combo in combinations(names, r):
            inter = set.intersection(*(sets[name] for name in combo))
            others = set().union(*(sets[name] for name in names if name not in combo)) if len(combo) < len(names) else set()
            exclusive = inter - others
            if exclusive:
                intersections.append({"combo": combo, "count": len(exclusive)})
    intersections = sorted(intersections, key=lambda x: x["count"], reverse=True)[:max_intersections]
    if not intersections:
        return _placeholder_plot(this_run_folder, out_file, "No exclusive intersections", "DEP sets overlap, but no exclusive intersections were available after filtering.")

    file_path = _build_output_path(this_run_folder, out_file)
    _set_visual_theme("notebook")
    fig = plt.figure(figsize=(max(10, 0.48 * len(intersections) + 6), max(6.5, 0.34 * len(names) + 4)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.9], width_ratios=[0.28, 1], hspace=0.05, wspace=0.06)
    ax_sizes = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bar)

    counts = [item["count"] for item in intersections]
    x = np.arange(len(intersections))
    ax_bar.bar(x, counts, color="#4C78A8")
    ax_bar.set_ylabel("Exclusive proteins")
    ax_bar.set_title("DEP Overlap UpSet-like Summary")
    ax_bar.set_xticks([])
    _apply_nature_axes(ax_bar)

    set_sizes = [len(sets[name]) for name in names]
    y = np.arange(len(names))
    ax_sizes.barh(y, set_sizes, color="#9D755D")
    ax_sizes.set_yticks(y)
    ax_sizes.set_yticklabels([_wrap_label(name, 24) for name in names], fontsize=8)
    ax_sizes.invert_xaxis()
    ax_sizes.set_xlabel("Set size")
    _apply_nature_axes(ax_sizes)

    for idx, item in enumerate(intersections):
        active = set(item["combo"])
        active_y = []
        for yi, name in enumerate(names):
            is_active = name in active
            ax_matrix.scatter(idx, yi, s=60 if is_active else 22, color="#222222" if is_active else "#d8d8d8")
            if is_active:
                active_y.append(yi)
        if len(active_y) > 1:
            ax_matrix.plot([idx, idx], [min(active_y), max(active_y)], color="#222222", linewidth=1.0)
    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels([])
    ax_matrix.set_xlabel("Top exclusive intersections")
    ax_matrix.set_ylim(-0.5, len(names) - 0.5)
    ax_matrix.invert_yaxis()
    _apply_nature_axes(ax_matrix)

    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    return {"file_path": file_path, "n_sets": int(len(names)), "n_intersections": int(len(intersections))}


def plot_mechanism_contrast_evidence(this_run_folder: str, contrast: str, dep_df: pd.DataFrame,
                                     enrichment_df: pd.DataFrame, data_BC: pd.DataFrame,
                                     sampleinfo: pd.DataFrame,
                                     top_n_proteins: int = 12,
                                     max_labels: int = 12) -> Dict[str, Any]:
    sub = dep_df[dep_df["contrast"].astype(str) == str(contrast)].copy()
    if sub.empty:
        return _placeholder_plot(this_run_folder, f"mechanism_contrast_evidence_{_safe_filename(contrast)}.png", "No contrast data", str(contrast))
    safe_contrast = _safe_filename(contrast)
    file_path = _build_output_path(this_run_folder, f"mechanism_contrast_evidence_{safe_contrast}.png")
    sub["sig_class"] = "Not significant"
    sub.loc[(sub["is_significant"]) & (sub["logFC"] > 0), "sig_class"] = "Up"
    sub.loc[(sub["is_significant"]) & (sub["logFC"] < 0), "sig_class"] = "Down"
    top = sub[sub["is_significant"]].sort_values("score", ascending=False).head(top_n_proteins)
    if top.empty:
        top = sub.sort_values("score", ascending=False).head(top_n_proteins)
    protein_ids = top["protein_id"].astype(str).tolist()
    mean_df, _ = _group_summary_for_proteins(data_BC, sampleinfo, protein_ids)

    enrich_sub = pd.DataFrame()
    if enrichment_df is not None and not enrichment_df.empty:
        contrast_key = str(contrast).replace(" ", "")
        enrich_sub = enrichment_df[
            enrichment_df["contrast"].astype(str).str.replace(" ", "", regex=False).isin([contrast_key])
        ].sort_values("p.adjust").head(10)

    _set_visual_theme("notebook")

    # Step 04: panels without matching data are dropped from the layout instead
    # of being rendered as gray-text placeholders.
    drawers: List[tuple] = []
    omitted_panels: List[str] = []

    def _draw_volcano(ax) -> None:
        colors = {"Up": "#d1495b", "Down": "#2c7fb8", "Not significant": "#b8b8b8"}
        for cls in ["Not significant", "Down", "Up"]:
            ss = sub[sub["sig_class"] == cls]
            ax.scatter(
                ss["logFC"],
                ss["neglog10p"],
                s=18 if cls == "Not significant" else 32,
                color=colors[cls],
                alpha=0.45 if cls == "Not significant" else 0.82,
                edgecolors="none",
                label=f"{cls} (n={len(ss)})",
                rasterized=True,
            )
        label_df = top.drop_duplicates("protein_label").head(max_labels)
        texts = [ax.text(r["logFC"], r["neglog10p"], r["protein_label"], fontsize=8) for _, r in label_df.iterrows()]
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5))
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.9)
        ax.axvline(-1.0, color="#666666", linestyle="--", linewidth=0.9)
        ax.axhline(-np.log10(0.05), color="#666666", linestyle="--", linewidth=0.9)
        ax.set_title("Volcano evidence")
        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10(FDR)")
        _apply_nature_axes(ax)
        _legend_outside(ax, title="Class")

    def _draw_top_proteins(ax) -> None:
        lolli = top.sort_values("logFC")
        y = np.arange(len(lolli))
        ax.hlines(y, 0, lolli["logFC"], color="#d0d0d0", linewidth=1.2)
        ax.scatter(lolli["logFC"], y, c=np.where(lolli["logFC"] >= 0, "#d1495b", "#2c7fb8"), s=np.clip(lolli["neglog10p"] * 30, 35, 220))
        ax.axvline(0, color="#666666", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(x, 22) for x in lolli["protein_label"]], fontsize=8)
        ax.set_xlabel("log2 Fold Change")
        ax.set_title("Top differential proteins")
        _apply_nature_axes(ax)

    def _draw_enrichment(ax) -> None:
        enrich_plot = enrich_sub.sort_values("score")
        ax.barh(
            [_wrap_label(x, 38) for x in enrich_plot["Description"]],
            enrich_plot["score"],
            color="#54A24B",
        )
        ax.set_xlabel("-log10(FDR)")
        ax.set_title("Top enriched terms")
        _apply_nature_axes(ax)

    def _draw_group_means(ax) -> None:
        labels = top.set_index("protein_id").reindex(mean_df.index)["protein_label"].fillna(mean_df.index.to_series()).astype(str)
        mean_z = mean_df.sub(mean_df.mean(axis=1), axis=0).div(mean_df.std(axis=1).replace(0, np.nan), axis=0).fillna(0).clip(-2.5, 2.5)
        mean_z.index = labels
        sns.heatmap(
            mean_z,
            ax=ax,
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            center=0,
            vmin=-2.5,
            vmax=2.5,
            rasterized=True,
            cbar_kws={"label": "Mean row Z-score"},
        )
        ax.set_title("Group means of top proteins")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
        _wrap_axis_ticklabels(ax, "x", width=16, rotation=25)
        _apply_nature_heatmap_frame(ax)

    drawers.append(("volcano_evidence", _draw_volcano))
    drawers.append(("top_proteins", _draw_top_proteins))
    if not enrich_sub.empty:
        drawers.append(("enrichment_terms", _draw_enrichment))
    else:
        omitted_panels.append("enrichment_terms")
    if not mean_df.empty:
        drawers.append(("group_means", _draw_group_means))
    else:
        omitted_panels.append("group_means")

    ncols = 2 if len(drawers) > 1 else 1
    nrows = (len(drawers) + ncols - 1) // ncols
    fig = plt.figure(figsize=(15, 5.6 * nrows + 0.6))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=([1.0, 1.05] * nrows)[:nrows])
    for idx, (_name, draw) in enumerate(drawers):
        row, col = divmod(idx, ncols)
        spans_row = (row == nrows - 1) and ncols > 1 and (len(drawers) % ncols == 1)
        ax = fig.add_subplot(gs[row, :] if spans_row else gs[row, col])
        draw(ax)

    fig.suptitle(f"Mechanism Evidence: {contrast}", fontsize=18, weight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(file_path, fig=fig, dpi=300, bbox_inches="tight")
    return {
        "file_path": file_path,
        "contrast": str(contrast),
        "n_top_proteins": int(len(top)),
        "omitted_panels": omitted_panels,
    }

def _run_enrichment(gene_list, enrich_type):

    r_genes = StrVector(gene_list)
    robjects.r.assign("genes", r_genes)
    try:
        if enrich_type.lower() == "go":
            robjects.r(f'''
                # ---- 1. ID 转换
                gene_df <- bitr(
                    unique(genes),
                    fromType = "UNIPROT",
                    toType   = "SYMBOL",
                    OrgDb    = org.Hs.eg.db
                )
                genes <- unique(na.omit(gene_df$SYMBOL))
                evalue <- enrichGO(
                    gene         = genes,
                    OrgDb        = org.Hs.eg.db,
                    keyType      = "SYMBOL",
                    ont          = "ALL",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05,
                    readable     = TRUE
                )
                '''
                )
        elif enrich_type.lower() == "kegg":
            robjects.r(f'''
                evalue <- enrichKEGG(
                    gene         = genes,
                    organism     = "hsa",
                    keyType = "uniprot",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05
                )
            '''
            )
        elif enrich_type.lower() == "reactome":  # reactome
            robjects.r(f'''
                genes_entrez <- bitr(
                    genes,
                    fromType = "UNIPROT",
                    toType   = "ENTREZID",
                    OrgDb    = org.Hs.eg.db
                )
                evalue <- enrichPathway(
                    gene          = unique(na.omit(genes_entrez$ENTREZID)),
                    organism      = "human",
                    pAdjustMethod = "BH",
                    pvalueCutoff  = 0.05,
                    qvalueCutoff  = 0.05,
                    readable      = TRUE
                )
            '''
            )
    except Exception as e:
        print(f"[Error] Enrichment analysis failed for type {enrich_type}: {e}")
        return pd.DataFrame()

    # 把结果转换成 pandas DataFrame
    with conversion.localconverter(default_converter + pandas2ri.converter):
        enrich_df = conversion.rpy2py(robjects.r('as.data.frame(evalue)'))
    return enrich_df

def _run_enrichment_mouse(gene_list, enrich_type):

    r_genes = StrVector(gene_list)
    robjects.r.assign("genes", r_genes)
    try:
        if enrich_type.lower() == "go":
            robjects.r(f'''
                # ---- 1. ID 转换
                gene_df <- bitr(
                    unique(genes),
                    fromType = "UNIPROT",
                    toType   = "SYMBOL",
                    OrgDb    = org.Mm.eg.db
                )
                genes <- unique(na.omit(gene_df$SYMBOL))
                evalue <- enrichGO(
                    gene         = genes,
                    OrgDb        = org.Mm.eg.db,
                    keyType      = "SYMBOL",
                    ont          = "ALL",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05,
                    readable     = TRUE
                )
                '''
                )
        elif enrich_type.lower() == "kegg":
            robjects.r(f'''
                evalue <- enrichKEGG(
                    gene         = genes,
                    organism     = "mmu",
                    keyType = "uniprot",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05
                )
            '''
            )
        elif enrich_type.lower() == "reactome":  # reactome
            robjects.r(f'''
                genes_entrez <- bitr(
                    genes,
                    fromType = "UNIPROT",
                    toType   = "ENTREZID",
                    OrgDb    = org.Mm.eg.db
                )
                evalue <- enrichPathway(
                    gene          = unique(na.omit(genes_entrez$ENTREZID)),
                    organism      = "mouse",
                    pAdjustMethod = "BH",
                    pvalueCutoff  = 0.05,
                    qvalueCutoff  = 0.05,
                    readable      = TRUE
                )
            '''
            )
    except Exception as e:
        print(f"[Error] Enrichment analysis failed for type {enrich_type}: {e}")
        return pd.DataFrame()

    # 把结果转换成 pandas DataFrame
    with conversion.localconverter(default_converter + pandas2ri.converter):
        enrich_df = conversion.rpy2py(robjects.r('as.data.frame(evalue)'))
    return enrich_df


def _prepare_gene_dict(processed_proteins_folder, contrast_list, exclude_genes):
    """
    只做一次 IO，把所有 contrast 的 up/down 整理好
    """
    gene_dict = {}

    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")
        up_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_up.csv")
        down_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_down.csv")
        if not os.path.exists(up_path) or not os.path.exists(down_path):
            continue

        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        gene_col = "PG.Genes" if "PG.Genes" in up_df.columns and "PG.Genes" in down_df.columns else "PG.ProteinGroups"
        up_gene_list = [
            gene
            for value in up_df.loc[:, gene_col].astype(str).tolist()
            for gene in offline_enrich.split_gene_tokens(value)
        ]
        down_gene_list = [
            gene
            for value in down_df.loc[:, gene_col].astype(str).tolist()
            for gene in offline_enrich.split_gene_tokens(value)
        ]

        # 去除不需要基因
        if exclude_genes:
            up_gene_list = [g for g in up_gene_list if not any(g.startswith(x) for x in exclude_genes)]
            down_gene_list = [g for g in down_gene_list if not any(g.startswith(x) for x in exclude_genes)]

        gene_dict[f"{safe_contrast}_up"] = list(set(up_gene_list))
        gene_dict[f"{safe_contrast}_down"] = list(set(down_gene_list))

    return gene_dict


def _run_enrichment_batch_human(gene_dict: Dict[str, List[str]], enrich_type: str):

    r_gene_dict = robjects.ListVector({
        k: StrVector(v) for k, v in gene_dict.items() if len(v) > 0
    })

    robjects.r.assign("gene_dict", r_gene_dict)

    if "go" in enrich_type.lower():
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
            library(ReactomePA)
            library(org.Hs.eg.db)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            gene_df <- bitr(
                genes,
                fromType="UNIPROT",
                toType="SYMBOL",
                OrgDb=org.Hs.eg.db
            )
            gene_df <- gene_df[!duplicated(gene_df$UNIPROT), ]
            genes2 <- unique(na.omit(gene_df$SYMBOL))

            if (length(genes2) > 5) {
                res <- enrichGO(
                    gene = genes2,
                    OrgDb = org.Hs.eg.db,
                    keyType = "SYMBOL",
                    ont = "ALL",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    elif "kegg" in enrich_type.lower():
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            if (length(genes) > 5) {
                res <- enrichKEGG(
                    gene = genes,
                    organism = "hsa",
                    keyType = "uniprot",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    elif "reactome" in enrich_type.lower():
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
            library(ReactomePA)
            library(org.Hs.eg.db)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            gene_df <- bitr(
                genes,
                fromType="UNIPROT",
                toType="ENTREZID",
                OrgDb=org.Hs.eg.db
            )
            gene_df <- gene_df[!duplicated(gene_df$UNIPROT), ]
            genes2 <- unique(na.omit(gene_df$ENTREZID))
            if (length(genes2) > 5) {
                res <- enrichPathway(
                    gene = genes2,
                    organism = "human",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff  = 0.05,
                    readable      = TRUE
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    # 转 pandas
    results = {}
    r_res = robjects.r("results")

    for name in r_res.names:
        with conversion.localconverter(default_converter + pandas2ri.converter):
            results[name] = robjects.conversion.rpy2py(r_res.rx2(name))
    return results

def _run_enrichment_batch_mouse(gene_dict: Dict[str, List[str]], enrich_type: str):

    r_gene_dict = robjects.ListVector({
        k: StrVector(v) for k, v in gene_dict.items() if len(v) > 0
    })

    robjects.r.assign("gene_dict", r_gene_dict)

    if enrich_type.lower() == "go":
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
            library(ReactomePA)
            library(org.Mm.eg.db)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            gene_df <- bitr(
                genes,
                fromType="UNIPROT",
                toType="SYMBOL",
                OrgDb=org.Mm.eg.db
            )
            gene_df <- gene_df[!duplicated(gene_df$UNIPROT), ]
            genes2 <- unique(na.omit(gene_df$SYMBOL))

            if (length(genes2) > 5) {
                res <- enrichGO(
                    gene = genes2,
                    OrgDb = org.Mm.eg.db,
                    keyType = "SYMBOL",
                    ont = "ALL",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    elif enrich_type.lower() == "kegg":
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            if (length(genes) > 5) {
                res <- enrichKEGG(
                    gene = genes,
                    organism = "mmu",
                    keyType = "uniprot",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    elif enrich_type.lower() == "reactome":
        robjects.r("""
        suppressMessages({
            library(clusterProfiler)
            library(ReactomePA)
            library(org.Mm.eg.db)
        })
        results <- list()
        for (nm in names(gene_dict)) {
            genes <- unique(gene_dict[[nm]])
            gene_df <- bitr(
                genes,
                fromType="UNIPROT",
                toType="ENTREZID",
                OrgDb=org.Mm.eg.db
            )
            gene_df <- gene_df[!duplicated(gene_df$UNIPROT), ]
            genes2 <- unique(na.omit(gene_df$ENTREZID))
            if (length(genes2) > 5) {
                res <- enrichPathway(
                    gene = genes2,
                    organism = "mouse",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff  = 0.05,
                    readable      = TRUE
                )
                results[[nm]] <- as.data.frame(res)
            } else {
                results[[nm]] <- data.frame()
            }
        }
        """)

    # 转 pandas
    results = {}
    r_res = robjects.r("results")

    for name in r_res.names:
        with conversion.localconverter(default_converter + pandas2ri.converter):
            results[name] = robjects.conversion.rpy2py(r_res.rx2(name))
    return results


# =========================
# 🔥 并行 wrapper
# =========================

def _run_single_enrich(args):
    enrich_type, gene_dict, species, fdr_cutoff, top_n = args
    try:
        species_name = offline_enrich.canonical_species(species)
        top_n_limit = int(top_n) if top_n is not None and int(top_n) > 0 else None
        fdr_limit = None if fdr_cutoff is None else float(fdr_cutoff)
        gene_sets = offline_enrich.load_gene_sets(enrich_type, species=species_name)
        universe_set = {gene for entry in gene_sets.values() for gene in entry["genes"]}
        results = {}
        for key, genes in gene_dict.items():
            normalized_genes = offline_enrich.normalize_gene_list(genes, species_name)
            matched_genes = sorted({gene for gene in normalized_genes if gene in universe_set})
            diagnostics = {
                "namespace": enrich_type,
                "species": species_name,
                "input_gene_count": int(len([g for g in genes if str(g).strip()])),
                "normalized_query_gene_count": int(len(normalized_genes)),
                "matched_offline_gene_count": int(len(matched_genes)),
                "strict_fdr_rows": 0,
                "unfiltered_rows": 0,
                "selected_rows": 0,
                "fdr_cutoff": fdr_limit,
                "passes_fdr": False,
                "significance_status": "not_run",
                "fallback_reason": "",
            }

            df = pd.DataFrame(columns=offline_enrich.ora_columns())
            if not gene_sets:
                diagnostics["significance_status"] = "no_gene_set_resource"
                diagnostics["fallback_reason"] = "offline_gene_set_resource_missing_or_empty"
            elif not normalized_genes:
                diagnostics["significance_status"] = "no_input_genes"
                diagnostics["fallback_reason"] = "no_query_genes_after_cleaning"
            elif not matched_genes:
                diagnostics["significance_status"] = "no_genes_matched_offline_resource"
                diagnostics["fallback_reason"] = "query_genes_not_found_in_offline_gene_sets"
            else:
                strict_df = offline_enrich.run_ora(
                    normalized_genes,
                    enrich_type,
                    species=species_name,
                    fdr_cutoff=fdr_limit,
                    top_n=None,
                    min_overlap=1,
                )
                diagnostics["strict_fdr_rows"] = int(len(strict_df))
                if len(strict_df) > 0:
                    df = strict_df.head(top_n_limit).copy() if top_n_limit else strict_df.copy()
                    diagnostics["passes_fdr"] = True
                    diagnostics["significance_status"] = "fdr_significant"
                else:
                    unfiltered_df = offline_enrich.run_ora(
                        normalized_genes,
                        enrich_type,
                        species=species_name,
                        fdr_cutoff=None,
                        top_n=None,
                        min_overlap=1,
                    )
                    diagnostics["unfiltered_rows"] = int(len(unfiltered_df))
                    if len(unfiltered_df) > 0:
                        df = unfiltered_df.head(top_n_limit).copy() if top_n_limit else unfiltered_df.copy()
                        diagnostics["significance_status"] = "exploratory_not_fdr_significant"
                        diagnostics["fallback_reason"] = "strict_fdr_empty_using_top_unfiltered_terms"
                    else:
                        diagnostics["significance_status"] = "no_overlapping_terms"
                        diagnostics["fallback_reason"] = "no_terms_met_min_overlap"

            diagnostics["selected_rows"] = int(len(df))
            if len(df) > 0:
                for col, value in diagnostics.items():
                    df[col] = value
            results[key] = {"df": df, "diagnostics": diagnostics}
        return enrich_type, results
    except Exception as e:
        print(f"[Error] {enrich_type} failed: {e}")
        return enrich_type, {}


def _enrichment_cache_has_diagnostics(obj):
    if isinstance(obj, dict):
        if "diagnostics" in obj:
            return True
        return any(_enrichment_cache_has_diagnostics(value) for value in obj.values())
    if isinstance(obj, list):
        return any(_enrichment_cache_has_diagnostics(value) for value in obj)
    return False



@tool
def run_enrichment(
    contrast_type: str = "ALL",
    exclude_genes: List[str] = [],
    enrich_types: List[str] = ['GO', 'KEGG', 'Reactome'],
    species: str = "human",
    fdr_cutoff: float = 0.05,
    top_n: int = 10,
    *args, **kwargs
):
    """
    ✅ Batch + 并行版本
    species: human / mouse
    """

    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    out_folder = os.path.join(this_run_folder, "enrichment_results")
    os.makedirs(out_folder, exist_ok=True)
    enrich_path = os.path.join(out_folder, "go_kegg_reactome_results.json")
    if os.path.exists(enrich_path):
        pre_info = load_json(enrich_path)
        if _enrichment_cache_has_diagnostics(pre_info):
            if contrast_type.lower() == "all" and contrast_type in pre_info:
                return pre_info
            else:
                if contrast_type in pre_info:
                    return pre_info[contrast_type]
                if normalize_contrast(contrast_type) in pre_info:
                    return pre_info[normalize_contrast(contrast_type)]
        else:
            print("Cached enrichment results lack diagnostics; recomputing enrichment.")

            # for enrich_type in enrich_types:
            #     if enrich_type in pre_info:
            #         if contrast_type in pre_info[enrich_type]:
            #             results[enrich_type] = pre_info[enrich_type][contrast_type]
            #         if normalize_contrast(contrast_type) in pre_info[enrich_type]:
            #             results[enrich_type] = pre_info[enrich_type][normalize_contrast(contrast_type)]
            # if results:
            #     return results


    contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f)) and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())]

    if not contrast_list:
        contrast_list = [f for f in os.listdir(processed_proteins_folder) if os.path.isdir(os.path.join(processed_proteins_folder, f))]


    gene_dict = _prepare_gene_dict(processed_proteins_folder, contrast_list, exclude_genes)

    tasks = [(et, gene_dict, species, fdr_cutoff, top_n) for et in enrich_types]
    # results = {}
    results = {}
    output_folder = os.path.join(this_run_folder, "enrichment_results")
    os.makedirs(output_folder, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(enrich_types), 4)) as executor:
        futures = [executor.submit(_run_single_enrich, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Enrichment progress"):
            enrich_type, res_dict = future.result()
            # results[enrich_type] = res_dict
    # with ProcessPoolExecutor(max_workers=min(len(enrich_types), 4)) as executor:
    #     for enrich_type, res_dict in executor.map(_run_single_enrich, tasks):
    #         results[enrich_type] = res_dict
            temp = {}
            for key, payload in res_dict.items():
                if isinstance(payload, dict) and "df" in payload:
                    df = payload.get("df")
                    diagnostics = payload.get("diagnostics", {})
                else:
                    df = payload
                    diagnostics = {}
                if key.endswith("_up"):
                    contrast = key.replace("_up", "")
                    direction = "upregulated"
                else:
                    contrast = key.replace("_down", "")
                    direction = "downregulated"
                if contrast not in temp:
                    temp[contrast] = {"upregulated": {}, "downregulated": {}, "diagnostics": {}}
                temp[contrast]["diagnostics"][direction] = diagnostics

                if df is None or len(df) == 0:
                    temp[contrast][direction] = {}
                    continue

                sub_results = {}
                for _, row in df.iterrows():
                    gid = row.get("geneID", "")
                    desc = row.get("Description", "")
                    if gid:
                        sub_results[gid] = sub_results.get(gid, "") + "; " + desc if gid in sub_results else desc
                temp[contrast][direction] = sub_results

                # sub_results = []
                # for _, row in df.iterrows():
                #     gid = row.get("geneID", "")
                #     desc = row.get("Description", "")
                #     if gid:
                #         sub_results.append(desc)
                # temp[contrast][direction] = ';'.join(sub_results)


                if df is None or len(df) == 0:
                        continue
                # 判断 up/down
                if key.endswith("_up"):
                    contrast = key.replace("_up", "")
                    direction = "upregulated"
                else:
                    contrast = key.replace("_down", "")
                    direction = "downregulated"
                safe_contrast = contrast
                output_file = os.path.join(output_folder, f"{enrich_type}_{safe_contrast}_{direction}_proteins.csv")
                df_out = df.copy()
                df_out.insert(0, "Cluster", safe_contrast)
                _safe_to_csv(df_out, output_file, index=False, encoding='utf-8-sig')
            results[enrich_type] = temp

    final_results = {contrast_type: results,}
    out_folder = os.path.join(this_run_folder, "enrichment_results")
    os.makedirs(out_folder, exist_ok=True)
    enrich_path = os.path.join(out_folder, "go_kegg_reactome_results.json")
    save_json(final_results, enrich_path)
    diagnostic_counts = {}
    for enrich_type, payload in results.items():
        passes = 0
        exploratory = 0
        for contrast_payload in payload.values():
            if not isinstance(contrast_payload, dict):
                continue
            for direction_payload in contrast_payload.get("diagnostics", {}).values():
                if direction_payload.get("passes_fdr"):
                    passes += 1
                if direction_payload.get("significance_status") == "exploratory_not_fdr_significant":
                    exploratory += 1
        diagnostic_counts[enrich_type] = {"passes_fdr_directions": passes, "exploratory_directions": exploratory}
    append_evidence(
        this_run_folder,
        source="offline_enrichment",
        tool="run_enrichment",
        claim="Offline GO/KEGG/Reactome enrichment completed; FDR and exploratory status are recorded per contrast and direction.",
        files=[enrich_path],
        metrics=diagnostic_counts,
        confidence="moderate",
        limitations=["Exploratory enrichment terms must not be reported as FDR-significant findings."],
    )

    return final_results


@tool
def run_pairwise_limma(
    *args, **kwargs
):
    """
    基于 ComBat 校正后的蛋白定量矩阵执行两两分组的 limma 差异分析。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')
    parameters_path = resolve_path('parameters_path')
    parameters = {}
    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(processed_proteins_folder, exist_ok=True)
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')
    protein_gene_map_path = os.path.join(processed_proteins_folder, 'Protein_Gene_Map.csv')


    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    def _select_group_column(df: pd.DataFrame) -> str | None:
        design_group = _design_value("group_col")
        if design_group in df.columns:
            return design_group
        priority = [
            "MembraneStatus", "Type1", "Cluster", "Group", "Condition", "Treatment", "CellType",
            "cellType", "Type", "State", "SampleGroup", "Phenotype",
        ]
        excluded = {"FileName", "Batch", "Date", "Run", "RawFile", "Sample", "SampleID"}
        candidates = priority + [col for col in df.columns if col not in priority and col not in excluded]
        for col in candidates:
            if col not in df.columns:
                continue
            values = df[col].dropna().astype(str).str.strip()
            values = values[values != ""]
            n_unique = int(values.nunique())
            if 2 <= n_unique < len(values) and n_unique <= 30:
                return col
        return None

    group_col = _select_group_column(sampleinfo)
    sample_col = _sample_id_col(sampleinfo)
    if sample_col != "FileName":
        sampleinfo = sampleinfo.copy()
        sampleinfo["FileName"] = sampleinfo[sample_col].astype(str)
    if group_col is None:
        processed_proteins_folder = os.path.join(resolve_path('this_run_folder_path'), 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
        else:
            run_umap.func()
            if os.path.exists(sampleinfo_with_cluster_path):
                sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
            else:
                raise ValueError("No usable grouping column found in sampleinfo and sampleinfo_with_clusters.csv was not generated. Please check metadata or the UMAP clustering step.")
        group_col = "Cluster"
    if group_col not in sampleinfo.columns:
        raise ValueError(f"Selected grouping column '{group_col}' is missing from sampleinfo.")
    if group_col != "Cluster":
        sampleinfo = sampleinfo.copy()
        sampleinfo["Cluster"] = sampleinfo[group_col]
    selected_contrasts = _design_contrasts(sampleinfo, "Cluster")


    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if os.path.exists(protein_quant_combat_path):
        data_BC = pd.read_csv(protein_quant_combat_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")


    if not os.path.exists(protein_gene_map_path):
        print("Gene map file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if os.path.exists(protein_gene_map_path):
        protein_gene_map = pd.read_csv(protein_gene_map_path)
    else:
        raise FileNotFoundError(f"The file of {protein_gene_map_path} is not found.")


    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)

    if not R_AVAILABLE:
        if "FileName" not in sampleinfo.columns:
            raise ValueError("SampleInfo is missing required column: FileName")
        if "PG.ProteinGroups" not in data_BC.columns:
            raise ValueError("ProteinQuant_ComBat.csv is missing required column: PG.ProteinGroups")

        sampleinfo_filtered = sampleinfo.dropna(subset=["FileName", "Cluster"]).copy()
        sampleinfo_filtered["FileName"] = sampleinfo_filtered["FileName"].astype(str)
        sampleinfo_filtered["Cluster"] = sampleinfo_filtered["Cluster"].astype(str).str.strip()
        sampleinfo_filtered = sampleinfo_filtered[sampleinfo_filtered["Cluster"] != ""]
        sampleinfo_filtered = sampleinfo_filtered[sampleinfo_filtered["FileName"].isin(data_BC.columns)]

        if sampleinfo_filtered.empty:
            raise ValueError("No SampleInfo FileName values matched ProteinQuant_ComBat sample columns.")

        sample_cols = sampleinfo_filtered["FileName"].drop_duplicates().tolist()
        matrix = data_BC.loc[:, ["PG.ProteinGroups"] + sample_cols].copy()
        matrix.loc[:, sample_cols] = matrix.loc[:, sample_cols].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.groupby("PG.ProteinGroups", as_index=False)[sample_cols].mean()

        gene_lookup = {}
        if {"PG.ProteinGroups", "PG.Genes"}.issubset(protein_gene_map.columns):
            gene_map = protein_gene_map.loc[:, ["PG.ProteinGroups", "PG.Genes"]].copy()
            gene_map["PG.ProteinGroups"] = gene_map["PG.ProteinGroups"].astype(str).str.split(";").str[0]
            gene_map["PG.Genes"] = gene_map["PG.Genes"].astype(str)
            gene_lookup = gene_map.drop_duplicates("PG.ProteinGroups").set_index("PG.ProteinGroups")["PG.Genes"].to_dict()

        def _safe_name(value: Any) -> str:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
            return name or "group"

        def _write_pairwise(
            sub_info: pd.DataFrame,
            compare_col: str,
            prefix: str | None = None,
            contrast_specs: List[Dict[str, str]] | None = None,
        ) -> Dict[str, Any]:
            sub_info = sub_info.dropna(subset=[compare_col, "FileName"]).copy()
            sub_info[compare_col] = sub_info[compare_col].astype(str).str.strip()
            sub_info = sub_info[sub_info[compare_col] != ""]
            levels = sorted(sub_info[compare_col].dropna().unique().tolist())
            summary: Dict[str, Any] = {}
            if len(levels) < 2:
                return summary

            pair_specs = contrast_specs or [
                {"name": f"{_safe_name(group_a)}_vs_{_safe_name(group_b)}", "group_a": str(group_a), "group_b": str(group_b)}
                for group_a, group_b in combinations(levels, 2)
            ]
            for spec in pair_specs:
                group_a = str(spec.get("group_a", "")).strip()
                group_b = str(spec.get("group_b", "")).strip()
                if group_a not in levels or group_b not in levels:
                    continue
                samples_a = [s for s in sub_info.loc[sub_info[compare_col] == group_a, "FileName"].astype(str) if s in matrix.columns]
                samples_b = [s for s in sub_info.loc[sub_info[compare_col] == group_b, "FileName"].astype(str) if s in matrix.columns]
                if not samples_a or not samples_b:
                    continue

                values_a = matrix.loc[:, samples_a].to_numpy(dtype=float)
                values_b = matrix.loc[:, samples_b].to_numpy(dtype=float)
                mean_a = np.nanmean(values_a, axis=1)
                mean_b = np.nanmean(values_b, axis=1)
                logfc = mean_a - mean_b
                ave_expr = np.nanmean(np.concatenate([values_a, values_b], axis=1), axis=1)

                t_stats = []
                p_values = []
                for row_a, row_b in zip(values_a, values_b):
                    row_a = row_a[~np.isnan(row_a)]
                    row_b = row_b[~np.isnan(row_b)]
                    if len(row_a) < 2 or len(row_b) < 2:
                        t_stats.append(0.0)
                        p_values.append(1.0)
                        continue
                    if np.nanstd(row_a) == 0 and np.nanstd(row_b) == 0:
                        t_stats.append(0.0)
                        p_values.append(1.0 if np.nanmean(row_a) == np.nanmean(row_b) else 0.05)
                        continue
                    stat, p_val = ttest_ind(row_a, row_b, equal_var=False, nan_policy="omit")
                    t_stats.append(float(stat) if np.isfinite(stat) else 0.0)
                    p_values.append(float(p_val) if np.isfinite(p_val) else 1.0)

                p_values = np.asarray(p_values, dtype=float)
                adj_p = multipletests(p_values, method="fdr_bh")[1] if len(p_values) else np.asarray([])
                canonical_pg = matrix["PG.ProteinGroups"].astype(str).str.split(";").str[0]
                contrast_core = f"{_safe_name(group_a)}_vs_{_safe_name(group_b)}"
                requested_name = spec.get("name") or contrast_core
                contrast = _safe_name(requested_name)
                if prefix and not contrast.startswith(f"{_safe_name(prefix)}_"):
                    contrast = f"{_safe_name(prefix)}_{contrast}"

                df_out = pd.DataFrame({
                    "PG.ProteinGroups": matrix["PG.ProteinGroups"].astype(str),
                    "PG.Genes": canonical_pg.map(gene_lookup).fillna(""),
                    "contrast": contrast,
                    "protein": matrix["PG.ProteinGroups"].astype(str),
                    "logFC": logfc,
                    "AveExpr": ave_expr,
                    "t": t_stats,
                    "P.Value": p_values,
                    "adj.P.Val": adj_p,
                    "B": np.nan,
                })
                df_out = df_out.sort_values("logFC", ascending=False)
                out_file = os.path.join(output_folder, f"differential_{contrast}.csv")
                _safe_to_csv(df_out, out_file, index=False, encoding="utf-8-sig")

                diff_cfg = _get_analysis_design().get("differential", {}) or {}
                p_thresh_summary = float(diff_cfg.get("p_thresh", 0.05) or 0.05)
                logfc_thresh_summary = float(diff_cfg.get("logfc_thresh", 0.25) or 0.25)
                unit_brief = {}
                try:
                    import analysis_extensions as _unit_mod
                    _gene_map = {}
                    if "PG.Genes" in df_out.columns:
                        _gene_map = {str(k): str(v) for k, v in zip(df_out["PG.ProteinGroups"], df_out["PG.Genes"])}
                    _unit_res = _unit_mod.unit_aware_sensitivity(
                        matrix=matrix,
                        sampleinfo=sub_info,
                        design=_get_analysis_design(),
                        group_col=compare_col,
                        arm_a=group_a,
                        arm_b=group_b,
                        contrast=contrast,
                        sample_col="FileName",
                        protein_col="PG.ProteinGroups",
                        gene_map=_gene_map,
                        p_thresh=p_thresh_summary,
                        logfc_thresh=logfc_thresh_summary,
                        effect_scale=_unit_mod.effect_scale_record(_get_analysis_design()),
                        out_dir=output_folder,
                    )
                    unit_brief = {
                        "status": _unit_res.get("status"),
                        "estimability": _unit_res.get("estimability"),
                        "estimability_declared": _unit_res.get("estimability_declared"),
                        "analysis_route": _unit_res.get("analysis_route"),
                        "direction": _unit_res.get("direction"),
                        "design_class": _unit_res.get("design_class"),
                        "analysis_unit": _unit_res.get("unit_column") or "",
                        "unit_column_source": _unit_res.get("unit_column_source") or "",
                        "unit_exclusion": _unit_res.get("unit_exclusion") or {},
                        "within_unit": _unit_res.get("within_unit") or {},
                        "between_unit": _unit_res.get("between_unit") or {},
                        "confounding": _unit_res.get("confounding") or {},
                        "test": _unit_res.get("test") or "",
                        "estimand": _unit_res.get("estimand") or "",
                        "n_units_a": (_unit_res.get("n_units") or {}).get("group_a"),
                        "n_units_b": (_unit_res.get("n_units") or {}).get("group_b"),
                        "n_units_shared": (_unit_res.get("n_units") or {}).get("shared"),
                        "n_units_a_only": (_unit_res.get("n_units") or {}).get("group_a_only"),
                        "n_units_b_only": (_unit_res.get("n_units") or {}).get("group_b_only"),
                        "n_units_declared": _unit_res.get("n_units_declared"),
                        "n_pairs": _unit_res.get("n_pairs"),
                        "n_proteins_tested": _unit_res.get("n_proteins_tested"),
                        "n_proteins_not_tested": _unit_res.get("n_proteins_not_tested"),
                        "n_passing_fdr": _unit_res.get("n_passing_fdr"),
                        "n_passing_fdr_only": _unit_res.get("n_passing_fdr_only"),
                        "n_passing_joint": _unit_res.get("n_passing_joint"),
                        "logfc_threshold": _unit_res.get("logfc_threshold"),
                        "effect_column": _unit_res.get("effect_column"),
                        "effect_scale": _unit_res.get("effect_scale") or {},
                        "correction_scope": _unit_res.get("correction_scope"),
                        "filter_rule": _unit_res.get("filter_rule"),
                        "min_adj_p_value": _unit_res.get("min_adj_p_value"),
                        "results_file": _unit_res.get("results_file"),
                        "reasons": _unit_res.get("reasons") or [],
                        "role": "sensitivity_analysis",
                        "reference_analysis": _unit_res.get("reference_analysis"),
                    }
                    save_json(_unit_res, os.path.join(output_folder, "unit_structure_%s.json" % _safe_name(contrast)))
                except Exception as _unit_exc:
                    unit_brief = {"status": "error", "error": str(_unit_exc), "role": "sensitivity_analysis"}
                sig = df_out[df_out["adj.P.Val"] < p_thresh_summary]
                sig_design = sig[sig["logFC"].abs() > logfc_thresh_summary]
                parameters[contrast] = {
                    "differential_protein_file": out_file,
                    "contrast_type": contrast,
                    "method": "python_welch_ttest_fallback",
                    "group_col": group_col if group_col else compare_col,
                    "analysis_column": compare_col,
                    "groups": [group_a, group_b],
                    "thresholds": diff_cfg,
                    "statistical_backend": "python_fallback",
                    "r_available": False,
                }
                if unit_brief:
                    parameters[contrast]["unit_sensitivity"] = unit_brief
                summary[contrast] = {
                    "contrast": contrast,
                    "method": "Python Welch t-test fallback with Benjamini-Hochberg FDR",
                    "evidence_source": "current_matrix",
                    "statistical_backend": "python_fallback",
                    "group_col": group_col if group_col else compare_col,
                    "analysis_column": compare_col,
                    "analysis_design_source": _get_analysis_design().get("source", "unknown"),
                    "r_available": False,
                    "fallback_reason": f"R/rpy2 unavailable: {R_IMPORT_ERROR}",
                    "n_proteins": int(df_out.shape[0]),
                    "n_samples_group_a": int(len(samples_a)),
                    "n_samples_group_b": int(len(samples_b)),
                    "median_logFC_sig": round(float(sig_design["logFC"].median()), 4) if not sig_design.empty else 0,
                    "sanity_checks": {
                        "n_proteins_before_combat_calibration": int(data_BC.shape[0]),
                        "n_proteins_after_combat_calibration": int(df_out.shape[0]),
                        "n_significant": int(sig_design.shape[0]),
                        "n_up": int((sig_design["logFC"] > 0).sum()),
                        "n_down": int((sig_design["logFC"] < 0).sum()),
                        "thresholds_used": {
                            "adj_p": p_thresh_summary,
                            "abs_logFC": logfc_thresh_summary,
                        },
                    },
                }
                if unit_brief:
                    summary[contrast]["unit_sensitivity"] = unit_brief
            return summary

        llm_summary: Dict[str, Any] = {}
        type_cols = [col for col in sampleinfo_filtered.columns if re.match(r"^Type[0-9]+$", str(col))]
        if selected_contrasts:
            llm_summary.update(_write_pairwise(
                sampleinfo_filtered,
                "Cluster",
                contrast_specs=selected_contrasts,
            ))
        elif type_cols:
            for type_col in type_cols:
                for value in sampleinfo_filtered[type_col].dropna().astype(str).unique():
                    subset = sampleinfo_filtered[sampleinfo_filtered[type_col].astype(str) == value]
                    llm_summary.update(_write_pairwise(subset, "Cluster", prefix=value))
                llm_summary.update(_write_pairwise(sampleinfo_filtered, type_col))
        else:
            llm_summary.update(_write_pairwise(
                sampleinfo_filtered,
                "Cluster",
                prefix="ALL",
                contrast_specs=selected_contrasts or None,
            ))

        if not llm_summary:
            raise ValueError("No pairwise differential comparisons could be generated from the available grouping metadata.")

        limma_summary_path = os.path.join(output_folder, "limma_summary.json")
        save_json(llm_summary, limma_summary_path)
        save_json(parameters, parameters_path)
        append_evidence(
            this_run_folder,
            source="current_matrix",
            tool="run_pairwise_limma",
            claim="Differential protein analysis completed with Python fallback; contrasts were taken from analysis_design when available.",
            files=[limma_summary_path],
            metrics={
                "statistical_backend": "python_fallback",
                "group_col": group_col,
                "n_contrasts": len(llm_summary),
                "design_contrasts_used": bool(selected_contrasts),
                "unit_sensitivity_contrasts": int(sum(
                    1 for row in llm_summary.values()
                    if isinstance(row, dict) and (row.get("unit_sensitivity") or {}).get("status") == "completed")),
                "unit_analysis_units": sorted({
                    str((row.get("unit_sensitivity") or {}).get("analysis_unit") or "")
                    for row in llm_summary.values() if isinstance(row, dict)
                    and (row.get("unit_sensitivity") or {}).get("analysis_unit")}),
                "unit_sensitivity_passed_fdr": int(sum(
                    int(((row.get("unit_sensitivity") or {}).get("n_passing_fdr") or 0))
                    for row in llm_summary.values() if isinstance(row, dict))),
                "unit_sensitivity_passed_joint": int(sum(
                    int(((row.get("unit_sensitivity") or {}).get("n_passing_joint") or 0))
                    for row in llm_summary.values() if isinstance(row, dict))),
                "unit_sensitivity_excluded_labels": sorted({
                    str(label)
                    for row in llm_summary.values() if isinstance(row, dict)
                    for label in (((row.get("unit_sensitivity") or {}).get("unit_exclusion") or {}).get("pooled_source") or [])
                    + (((row.get("unit_sensitivity") or {}).get("unit_exclusion") or {}).get("unconfirmed_source") or [])}),
                "unit_sensitivity_analysis_routes": sorted({
                    str((row.get("unit_sensitivity") or {}).get("analysis_route") or "")
                    for row in llm_summary.values() if isinstance(row, dict)
                    and (row.get("unit_sensitivity") or {}).get("analysis_route")}),
                "unit_sensitivity_scope": [str(((row.get("unit_sensitivity") or {}).get("correction_scope") or ""))
                                           for row in llm_summary.values() if isinstance(row, dict)][:1],
            },
            confidence=_confidence_from_design_and_backend("python_fallback"),
            limitations=[f"R/rpy2 unavailable: {R_IMPORT_ERROR}"],
        )
        return llm_summary

    # limma = importr("limma")

    # Python → R
    with conversion.localconverter(converter):
        r_data_BC = conversion.py2rpy(data_BC)
        r_sampleinfo = conversion.py2rpy(sampleinfo)
        r_protein_gene_map = conversion.py2rpy(protein_gene_map)
    robjects.globalenv["data_BC"] = r_data_BC
    robjects.globalenv["sampleinfo"] = r_sampleinfo
    robjects.globalenv["protein_gene_map"] = r_protein_gene_map
    robjects.globalenv["output_folder"] = output_folder


    # ===== 原始 R 代码：完全不动 =====
    robjects.r("""
    library(limma)

    run_limma_block <- function(sampleinfo_filtered, factor_vec) {
        valid_idx <- !is.na(factor_vec)
        factor_vec <- factor_vec[valid_idx]
        sample_sub <- sampleinfo_filtered[valid_idx, ]

        # ---------- 表达矩阵 ----------

        data_Anal <- data_BC[, sample_sub$FileName]
        data_Anal <- as.data.frame(data_Anal)

        data_Anal$PG <- data_BC$PG.ProteinGroups
        data_Anal <- aggregate(. ~ PG, data = data_Anal, FUN = mean)

        rownames(data_Anal) <- data_Anal$PG
        data_Anal$PG <- NULL

        # ---------- 设计矩阵 ----------
        condition_clean <- make.names(factor_vec)

        design <- model.matrix(~0 + factor(condition_clean))
        colnames(design) <- levels(factor(condition_clean))
        rownames(design) <- colnames(data_Anal)

        cond_levels <- levels(factor(condition_clean))

        if (length(cond_levels) < 2) {
            return(NULL)
        }

        # ---------- 构建对比 ----------
        comb <- combn(cond_levels, 2)
        contrast_list <- c()

        for (i in 1:ncol(comb)) {
            contrast_list <- c(
                contrast_list,
                paste0(comb[1, i], "-", comb[2, i])
            )
        }

        contrast <- makeContrasts(
            contrasts = contrast_list,
            levels = design
        )

        # ---------- limma ----------

        fit <- lmFit(data_Anal, design)
        fit <- contrasts.fit(fit, contrast)
        fit <- eBayes(fit)

        # ---------- 输出 ----------
        dep_list <- lapply(colnames(fit$contrasts), function(con) {
            dep <- topTable(fit, coef = con, n = Inf, sort.by = "logFC")
            dep <- na.omit(dep)
        })

        names(dep_list) <- colnames(fit$contrasts)

        invisible(lapply(names(dep_list), function(con) {

            dep <- dep_list[[con]]

            dep <- cbind(
                contrast = gsub("-", "_vs_", con),
                protein = rownames(dep),
                dep
            )

            dep <- cbind(
                protein_gene_map[match(rownames(dep), protein_gene_map$PG.ProteinGroups), ],
                dep
            )

            write.csv(
                dep,
                file = file.path(output_folder, paste0("differential_", gsub("-", "_vs_", con), ".csv")),
                row.names = FALSE,
                na = ""
            )

        }))
    }

    run_block <- function(sample_sub, prefix) {
        # 表达矩阵
        data_Anal <- data_BC[, sample_sub$FileName]
        data_Anal <- as.data.frame(data_Anal)

        data_Anal$PG <- data_BC$PG.ProteinGroups
        data_Anal <- aggregate(. ~ PG, data = data_Anal, FUN = mean)

        rownames(data_Anal) <- data_Anal$PG
        data_Anal$PG <- NULL

        # design（仅用 Cluster）
        condition <- make.names(sample_sub$Cluster)
        design <- model.matrix(~0 + factor(condition))
        colnames(design) <- levels(factor(condition))
        rownames(design) <- colnames(data_Anal)

        cond_levels <- levels(factor(condition))
        if (length(cond_levels) < 2) return(NULL)

        # 两两对比
        comb <- combn(cond_levels, 2)
        contrast_list <- c()
        contrast_prefix <- c()

        for (i in 1:ncol(comb)) {
            contrast_list <- c(contrast_list, paste0(comb[1,i], "-", comb[2,i]))
            contrast_prefix <- c(contrast_prefix, paste0(prefix, "_", comb[1,i], "-", comb[2,i]))
        }

        contrast <- makeContrasts(contrasts = contrast_list, levels = design)

        fit <- lmFit(data_Anal, design)
        fit <- contrasts.fit(fit, contrast)
        fit <- eBayes(fit)

        dep_list <- lapply(colnames(fit$contrasts), function(con) {
            dep <- topTable(fit, coef = con, n = Inf, sort.by = "logFC")
            na.omit(dep)
        })
        names(dep_list) <- colnames(fit$contrasts)

        invisible(lapply(seq_along(dep_list), function(i) {
            dep <- dep_list[[i]]

            dep <- cbind(
                contrast = gsub("-", "_vs_", contrast_prefix[i]),
                protein = rownames(dep),
                dep
            )

            dep <- cbind(
                protein_gene_map[match(rownames(dep), protein_gene_map$PG.ProteinGroups), ],
                dep
            )

            write.csv(
                dep,
                file = file.path(output_folder, paste0("differential_", gsub("-", "_vs_", contrast_prefix[i]), ".csv")),
                row.names = FALSE,
                na = ""
            )
        }))
    }

    # ---------- 主流程 ----------
    valid_idx <- !is.na(sampleinfo$Cluster)
    sampleinfo_filtered <- sampleinfo[valid_idx, ]


    # ---- 自动识别 Type 列 ----
    type_cols <- grep("^Type[0-9]+$", colnames(sampleinfo_filtered), value = TRUE)

    has_type <- length(type_cols) > 0

    # ---- 1) 按 Type  ----
    if (has_type) {
        for (type_col in type_cols) {

            type_values <- unique(sampleinfo_filtered[[type_col]])
            for (val in type_values) {
                sub_idx <- sampleinfo_filtered[[type_col]] == val
                sample_sub <- sampleinfo_filtered[sub_idx, ]
                # prefix <- paste0(type_col, "_", val)
                run_block(sample_sub, val)
            }

            # limma：在该 Type 下，基于 Cluster 做对比
            run_limma_block(sampleinfo_filtered, sampleinfo_filtered[[type_col]])
        }
    }

    # ---- 2) fallback：没有 Type 列 ----
    if (!has_type) {
    run_block(sampleinfo_filtered, "ALL")

    # # 直接用 Cluster 做 limma
    # run_limma_block(sampleinfo_filtered, sampleinfo_filtered$Cluster)
    }
    """)


    # ===== Python：为 LLM 整理返回内容 =====
    allowed_contrasts = {spec.get("name") for spec in selected_contrasts} if selected_contrasts else set()
    llm_summary = {}
    for fn in os.listdir(output_folder):
        if not fn.startswith("differential_") or not fn.endswith(".csv"):
            continue
        contrast = fn.replace("differential_", "").replace(".csv", "")
        if allowed_contrasts and contrast not in allowed_contrasts and contrast.replace("ALL_", "") not in allowed_contrasts:
            try:
                os.remove(os.path.join(output_folder, fn))
            except OSError:
                pass
            continue
        parameters[contrast] = {"differential_protein_file": os.path.join(output_folder, fn), "contrast_type": contrast}
        df = pd.read_csv(os.path.join(output_folder, fn))
        # summary
        diff_cfg = _get_analysis_design().get("differential", {}) or {}
        p_thresh_summary = float(diff_cfg.get("p_thresh", 0.05) or 0.05)
        logfc_thresh_summary = float(diff_cfg.get("logfc_thresh", 0.25) or 0.25)
        sig = df[df["adj.P.Val"] < p_thresh_summary]
        sig_design = sig[sig["logFC"].abs() > logfc_thresh_summary]
        llm_summary[contrast] = {
            "contrast": contrast,
            "n_proteins": round(float(df.shape[0]), 4),
            "median_logFC_sig": round(sig_design["logFC"].median(), 4) if not sig_design.empty else 0
            }

        sig = df[df["adj.P.Val"] < p_thresh_summary]
        sig_design = sig[sig["logFC"].abs() > logfc_thresh_summary]

        llm_summary[contrast].update({
            "method": "R limma empirical Bayes",
            "statistical_backend": "r_limma",
            "evidence_source": "current_matrix",
            "group_col": group_col,
            "analysis_design_source": _get_analysis_design().get("source", "unknown"),
            "sanity_checks": {
                "n_proteins_before_combat_calibration": int(data_BC.shape[0]),
                "n_proteins_after_combat_calibration": int(df.shape[0]),
                "n_significant": int(sig_design.shape[0]),
                "n_up": int((sig_design["logFC"] > 0).sum()),
                "n_down": int((sig_design["logFC"] < 0).sum()),
                "thresholds_used": {
                    "adj_p": p_thresh_summary,
                    "abs_logFC": logfc_thresh_summary,
                },
            },
            # "columns_in_output": list(df.columns)
        })

    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)
    limma_summary_path = os.path.join(output_folder, "limma_summary.json")

    save_json(llm_summary, limma_summary_path)
    save_json(parameters, parameters_path)
    append_evidence(
        this_run_folder,
        source="current_matrix",
        tool="run_pairwise_limma",
        claim="Differential protein analysis completed with R limma; contrasts were filtered by analysis_design when provided.",
        files=[limma_summary_path],
        metrics={
            "statistical_backend": "r_limma",
            "group_col": group_col,
            "n_contrasts": len(llm_summary),
            "design_contrasts_used": bool(selected_contrasts),
        },
        confidence=_confidence_from_design_and_backend("r_limma"),
    )
    return llm_summary




@tool
def detect_batch_and_doublets(
    *args, **kwargs
):
    """
    检测潜在批次效应并识别异常样本（doublets / 污染）。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(processed_proteins_folder, exist_ok=True)
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')

    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if os.path.exists(protein_quant_combat_path):
        data_BC = pd.read_csv(protein_quant_combat_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)


    data_BC = data_BC.loc[:, sampleinfo["FileName"]]

    # ===== QC metrics =====
    qc = pd.DataFrame(index=data_BC.columns)
    qc["n_detected_proteins"] = data_BC.notna().sum()
    qc["missing_rate"] = data_BC.isna().mean()
    qc["total_intensity"] = np.nansum(data_BC.values, axis=0)
    # 1. TIC（Total Ion Current，等价于 total_intensity，但保留显式语义）
    qc["TIC"] = qc["total_intensity"]

    # 2. 前体数（近似：非缺失蛋白数，可作为 proxy）
    qc["n_precursors_proxy"] = qc["n_detected_proteins"]

    # 4. 内源 / 外源标记占比（基于 SampleInfo 中的 Type / Label 信息）
    if "Type" in qc.columns:
        qc["is_external_control"] = qc["Type"].astype(str).str.contains(
            "control|spike|std", case=False, regex=True
        )
    else:
        qc["is_external_control"] = False

    # 5. 简单阈值标记（不直接过滤，只打 flag）
    qc["qc_flag_low_protein"] = qc["n_detected_proteins"] < 300
    # qc["qc_flag_high_mito"] = qc["mito_fraction"] > 0.2

    low_TIC_quantile = 0.05
    low_TIC_threshold = qc["TIC"].quantile(low_TIC_quantile)

    qc["qc_flag_low_TIC"] = qc["TIC"] < low_TIC_threshold
    low_TIC_samples = qc.index[qc["qc_flag_low_TIC"]].tolist()

    qc = qc.join(sampleinfo.set_index("FileName"))

    sampleinfo_qc = sampleinfo.set_index("FileName").join(
        qc[["qc_flag_low_TIC", "qc_flag_low_protein"]],
        how="left"
    )
    sampleinfo_qc = sampleinfo_qc.reset_index()

    sampleinfo_qc.to_csv(
        f"{output_folder}/sampleinfo_with_qc_flags.csv",
        index=False
    )

    # ===== PCA (post-ComBat residual structure) =====
    X = data_BC.fillna(data_BC.median(axis=1))

    # Step 2: 对仍然存在 NaN 的位置（全缺失行），用 0 或全局中位数
    X = X.fillna(0)

    X = StandardScaler().fit_transform(X.T)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)

    for i in range(3):
        qc[f"PC{i+1}"] = pcs[:, i]

    # ===== 按 Date / Type / Group 可视化 =====
    for col in ["Date", "Type", "Group"]:
        if col in qc.columns:
            plt.figure(figsize=(6, 5))
            for v in qc[col].astype(str).unique():
                idx = qc[col].astype(str) == v
                plt.scatter(qc.loc[idx, "PC1"], qc.loc[idx, "PC2"], label=v, s=35)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"PCA colored by {col}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            _save_figure(f"{output_folder}/pca_by_{col}.png", dpi=150)

    # ===== Quantify batch association =====
    batch_association = {}

    for col in ["Date", "Type", "Group"]:
        if col not in qc.columns:
            continue

        if qc[col].dtype.kind in "if":  # numeric
            corr, pval = spearmanr(qc["PC1"], qc[col], nan_policy="omit")
            batch_association[col] = {
                "PC1_spearman_r": round(float(corr), 4),
                "p_value": round(float(pval), 4),
                "interpretation": "continuous metadata correlation"
            }
        else:
            # categorical: variance explained (R^2)
            groups = qc[col].astype(str)
            pc1 = qc["PC1"]

            overall_var = np.var(pc1)
            group_means = pc1.groupby(groups).mean()
            explained = np.var(group_means.reindex(groups).values)

            batch_association[col] = {
                "PC1_variance_explained": round(float(explained / overall_var), 4),
                "interpretation": "fraction of PC1 variance explained by category"
            }

    # ===== Outlier / doublet detection =====
    z_metrics = {}
    for m in ["n_detected_proteins", "missing_rate", "total_intensity"]:
        z = (qc[m] - qc[m].mean()) / qc[m].std()
        z_metrics[m] = z

    qc["outlier_score"] = np.sqrt(
        z_metrics["n_detected_proteins"]**2 +
        z_metrics["missing_rate"]**2 +
        z_metrics["total_intensity"]**2
    )

    flagged = qc[qc["outlier_score"] > 4]

    outlier_details = {}
    for s in flagged.index:
        contributions = {
            m: float(z_metrics[m].loc[s]) for m in z_metrics
        }
        outlier_details[s] = {
            "outlier_score": float(flagged.loc[s, "outlier_score"]),
            "z_contributions": contributions,
            "suspected_pattern": (
                "potential_doublet"
                if contributions["total_intensity"] > 2
                else "low_quality_or_missing"
            )
        }
    # ===== 保存 =====
    qc.to_csv(f"{output_folder}/qc_metrics.csv")
    flagged.to_csv(f"{output_folder}/flagged_samples.csv")
    # ===== LLM-friendly summary =====
    llm_summary = {
        "file_name": "ProteinQuant_ComBat.csv",
        "n_samples": int(qc.shape[0]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "batch_association_with_PC1": batch_association,
        "outlier_summary": {
            "n_outlier_samples": int(flagged.shape[0]),
            "outlier_sample_ids": list(flagged.index),
            "outlier_details": outlier_details
        },
        "qc_statistics": {
            "mean_detected_proteins": round(float(qc["n_detected_proteins"].mean()), 4),
            "mean_missing_rate": round(float(qc["missing_rate"].mean()), 4),
            "mean_total_intensity": round(float(qc["total_intensity"].mean()), 4)
        }
    }

    return llm_summary

@tool
def prepare_pseudobulk(
    top_var_proteins=20,
    *args, **kwargs
):
    """
    基于样本分组信息构建 pseudobulk 蛋白定量矩阵。

    参数:
        top_var_proteins:
            返回给 LLM 的高变异蛋白数量，用于概括表达差异。默认值为 20。
    """
    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(processed_proteins_folder, exist_ok=True)
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')

    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if os.path.exists(protein_quant_combat_path):
        data_BC = pd.read_csv(protein_quant_combat_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    group_col = "Cluster"
    if group_col not in sampleinfo.columns:
        processed_proteins_folder = os.path.join(resolve_path('this_run_folder_path'), 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
        else:
            run_umap.func()
            if os.path.exists(sampleinfo_with_cluster_path):
                sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
            else:
                raise ValueError(f"Neither '{group_col}' column found in sampleinfo nor sampleinfo_with_clusters.csv generated. Please check the UMAP clustering step.")

    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)

    group_cols=("Cluster",)

    # 只保留 sampleinfo 中定义的样本顺序
    data_BC = data_BC.loc[:, sampleinfo["FileName"]]

    # ---- 1. 构建 pseudobulk 元数据 ----
    meta = sampleinfo.copy()
    meta["pseudobulk_id"] = meta[list(group_cols)].astype(str).agg("_".join, axis=1)

    pb_matrix = {}
    pb_var = {}   # 新增：组内方差 proxy
    pb_meta = []

    for pid, sub in meta.groupby("pseudobulk_id"):
        samples = sub["FileName"].tolist()
        mat = data_BC[samples]

        # pseudobulk mean
        pb_matrix[pid] = np.nanmean(mat.values, axis=1)

        # 新增：组内变异（用于不平衡诊断）
        pb_var[pid] = np.nanvar(mat.values, axis=1, ddof=1)

        pb_meta.append({
            "pseudobulk_id": pid,
            "n_samples": len(samples),
            "effective_weight": float(len(samples)),  # 显式权重
            **{c: sub[c].iloc[0] for c in group_cols}
        })

    pb_matrix = pd.DataFrame(pb_matrix, index=data_BC.index)
    pb_var = pd.DataFrame(pb_var, index=data_BC.index)
    pb_meta = pd.DataFrame(pb_meta).set_index("pseudobulk_id")

    median_n = pb_meta["n_samples"].median()

    pb_meta["sample_size_imbalance_flag"] = (
        pb_meta["n_samples"] < 0.5 * median_n
    )


    # ---- 2. 保存 ----
    pb_matrix.to_csv(f"{output_folder}/pseudobulk_matrix.csv")
    pb_meta.to_csv(f"{output_folder}/pseudobulk_metadata.csv")


    # ---- 3. 设计信息 ----
    design_summary = {
        "n_pseudobulk": pb_meta.shape[0],
        "grouping_rule": list(group_cols),
        "pseudobulks": [
            {
                "id": pid,
                "n_samples": int(pb_meta.loc[pid, "n_samples"]),
                **{
                    c: str(pb_meta.loc[pid, c])
                    for c in group_cols
                }
            }
            for pid in pb_meta.index
        ]
    }

    # ---- 4. 表达整体轮廓 ----
    expr_overview = {
        pid: {
            "mean_expression": float(pb_matrix[pid].mean()),
            "std_expression": float(pb_matrix[pid].std())
        }
        for pid in pb_matrix.columns
    }

    # ---- 5. 全局最变异蛋白 ----
    protein_variance = pb_matrix.var(axis=1).sort_values(ascending=False)
    top_variable = [
        {
            "protein": str(pid),
            "variance": round(float(var), 4)
        }
        for pid, var in protein_variance.head(top_var_proteins).items()
    ]

    # ---- 6. 合法的 pairwise contrast 空间 ----
    contrast_space = [
        f"{a} vs {b}"
        for a, b in combinations(pb_matrix.columns, 2)
    ]

    llm_summary = {
        "design": design_summary,
        "expression_overview": expr_overview,
        "top_variable_proteins": top_variable,
        "contrast_space": contrast_space
    }
    sample_balance_summary = {
        "n_samples_per_pseudobulk": {
            pid: int(pb_meta.loc[pid, "n_samples"])
            for pid in pb_meta.index
        },
        "min_samples": int(pb_meta["n_samples"].min()),
        "max_samples": int(pb_meta["n_samples"].max()),
        "imbalance_ratio_max_to_min": round(
            float(pb_meta["n_samples"].max() / pb_meta["n_samples"].min()), 2
        ),
        "imbalance_flagged_pseudobulks": pb_meta.index[
            pb_meta["sample_size_imbalance_flag"]
        ].tolist(),
    }
    llm_summary["sample_balance_audit"] = sample_balance_summary

    # ---- 7. 返回：程序用 + LLM 用 ----

    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)
    pseudobulk_path = os.path.join(output_folder, "pseudobulk.json")

    save_json(llm_summary, pseudobulk_path)

    return llm_summary


def _umap_clustering(this_run_folder, contrast_type, sampleinfo, data_BC, cluster_col, random_state=50, logfc_cut=1.0, pval_cut=0.05, top_n_markers=20, file_name='umap'):

    sampleinfo_plot = sampleinfo.copy()
    sample_cols = [col for col in sampleinfo_plot["FileName"] if col in data_BC.columns]
    if not sample_cols:
        raise ValueError("No sample columns from sampleinfo['FileName'] were found in data_BC")

    sampleinfo_plot = (
        sampleinfo_plot[sampleinfo_plot["FileName"].isin(sample_cols)]
        .drop_duplicates(subset=["FileName"])
        .set_index("FileName")
        .loc[sample_cols]
        .reset_index()
    )

    data_mat = data_BC.set_index("PG.ProteinGroups")[sample_cols]
    protein_names = data_mat.index.to_numpy()
    X_raw = data_mat.T.fillna(0).to_numpy()

    # Focus UMAP on informative proteins instead of all proteins equally.
    if X_raw.shape[1] > 200:
        var = np.var(X_raw, axis=0)
        top_n_var = min(2000, max(200, X_raw.shape[0] * 50), X_raw.shape[1])
        top_idx = np.argsort(var)[-top_n_var:]
        X_use = X_raw[:, top_idx]
        protein_names = protein_names[top_idx]
    else:
        X_use = X_raw

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_use)

    n_samples = X_scaled.shape[0]
    n_features = X_scaled.shape[1]
    n_pcs = min(20, max(1, n_samples - 1), n_features)
    pca = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.35,
        metric="cosine",
        init="spectral",
        random_state=random_state
    )
    embedding = reducer.fit_transform(X_pca)

    umap_df = pd.DataFrame(
        embedding,
        columns=["umap.1", "umap.2"],
        index=sampleinfo_plot["FileName"]
    )

    clusters = sampleinfo_plot[cluster_col]
    clusters = np.array(["" if str(c) == "nan" else f"{c}" for c in clusters])

    markers = []
    for c in np.unique(clusters):
        if c == "":
            continue
        idx1 = clusters == c
        idx2 = clusters != c
        group1 = X_use[idx1]
        group2 = X_use[idx2]
        _, p = ttest_ind(group1, group2, axis=0, nan_policy="omit", equal_var=False)
        logfc = group1.mean(axis=0) - group2.mean(axis=0)
        df = pd.DataFrame({
            "protein": protein_names,
            "cluster": c,
            "logFC": logfc,
            "pvalue": p
        })
        df["padj"] = multipletests(df["pvalue"], method="fdr_bh")[1]
        df = df[(df["logFC"].abs() > logfc_cut) & (df["padj"] < pval_cut)]
        markers.append(df)

    markers_df = pd.concat(markers, ignore_index=True) if markers else pd.DataFrame()
    cluster_markers = {}
    if not markers_df.empty:
        for c in markers_df["cluster"].unique():
            sub = markers_df[markers_df["cluster"] == c]
            sub = sub.sort_values(["logFC", "padj"], ascending=[False, True])
            cluster_markers[c] = sub.head(top_n_markers)["protein"].tolist()

    sampleinfo_plot = sampleinfo_plot.set_index("FileName")
    umap_data = pd.concat([sampleinfo_plot[[cluster_col]], umap_df], axis=1, join="inner").reset_index(names="sample")

    centers = (
        umap_data.groupby(cluster_col)[["umap.1", "umap.2"]]
        .mean()
        .rename(columns=lambda x: x + ".center")
    )
    umap_data = umap_data.join(centers, on=cluster_col)

    umap_data["dist_to_center"] = np.sqrt(
        (umap_data["umap.1"] - umap_data["umap.1.center"]) ** 2 +
        (umap_data["umap.2"] - umap_data["umap.2.center"]) ** 2
    )

    cluster_compactness = (
        umap_data.groupby(cluster_col)["dist_to_center"]
        .mean()
        .reset_index(name="mean_radius")
    )

    center_coords = centers.values
    cluster_separation = pd.DataFrame(
        squareform(pdist(center_coords)),
        index=centers.index,
        columns=centers.index
    )

    cluster_order = sorted(
        [str(c) for c in umap_data[cluster_col].dropna().unique()],
        key=lambda x: (not str(x).isdigit(), str(x))
    )
    n_clusters = len(cluster_order)
    if n_clusters <= 10:
        palette_colors = sns.color_palette("tab10", n_colors=n_clusters)
    elif n_clusters <= 20:
        palette_colors = sns.color_palette("tab20", n_colors=n_clusters)
    else:
        palette_colors = sns.color_palette("husl", n_colors=n_clusters)
    palette = dict(zip(cluster_order, palette_colors))

    _set_visual_theme("talk")
    fig, ax = plt.subplots(figsize=(10.5, 8))

    for cluster in cluster_order:
        sub = umap_data[umap_data[cluster_col].astype(str) == cluster]
        color = palette[cluster]
        ax.scatter(
            sub["umap.1"],
            sub["umap.2"],
            label=cluster,
            s=110,
            alpha=0.9,
            color=color,
            edgecolor="white",
            linewidth=0.8
        )

        if len(sub) >= 3:
            try:
                draw_confidence_ellipse(
                    sub["umap.1"].values,
                    sub["umap.2"].values,
                    ax=ax,
                    n_std=1.8,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.12,
                    linewidth=1.5
                )
            except Exception:
                pass

        cx = sub["umap.1"].mean()
        cy = sub["umap.2"].mean()
        ax.scatter(
            cx,
            cy,
            s=260,
            marker="X",
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=5
        )
        ax.text(
            cx,
            cy,
            str(cluster),
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                linewidth=1,
                alpha=0.95
            ),
            zorder=6
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of Samples by {cluster_col}", fontsize=18, pad=14, weight="bold")
    _apply_nature_axes(ax)
    ax.legend(
        title=cluster_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True
    )

    output_folder = _ensure_dir(os.path.join(this_run_folder, "visualize_results"))
    img_path = os.path.join(output_folder, f"umap_{file_name}.png")
    _save_figure(img_path, fig=fig, dpi=300, bbox_inches="tight")

    summary = {
        "cluster_compactness": cluster_compactness.to_dict(orient='dict'),
        "cluster_separation": cluster_separation.to_dict(orient='dict'),
        "cluster_number": int(len(set(clusters)) - (1 if -1 in clusters else 0)),
        "cluster_markers": cluster_markers,
        "umap_parameters": {
            "n_neighbors": int(n_neighbors),
            "min_dist": 0.35,
            "metric": "cosine",
            "n_pcs": int(n_pcs),
            "n_variable_proteins": int(X_use.shape[1])
        }
    }

    cluster_stability_proxy = (
        cluster_compactness
        .assign(
            relative_compactness=lambda df:
            df["mean_radius"] / df["mean_radius"].median()
        )
    )
    summary["cluster_stability_proxy"] = {
        "per_cluster": cluster_stability_proxy
            .set_index(cluster_col)
            .to_dict(orient="index")
    }

    summary = _round_dict(summary)

    return summary


@tool
def run_umap(
    random_state: int = 42,
    logfc_cut: float = 1.0,
    pval_cut: float = 0.05,
    top_n_markers:int = 50,
    *args, **kwargs
):
    """
    基于批次校正后的蛋白定量矩阵执行 UMAP 降维可视化。

    参数:
        random_state:
            UMAP 随机种子，用于保证降维结果的可复现性。默认值为 50。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')


    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(processed_proteins_folder, exist_ok=True)
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')

    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()

    if os.path.exists(protein_quant_combat_path):
        data_BC = pd.read_csv(protein_quant_combat_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    data_BC, sampleinfo = preprocess_data_and_sampleinfo(data_BC, sampleinfo)

    column_names = sampleinfo.columns
    has_group = [col for col in column_names if col.startswith("Type")]
    llm_summary = {}
    for col in has_group:
        for contrast_type in set(sampleinfo[col].values) :
            sampleinfo_sub = sampleinfo[sampleinfo[col] == contrast_type]
            sample_names = sampleinfo_sub["FileName"].tolist()
            data_BC_sub = data_BC[["PG.ProteinGroups"] + sample_names]
            sampleinfo_sub["FileName"] = sample_names
            file_name = f'Group_{contrast_type}_for_Cluster'
            llm_summary[file_name] = _umap_clustering(this_run_folder=this_run_folder, contrast_type=contrast_type, sampleinfo=sampleinfo_sub,
                             data_BC=data_BC_sub, cluster_col="Cluster", random_state=random_state,
                             logfc_cut=logfc_cut, pval_cut=pval_cut, top_n_markers=top_n_markers, file_name=file_name)

        sampleinfo_sub = sampleinfo[~sampleinfo[col].isna()]
        sample_names = sampleinfo_sub["FileName"].tolist()
        data_BC_sub = data_BC[["PG.ProteinGroups"] + sample_names]
        sampleinfo_sub["FileName"] = sample_names
        temp_name = '-'.join(list(set(sampleinfo[col].values)))
        file_name = f'Group_{temp_name}'
        llm_summary[file_name] = _umap_clustering(this_run_folder=this_run_folder, contrast_type='', sampleinfo=sampleinfo_sub,
                            data_BC=data_BC_sub, cluster_col=col, random_state=random_state,
                            logfc_cut=logfc_cut, pval_cut=pval_cut, top_n_markers=top_n_markers, file_name=file_name)

    has_cluster = [col for col in column_names if col.startswith("Cluster")]
    for col in has_cluster:
        sampleinfo_sub = sampleinfo[~sampleinfo[col].isna()]
        sample_names = sampleinfo_sub["FileName"].tolist()
        data_BC_sub = data_BC[["PG.ProteinGroups"] + sample_names]
        file_name = f'Cluster'
        llm_summary[file_name] = _umap_clustering(this_run_folder=this_run_folder, contrast_type="Cluster", sampleinfo=sampleinfo,
                             data_BC=data_BC_sub, cluster_col=col, random_state=random_state,
                             logfc_cut=logfc_cut, pval_cut=pval_cut, top_n_markers=top_n_markers, file_name=file_name)
        for cluster_type in set(sampleinfo[col].values):
            sampleinfo_cluster_sub = sampleinfo[sampleinfo[col] == cluster_type]
            sample_names = sampleinfo_cluster_sub["FileName"].tolist()
            data_BC_cluster_sub = data_BC[["PG.ProteinGroups"] + sample_names]
            for type_col in has_group:
                group_col = '-'.join(list(set(sampleinfo[type_col].values)))
                file_name = f'Cluster_{cluster_type}_for_{group_col}'
                llm_summary[file_name] = _umap_clustering(this_run_folder=this_run_folder, contrast_type='', sampleinfo=sampleinfo_cluster_sub,
                                data_BC=data_BC_cluster_sub, cluster_col=type_col, random_state=random_state,
                                logfc_cut=logfc_cut, pval_cut=pval_cut, top_n_markers=top_n_markers, file_name=file_name)


    output_folder = os.path.join(this_run_folder, 'umap_results')
    os.makedirs(output_folder, exist_ok=True)
    umap_path = os.path.join(output_folder, "umap.json")

    save_json(llm_summary, umap_path)

    return llm_summary




def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@tool
def load_ori_proteinquant_for_llm(
    *args, **kwargs
):
    """
    加载原始的蛋白质定量矩阵并生成其元数据摘要，供LLM使用。
    """

    protein_quant_path = resolve_path('protein_quant_path')

    round_digits = 2
    # ===== 基本维度 =====
    if not os.path.exists(protein_quant_path):
        return {"responses": f"Input file path does not exist: {protein_quant_path}"}

    df = _read_csv_checked(protein_quant_path, description="protein quantification file")
    annotation_cols = ["PG.ProteinGroups", "PG.Genes"]
    quant_cols = [c for c in df.columns if c not in annotation_cols]

    summary = {
        "dataset_type": "protein_quantification_matrix",
        "dimensions": {
            "n_proteins": int(df.shape[0] - 1), #减去表头行
            "n_samples": int(len(quant_cols))
        },
        "annotation_columns": list(annotation_cols)
    }

    # ===== 样本分组（从列名推断）=====
    groups = {}
    for c in quant_cols:
        group = c.split("_")[0]
        groups[group] = groups.get(group, 0) + 1

    summary["sample_groups"] = groups

    # ===== 缺失结构 =====
    quant_df = df[quant_cols]

    global_missing = quant_df.isna().mean().mean()

    per_protein_missing = quant_df.isna().mean(axis=1)
    per_sample_missing = quant_df.isna().mean(axis=0)

    summary["missingness"] = {
        "global_missing_ratio": round(global_missing, round_digits),
        "per_protein_missing_ratio": {
            "median": round(per_protein_missing.median(), round_digits),
            "p90": round(per_protein_missing.quantile(0.9), round_digits),
            "p95": round(per_protein_missing.quantile(0.95), round_digits)
        },
        "per_sample_missing_ratio": {
            "median": round(per_sample_missing.median(), round_digits),
            "p90": round(per_sample_missing.quantile(0.9), round_digits)
        }
    }

    # ===== 检测覆盖度 =====
    detected_counts = (1 - per_protein_missing)

    summary["detection_summary"] = {
        "proteins_detected_in_>50%_samples": int((detected_counts > 0.5).sum()),
        "proteins_detected_in_<10%_samples": int((detected_counts < 0.1).sum())
    }

    # ===== 数值分布（忽略 NaN）=====
    values = quant_df.values.flatten()
    # values = values[~np.isnan(values)]
    values = [
        v for v in values
        if isinstance(v, (int, float)) and not np.isnan(v)
    ]
    summary["value_distribution"] = {
        "numeric_columns": {
            "min": round(float(np.min(values)), round_digits),
            "max": round(float(np.max(values)), round_digits),
            "median": round(float(np.median(values)), round_digits)
        }
    }

    return summary

@tool
def load_proteinquant_combat_for_llm(
    *args, **kwargs
):
    """
    加载ComBat校正的蛋白质定量矩阵并生成其元数据摘要，供LLM使用。
    """
    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = _build_output_dir(this_run_folder, 'processed_proteins')
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')

    sampleinfo_path = resolve_path('sampleinfo_path')

    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if not os.path.exists(protein_quant_combat_path):
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")


    row_id_col = "PG.ProteinGroups"
    round_digits = 2

    df = _read_csv_checked(protein_quant_combat_path, description="ComBat protein quantification file")

    if os.path.exists(sampleinfo_path):
        sampleinfo = _read_csv_checked(sampleinfo_path, description="sampleinfo file")
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    group_col = "Cluster"
    if group_col not in sampleinfo.columns:
        processed_proteins_folder = os.path.join(resolve_path('this_run_folder_path'), 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = _read_csv_checked(sampleinfo_with_cluster_path, description="sampleinfo_with_clusters file")
        else:
            run_umap.func()
            if os.path.exists(sampleinfo_with_cluster_path):
                sampleinfo = _read_csv_checked(sampleinfo_with_cluster_path, description="sampleinfo_with_clusters file")
            else:
                raise ValueError(f"Neither '{group_col}' column found in sampleinfo nor sampleinfo_with_clusters.csv generated. Please check the UMAP clustering step.")

    if row_id_col not in df.columns:
        raise ValueError(f"Missing row identifier column: {row_id_col}")

    quant_cols = [c for c in df.columns if c != row_id_col]

    # ===== 基本维度 =====
    summary = {
        "dimensions": {
            "n_proteins": int(df.shape[0]),
            "n_samples": int(len(quant_cols))
        },
        "row_identifier": row_id_col
    }

    # ===== 样本分组（从列名前缀推断）=====
    groups = {}
    for c in quant_cols:
        group = sampleinfo.loc[sampleinfo["FileName"] == c, "Cluster"].values
        if len(group) > 0:
            group = str(group[0])
            if group == '-1':
                continue
            groups[group] = groups.get(group, 0) + 1

    summary["sample_groups"] = groups

    # ===== 数值特征 =====
    values = df[quant_cols].values.flatten()
    values = values[~np.isnan(values)]

    summary["value_characteristics"] = {
        "contains_missing_values": bool(df[quant_cols].isna().any().any())
    }

    if not summary["value_characteristics"]["contains_missing_values"]:
        summary["value_characteristics"]["missing_value_handling"] = "Left-skewed distribution"
    else:
        summary["value_characteristics"]["missing_value_handling"] = "No processing applied"

    summary["value_distribution"] = {
        "min": round(float(values.min()), round_digits),
        "max": round(float(values.max()), round_digits),
        "median": round(float(np.median(values)), round_digits)
    }


    return summary

def _is_annotation_column(name: str, sample_ids) -> bool:
    """True for matrix annotation columns (protein ids, gene names, descriptions); a name that matches a SampleInfo identifier is never treated as annotation."""
    text = str(name or "").strip()
    if not text or text in sample_ids:
        return False
    low = text.lower()
    if low.startswith(("protein.", "pg.", "first.protein.")):
        return True
    parts = [part for part in re.split(r"[._\s]+", low) if part]
    words = {"protein", "proteins", "gene", "genes", "pg"}
    head = parts[0] if parts else ""
    if head in words:
        return True
    return head == "first" and "protein" in parts


@tool
def align_samples_from_paths(
    *args, **kwargs
    ):
    """
    从文件路径加载 ProQuant、ComBat、SampleInfo，
    对齐样本 ID，构建完整、可审计的 sample mapping 表，
    并返回 LLM 友好的结构化摘要。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    processed_proteins_folder = _build_output_dir(this_run_folder, 'processed_proteins')
    protein_quant_combat_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')

    if not os.path.exists(protein_quant_combat_path):
        print("ComBat corrected protein quantification file not found. Try to run batch correction first.")
        _ = combat_calibration.func()
    if not os.path.exists(protein_quant_combat_path):
        raise FileNotFoundError(f"The file of {protein_quant_combat_path} is not found.")

    # ===============================
    # 0. 基础检查
    # ===============================
    sampleinfo_id_col = "FileName"

    for p in [protein_quant_path, protein_quant_combat_path, sampleinfo_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # ===============================
    # 1. 加载数据
    # ===============================
    proquant_df = _read_csv_checked(protein_quant_path, description="protein quantification file", index_col=0)
    combat_df = _read_csv_checked(protein_quant_combat_path, description="ComBat protein quantification file", index_col=0)
    sampleinfo = _read_csv_checked(sampleinfo_path, description="sampleinfo file")

    now = pd.Timestamp.now().isoformat()

    # ===============================
    # 2. 初始化 mapping（以 ProQuant2 为源头）
    # ===============================
    known_sample_ids = set(sampleinfo[sampleinfo_id_col].astype(str)) if sampleinfo_id_col in sampleinfo.columns else set()
    column_names = [str(c) for c in proquant_df.columns]
    column_roles = {name: ("annotation" if _is_annotation_column(name, known_sample_ids) else "sample")
                    for name in column_names}
    annotation_columns = [name for name in column_names if column_roles[name] == "annotation"]
    sample_columns = [name for name in column_names if column_roles[name] == "sample"]

    mapping = pd.DataFrame({
        "raw_sample_id": column_names,
        "proquant_id": column_names,
        "combat_sample_id": column_names,
        "sampleinfo_id": None,
        "final_sample_id": column_names,
        "role": [column_roles[name] for name in column_names],
        "status": ["excluded" if column_roles[name] == "annotation" else "kept"
                  for name in column_names],
        "drop_reason": ["annotation_column" if column_roles[name] == "annotation" else None
                        for name in column_names],
        "merge_into": None,
        "stage": "ProQuant2",
        "timestamp": now,
        "note": None
    })

    # ===============================
    # 3. 对齐 ComBat
    # ===============================
    combat_cols = set(combat_df.columns)

    not_in_combat = (~mapping["combat_sample_id"].isin(combat_cols)) & (mapping["role"] == "sample")
    if not_in_combat.any():
        mapping.loc[not_in_combat, "status"] = "dropped"
        mapping.loc[not_in_combat, "drop_reason"] = "not_found_in_combat"
        mapping.loc[not_in_combat, "stage"] = "ComBat_align"
        mapping.loc[not_in_combat, "timestamp"] = now

    # ===============================
    # 4. 对齐 SampleInfo
    # ===============================
    sampleinfo_ids = set(sampleinfo[sampleinfo_id_col])

    not_in_sampleinfo = (~mapping["combat_sample_id"].isin(sampleinfo_ids)) & (mapping["role"] == "sample")

    # 成功对齐的
    ok_mask = ~not_in_sampleinfo & (mapping["status"] == "kept")
    mapping.loc[ok_mask, "sampleinfo_id"] = mapping.loc[ok_mask, "combat_sample_id"]

    # 失败的（只标记，不静默消失）
    fail_mask = not_in_sampleinfo & (mapping["status"] == "kept")
    mapping.loc[fail_mask, "status"] = "dropped"
    mapping.loc[fail_mask, "drop_reason"] = "not_found_in_sampleinfo"
    mapping.loc[fail_mask, "stage"] = "SampleInfo_align"
    mapping.loc[fail_mask, "timestamp"] = now

    # ===============================
    # 5. 构建 LLM 友好摘要
    # ===============================
    llm_summary = {
        "input_files": {
            "protein_quant_path": protein_quant_path,
            "protein_quant_combat_path": protein_quant_combat_path,
            "sampleinfo_path": sampleinfo_path,
        },
        "sample_counts": {
            "proquant_columns": int(proquant_df.shape[1]),
            "proquant_samples": int(len(sample_columns)),
            "annotation_columns": int(len(annotation_columns)),
            "annotation_column_names": annotation_columns,
            "combat_samples": int(combat_df.shape[1]),
            "sampleinfo_entries": int(sampleinfo.shape[0]),
            "final_kept_samples": int((mapping["status"] == "kept").sum()),
            "dropped_samples": int((mapping["status"] == "dropped").sum()),
            "excluded_annotation_columns": int((mapping["status"] == "excluded").sum()),
        },
        "drop_reasons": (
            mapping[mapping["status"] == "dropped"]
            .groupby("drop_reason")
            .size()
            .to_dict()
        ),
        "stages_encountered": mapping["stage"].unique().tolist(),
        "mapping_table_schema": mapping.columns.tolist(),
    }

    output_folder = os.path.join(this_run_folder, "processed_proteins")
    os.makedirs(output_folder, exist_ok=True)
    mapping_path = os.path.join(output_folder, "sample_mapping_table.csv")
    mapping.to_csv(mapping_path, index=False)

    return llm_summary

@tool
def report_replication_and_power(
    *args, **kwargs
):
    """
    在缺失生物学重复 / 技术重复 / 批次元数据的情况下，
    对每个 Cluster 明确报告：
      - 可观测样本数（cells / runs）
      - 生物学重复数是否可判定
      - 样本独立性假设
      - 统计功效的不可评估性与其后果

    """

    sampleinfo_path = resolve_path('sampleinfo_path')

    if not os.path.exists(sampleinfo_path):
        raise FileNotFoundError(f"File not found: {sampleinfo_path}")
    sampleinfo = pd.read_csv(sampleinfo_path)

    group_col = "Cluster"
    if group_col not in sampleinfo.columns:
        processed_proteins_folder = os.path.join(resolve_path('this_run_folder_path'), 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
        else:
            run_umap.func()
            if os.path.exists(sampleinfo_with_cluster_path):
                sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)
            else:
                raise ValueError(f"Neither '{group_col}' column found in sampleinfo nor sampleinfo_with_clusters.csv generated. Please check the UMAP clustering step.")


    # ===============================
    # 1. 每组可观测样本数（不是生物学重复）
    # ===============================
    # per_group_counts = (
    #     sampleinfo[group_col]
    #     .value_counts()
    #     .sort_index()
    #     .to_dict()
    # )

    # ===============================
    # 2. 生物学 / 技术重复状态（不可判定）
    # ===============================
    llm_summary = {}
    for sub_df in sampleinfo.groupby(group_col):
        g, g_df = sub_df
        n = g_df.shape[0]
        llm_summary[g] = {
            "repetition_count": int(n),
            # "biological_replicates_n": "The " if int(n) > 1 else "No",
            # "technical_replicates_present": "Yes" if int(n) > 1 else "No",
            "independence_statement": "Yes" if "Date" in g_df.columns and g_df.groupby("Date").size().max() > 1 else "No",
            "origin_of_replicates": "Multiple runs across Date" if "Date" in g_df.columns and g_df.groupby("Date").size().max() > 1 else "Unknown",
        }

    return llm_summary


@tool
def prepare_scoring_evidence_pack(
    *args, **kwargs
):
    """
    Generate dataset-specific user-visible evidence tables from the current matrix,
    SampleInfo, differential outputs, analysis design, and local versioned resources.
    """
    config = get_config()
    artifacts = prepare_user_visible_evidence_pack(config)
    return {
        "status": "completed",
        "evidence_scope": "user_visible",
        "evidence_source": "current_matrix",
        "requirements_path": artifacts.get("requirements_path", ""),
        "group_composition_qc": artifacts.get("group_qc", {}).get("csv", ""),
        "candidate_protein_evidence": artifacts.get("candidate_evidence", {}).get("csv", ""),
        "curated_module_scores": artifacts.get("module_scores", {}).get("group_summary_csv", ""),
        "scoring_standard_coverage": artifacts.get("scoring_coverage", {}).get("csv", ""),
        "summary_for_report": artifacts.get("markdown", ""),
    }


@tool
def search_google_information(
    queries: List[str],
    max_results: int = 5,
    *args, **kwargs
):
    """
    多查询智能搜索（科研优化版）

    搜索优先级：
    1. 本地缓存
    2. SerpAPI Google
    3. Google Custom Search
    4. DuckDuckGo（免费备用）

    特点：
    - 自动去重
    - query标准化
    - 永久缓存
    - 降低额度消耗
    """


    try:
        from ddgs import DDGS
        ddg_available = True
    except Exception:
        ddg_available = False

    # ==================================================
    # 配置区（建议改为环境变量）
    # ==================================================
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    cx = os.getenv("GOOGLE_CX", "")

    this_run_folder = resolve_path("this_run_folder_path")

    out_folder = os.path.join(this_run_folder, "google_results")
    os.makedirs(out_folder, exist_ok=True)

    all_google_info_path = os.path.join(out_folder, "all_google_info.json")
    google_path = os.path.join(out_folder, "google_info.json")

    # ==================================================
    # 工具函数
    # ==================================================
    def normalize_query(q):
        return " ".join(q.lower().strip().split())

    def safe_load_json(path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def safe_save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    # ==================================================
    # 读取缓存
    # ==================================================
    summary = {}
    all_pre_info = safe_load_json(all_google_info_path)

    # ==================================================
    # query 去重
    # ==================================================
    queries = list(dict.fromkeys(queries))
    normalized_queries = [normalize_query(q) for q in queries]
    query_signature = json.dumps(
        {
            "queries": normalized_queries,
            "max_results": int(max_results),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    if os.path.exists(google_path):
        pre_info = safe_load_json(google_path)
        if pre_info and pre_info.get("query_signature") == query_signature:
            return pre_info

    # ==================================================
    # 主循环
    # ==================================================
    for raw_query in queries:

        query = normalize_query(raw_query)

        # ------------------------
        # 缓存命中
        # ------------------------
        if query in all_pre_info:
            summary[raw_query] = all_pre_info[query]
            continue

        result_saved = False
        last_error = None

        # ==================================================
        # 1️⃣ SerpAPI Google
        # ==================================================
        if serpapi_key:

            try:
                params = {
                    "q": query,
                    "engine": "google",
                    "api_key": serpapi_key,
                    "num": max_results
                }

                r = _request_with_retry(
                    "GET",
                    "https://serpapi.com/search",
                    params=params,
                    timeout=15,
                )
                data = r.json()

                if "organic_results" in data:

                    results = []
                    snippets = []

                    for item in data["organic_results"][:max_results]:

                        results.append({
                            "title": item.get("title"),
                            "snippet": item.get("snippet"),
                            "url": item.get("link"),
                            "source": item.get("source", "unknown")
                        })

                        if item.get("snippet"):
                            snippets.append(item["snippet"])

                    summary[raw_query] = {
                        "engine": "serpapi-google",
                        "n_results": len(results),
                        "results": results,
                        "summary_for_llm": " ".join(snippets)[:1500]
                    }

                    result_saved = True

            except Exception as e:
                last_error = f"serpapi-google: {e}"

        # ==================================================
        # 2️⃣ Google Custom Search
        # ==================================================
        if (not result_saved) and google_api_key and cx:

            try:
                params = {
                    "q": query,
                    "key": google_api_key,
                    "cx": cx,
                    "num": min(max_results, 10)
                }

                r = _request_with_retry(
                    "GET",
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=15,
                )

                data = r.json()

                if "items" in data:

                    results = []
                    snippets = []

                    for item in data["items"][:max_results]:

                        results.append({
                            "title": item.get("title"),
                            "snippet": item.get("snippet"),
                            "url": item.get("link"),
                            "source": item.get("displayLink", "unknown")
                        })

                        if item.get("snippet"):
                            snippets.append(item["snippet"])

                    summary[raw_query] = {
                        "engine": "google-custom-search",
                        "n_results": len(results),
                        "results": results,
                        "summary_for_llm": " ".join(snippets)[:1500]
                    }

                    result_saved = True

            except Exception as e:
                last_error = f"google-custom-search: {e}"

        # ==================================================
        # 3️⃣ DuckDuckGo 免费备用
        # ==================================================
        if (not result_saved) and ddg_available:

            try:
                with DDGS() as ddgs:

                    rows = list(ddgs.text(query, max_results=max_results))

                results = []
                snippets = []

                for item in rows:

                    results.append({
                        "title": item.get("title"),
                        "snippet": item.get("body"),
                        "url": item.get("href"),
                        "source": "duckduckgo"
                    })

                    if item.get("body"):
                        snippets.append(item["body"])

                summary[raw_query] = {
                    "engine": "duckduckgo",
                    "n_results": len(results),
                    "results": results,
                    "summary_for_llm": " ".join(snippets)[:1500]
                }

                result_saved = True

            except Exception as e:
                last_error = f"duckduckgo: {e}"

        # ==================================================
        # 全失败
        # ==================================================
        if not result_saved:

            summary[raw_query] = {
                "engine": "none",
                "n_results": 0,
                "results": [],
                "summary_for_llm": "",
                "error": last_error
            }

        # 写缓存
        all_pre_info[query] = summary[raw_query]

        # 防止频繁请求
        time.sleep(1)

    # ==================================================
    # 保存
    # ==================================================
    output = {
        "engine_priority": [
            "serpapi-google",
            "google-custom-search",
            "duckduckgo"
        ],
        "query_signature": query_signature,
        "queries": queries,
        "normalized_queries": normalized_queries,
        "max_results": int(max_results),
        "n_queries": len(queries),
        "summary": summary
    }

    safe_save_json(output, google_path)
    safe_save_json(all_pre_info, all_google_info_path)
    google_records = []
    for raw_query, payload in summary.items():
        if not isinstance(payload, dict):
            continue
        for item in payload.get("results", [])[:max_results]:
            google_records.append({
                "query": raw_query,
                "source": item.get("source"),
                "url": item.get("url"),
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "engine": payload.get("engine"),
            })
    append_external_knowledge(
        this_run_folder,
        tool="search_google_information",
        query={"queries": queries, "max_results": max_results},
        evidence_source="external_literature",
        records=google_records,
    )
    append_evidence(
        this_run_folder,
        source="external_literature",
        tool="search_google_information",
        claim="Google/web search results were retrieved as external background annotations, not as current-matrix evidence.",
        files=[google_path],
        metrics={"n_records": len(google_records)},
        confidence="low",
        limitations=["External web search cannot raise confidence for matrix-derived claims by itself."],
    )

    return output


def query_uniprot_gene_names(uniprot_ids):
    """
    从UniProt批量查询基因名
    返回 dict: {uniprot_id: gene_name}
    """

    url = "https://rest.uniprot.org/uniprotkb/search"

    # UniProt推荐每次不要太多ID
    batch_size = 50
    mapping = {}

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])
        params = {
            "query": query,
            "fields": "accession,gene_primary",
            "format": "tsv",
            "size": len(batch)
        }
        try:
            response = _request_with_retry("GET", url, params=params, timeout=20)
        except requests.RequestException as e:
            print("UniProt query failed:", e)
            continue

        lines = response.text.strip().split("\n")
        for line in lines[1:]:  # 跳过header
            parts = line.split("\t")
            if len(parts) >= 2:
                accession = parts[0]
                gene_name = parts[1]
                mapping[accession] = gene_name
        time.sleep(1)  # 防止请求过快

    return mapping


def check_gene_name_from_uniprot(input_csv, output_csv=None):

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not exists: {input_csv}")
    df = pd.read_csv(input_csv)
    if "PG.ProteinGroups" not in df.columns:
        raise ValueError("Must contain column 'PG.ProteinGroups'.")

    if "PG.Genes" in df.columns:
        return

    # 拆分所有ID
    all_ids = set()
    for val in df["PG.ProteinGroups"].dropna():
        ids = str(val).split(";")
        for uid in ids:
            uid = uid.strip()
            if uid:
                all_ids.add(uid)

    all_ids = list(all_ids)
    mapping = query_uniprot_gene_names(all_ids)
    def map_ids_to_genes(protein_group):
        ids = str(protein_group).split(";")
        genes = []
        for uid in ids:
            uid = uid.strip()
            gene = mapping.get(uid, "")
            if gene:
                genes.append(gene.upper())
        return ";".join(genes)

    df["PG.Genes"] = df["PG.ProteinGroups"].apply(map_ids_to_genes)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return

@tool
def combat_calibration(
    reproducibility_cutoff: float = 0.1,
    need_calibration: bool = True,
    *args, **kwargs
):
    """
    对蛋白质定量矩阵实现的 ComBat 批次校正
    参数:
        reproducibility_cutoff:蛋白质重现性筛选的阈值，低于该比例的蛋白质将被过滤。默认值为 0。
        need_calibration:是否需要进行批处理校正，默认需要。
    """
    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    parameters_path = resolve_path('parameters_path')
    parameters = {}

    check_gene_name_from_uniprot(protein_quant_path, protein_quant_path)

    if os.path.exists(protein_quant_path):
        data_pq = pd.read_csv(protein_quant_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    output_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(output_folder, exist_ok=True)
    analysis_design = _get_analysis_design()
    matrix_design = analysis_design.get("matrix", {}) if isinstance(analysis_design.get("matrix"), dict) else {}
    matrix_state = matrix_design.get("matrix_state", "unknown")
    batch_col = analysis_design.get("batch_col") or "Batch"
    sample_col = _sample_id_col(sampleinfo)
    if sample_col != "FileName":
        sampleinfo = sampleinfo.copy()
        sampleinfo["FileName"] = sampleinfo[sample_col].astype(str)
    protein_col, gene_col = _protein_gene_columns(data_pq)
    if protein_col != "PG.ProteinGroups":
        data_pq = data_pq.copy()
        data_pq["PG.ProteinGroups"] = data_pq[protein_col]
    if gene_col and gene_col != "PG.Genes":
        data_pq = data_pq.copy()
        data_pq["PG.Genes"] = data_pq[gene_col]
    elif "PG.Genes" not in data_pq.columns:
        data_pq = data_pq.copy()
        data_pq["PG.Genes"] = data_pq["PG.ProteinGroups"]

    skip_repeated_batch_correction = (
        matrix_state in {"normalized", "batch_corrected"}
        and not bool(matrix_design.get("needs_batch_correction", False))
    )
    needs_log_transform = bool(matrix_design.get("needs_log_transform", True))

    if skip_repeated_batch_correction or not R_AVAILABLE:
        required_cols = ["PG.ProteinGroups", "PG.Genes"]
        missing_cols = [col for col in required_cols if col not in data_pq.columns]
        if missing_cols:
            raise ValueError(f"ProteinQuant is missing required columns: {missing_cols}")
        if "FileName" not in sampleinfo.columns:
            raise ValueError("SampleInfo is missing required column: FileName")

        sample_names = [name for name in sampleinfo["FileName"].dropna().astype(str).tolist() if name in data_pq.columns]
        if not sample_names:
            raise ValueError("No SampleInfo FileName values matched ProteinQuant sample columns.")

        sampleinfo_filtered = sampleinfo[sampleinfo["FileName"].astype(str).isin(sample_names)].copy()
        proquant = data_pq.loc[:, ["PG.ProteinGroups", "PG.Genes"] + sample_names].copy()
        numeric = proquant.loc[:, sample_names].apply(pd.to_numeric, errors="coerce")

        if "Type" in sampleinfo_filtered.columns:
            keep_rows = pd.Series(True, index=proquant.index)
            for cell_type in sampleinfo_filtered["Type"].dropna().unique():
                cell_samples = sampleinfo_filtered.loc[sampleinfo_filtered["Type"] == cell_type, "FileName"].astype(str)
                cell_samples = [name for name in cell_samples if name in numeric.columns]
                if cell_samples:
                    keep_rows &= numeric.loc[:, cell_samples].notna().sum(axis=1) / len(cell_samples) >= reproducibility_cutoff
            filtering_strategy = "by_cell_type"
        else:
            keep_rows = numeric.notna().sum(axis=1) / max(len(sample_names), 1) >= reproducibility_cutoff
            filtering_strategy = "global"

        proquant_filtered = proquant.loc[keep_rows].copy()
        numeric_filtered = numeric.loc[keep_rows].copy()

        def _impute_row(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if valid.empty:
                return row.fillna(0.0)
            half_min = float(valid.min()) / 2.0
            return row.fillna(half_min)

        imputed = numeric_filtered.apply(_impute_row, axis=1)
        _log2_applied = bool(needs_log_transform and np.nanmin(imputed.to_numpy(dtype=float)) >= 0)
        if _log2_applied:
            normalized = np.log2(imputed + 1.0)
        else:
            normalized = imputed
        # R25: record the transform that was actually applied in this run, so that a downstream
        # effect name (for example a log2-scale difference) rests on a run record instead of a
        # value-range heuristic.
        try:
            if isinstance(analysis_design, dict):
                _matrix_record = analysis_design.setdefault("matrix", {})
                if isinstance(_matrix_record, dict):
                    _matrix_record["log2_transform_applied"] = _log2_applied
                    _matrix_record["log2_transform_record_source"] = "protein_quant_combat_calibration"
        except Exception:
            pass

        sampleinfo_filtered.to_csv(os.path.join(output_folder, "SampleInfo_Filtered.csv"), index=False)
        proquant_filtered.to_csv(os.path.join(output_folder, "ProteinQuant_Filtered.csv"), index=False)

        protein_gene_map = proquant_filtered.loc[:, ["PG.ProteinGroups", "PG.Genes"]].copy()
        protein_gene_map["PG.ProteinGroups"] = protein_gene_map["PG.ProteinGroups"].astype(str).str.split(";").str[0]
        protein_gene_map["PG.Genes"] = protein_gene_map["PG.Genes"].astype(str).str.split(";").str[0]
        protein_gene_map.to_csv(os.path.join(output_folder, "Protein_Gene_Map.csv"), index=False)

        normalized_out = pd.concat(
            [proquant_filtered.loc[:, ["PG.ProteinGroups", "PG.Genes"]].reset_index(drop=True), normalized.reset_index(drop=True)],
            axis=1,
        )
        normalized_out.to_csv(os.path.join(output_folder, "ProQuant_Normalized.csv"), index=False)
        normalized_out.to_csv(os.path.join(output_folder, "ProteinQuant_ComBat.csv"), index=False)
        effective_matrix_status = (
            "processed_input_reused_without_new_batch_correction"
            if skip_repeated_batch_correction
            else "python_normalized_uncorrected_fallback"
        )
        compatibility_note = (
            "ProteinQuant_ComBat.csv is written as a downstream compatibility copy. "
            "It must not be described as newly ComBat-corrected unless combat_success is true."
        )

        # R28: persist what this step actually did to the matrix. The report used to state
        # "imputation not executed / log transform not executed" because the design metadata carried
        # no such field; the fields below come from the code path that ran, so a reader-facing
        # preprocessing statement can cite an execution record instead of an absent key.
        try:
            _transform_record = {
                "record_type": "matrix_transform_record",
                "stage": "protein_quant_combat_calibration",
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": {
                    "protein_quant_file": os.path.basename(str(protein_quant_path)),
                    "sampleinfo_file": os.path.basename(str(sampleinfo_path)),
                    "protein_rows": int(data_pq.shape[0]),
                    "samples": int(len(sample_names)),
                },
                "protein_filtering": {
                    "rule": ("按细胞类型分别要求非缺失比例 >= %s" % reproducibility_cutoff
                             if filtering_strategy == "by_cell_type"
                             else "全部样本中非缺失比例 >= %s" % reproducibility_cutoff),
                    "strategy": filtering_strategy,
                    "reproducibility_cutoff": reproducibility_cutoff,
                    "n_proteins_retained": int(proquant_filtered.shape[0]),
                },
                "imputation": {
                    "applied": bool(int(numeric_filtered.isna().sum().sum()) > 0),
                    "method": "逐蛋白取该行非缺失最小值的一半填补（half-min）",
                    "n_values_imputed": int(numeric_filtered.isna().sum().sum()),
                },
                "log2_transform": {
                    "applied": bool(_log2_applied),
                    "form": "log2(x+1)" if _log2_applied else "",
                    "needs_log_transform_flag": bool(needs_log_transform),
                    "not_applied_reason": ("" if _log2_applied else
                                           "矩阵含负值或未要求对数变换，本次按原尺度继续"),
                },
                "batch_correction": {
                    "combat_success": False,
                    "new_batch_correction_applied": False,
                    "batch_column": batch_col,
                    "effective_matrix_status": effective_matrix_status,
                    "reason": ("analysis_design matrix_state=%s; repeated batch correction skipped"
                               % matrix_state) if skip_repeated_batch_correction
                              else "R/rpy2 unavailable: %s" % R_IMPORT_ERROR,
                },
                "analysed_matrix": "ProQuant_Normalized.csv",
                "compatibility_matrix": "ProteinQuant_ComBat.csv",
                "code_path": "tools.py:protein_quant_combat_calibration",
            }
            with open(os.path.join(output_folder, "matrix_transform_record.json"), "w",
                      encoding="utf-8") as _tf:
                json.dump(_transform_record, _tf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        llm_summary = {
            "analysis_type": "protein_quant_combat_calibration",
            "input": {
                "protein_quant_file": protein_quant_path,
                "sampleinfo_file": sampleinfo_path,
                "initial_protein_rows": int(data_pq.shape[0]),
                "initial_samples": int(sampleinfo.shape[0]),
            },
            "protein_filtering": {
                "reproducibility_cutoff": reproducibility_cutoff,
                "filtering_strategy": filtering_strategy,
                "number_of_proteins_after_filtering": int(proquant_filtered.shape[0]),
            },
            "normalization": {
                "method": "Python half-min imputation" + (" with log2(x+1) when non-negative" if needs_log_transform else " without additional log transform"),
                "r_available": bool(R_AVAILABLE),
                "statistical_backend": "design_skip" if skip_repeated_batch_correction else "python_fallback",
            },
            "batch_correction": {
                "method": "No new batch correction; analysis_design marked the input matrix as already processed or batch-corrected" if skip_repeated_batch_correction else "ComBat skipped; Python normalized matrix used as fallback",
                "batch_column": batch_col,
                "parametric_prior": None,
                "combat_success": False,
                "new_batch_correction_applied": False,
                "fallback_behavior": effective_matrix_status,
                "fallback_reason": (
                    f"analysis_design matrix_state={matrix_state}; repeated batch correction skipped"
                    if skip_repeated_batch_correction
                    else f"R/rpy2 unavailable: {R_IMPORT_ERROR}"
                ),
                "effective_matrix_status": effective_matrix_status,
                "compatibility_note": compatibility_note,
                "matrix_state": matrix_state,
                "statistical_backend": "design_skip" if skip_repeated_batch_correction else "python_fallback",
            },
            "outputs": {
                "output_folder": output_folder,
                "files": {
                    "filtered_sampleinfo": "SampleInfo_Filtered.csv",
                    "filtered_protein_quant": "ProteinQuant_Filtered.csv",
                    "protein_gene_map": "Protein_Gene_Map.csv",
                    "normalized_matrix": "ProQuant_Normalized.csv",
                    "compatibility_matrix": "ProteinQuant_ComBat.csv",
                },
                "compatibility_note": compatibility_note,
            },
        }

        parameters["combat_calibration"] = {
            "files": {
                "filtered_sampleinfo": os.path.join(output_folder, "SampleInfo_Filtered.csv"),
                "filtered_protein_quant": os.path.join(output_folder, "ProteinQuant_Filtered.csv"),
                "protein_gene_map": os.path.join(output_folder, "Protein_Gene_Map.csv"),
                "normalized_matrix": os.path.join(output_folder, "ProQuant_Normalized.csv"),
                "compatibility_matrix": os.path.join(output_folder, "ProteinQuant_ComBat.csv"),
            },
            "batch_correction": llm_summary["batch_correction"],
        }
        save_json(parameters, parameters_path)
        append_evidence(
            this_run_folder,
            source="current_matrix",
            tool="combat_calibration",
            claim=(
                "Input matrix was marked processed/batch-corrected by analysis_design; no new batch correction was applied and the ComBat-named file is a compatibility copy."
                if skip_repeated_batch_correction
                else "Protein matrix was normalized and batch correction was unavailable; downstream results use a normalized/uncorrected Python fallback."
            ),
            files=[
                os.path.join(output_folder, "ProQuant_Normalized.csv"),
                os.path.join(output_folder, "ProteinQuant_ComBat.csv"),
            ],
            metrics={
                "matrix_state": matrix_state,
                "skip_repeated_batch_correction": skip_repeated_batch_correction,
                "statistical_backend": "design_skip" if skip_repeated_batch_correction else "python_fallback",
                "combat_success": False,
                "new_batch_correction_applied": False,
                "effective_matrix_status": effective_matrix_status,
                "n_proteins": int(proquant_filtered.shape[0]),
                "n_samples": int(len(sample_names)),
            },
            confidence=_confidence_from_design_and_backend("python_fallback"),
            limitations=[llm_summary["batch_correction"]["fallback_reason"]],
        )
        return llm_summary

    # if not need_calibration:
    #     combat_csv = data_pq
    #     combat_csv_path = os.path.join(output_folder, "ProteinQuant_ComBat.csv")
    #     combat_csv.to_csv(combat_csv_path)
    #     llm_summary = {
    #         "note": "The input file has been batch corrected, there is no need to repeat this operation."
    #     }
    #     return llm_summary

    # ---------- 传递数据到 R ----------
    with conversion.localconverter(converter):
        r_data_pq= conversion.py2rpy(data_pq)
        r_sampleinfo = conversion.py2rpy(sampleinfo)

    robjects.globalenv["data_pq"] = r_data_pq
    robjects.globalenv["sampleinfo"] = r_sampleinfo
    robjects.globalenv["repro_cutoff"] = reproducibility_cutoff
    robjects.globalenv["batch_col"] = batch_col
    robjects.globalenv["output_folder"] = output_folder
    robjects.globalenv["need_calibration"] = need_calibration

    robjects.r(f'''
        suppressMessages(library(MBQN))
        suppressMessages(library(sva))

        check_columns <- function(df, cols) {{
            missing <- setdiff(cols, colnames(df))
            if (length(missing) > 0)
                stop(paste("缺少字段:", paste(missing, collapse=",")))
        }}

        # ---------- 字段检查 ----------
        check_columns(data_pq, c("PG.ProteinGroups", "PG.Genes"))
        check_columns(sampleinfo, c("FileName"))

        # ---------- 样本筛选 ----------
        sample_names <- sampleinfo$FileName

        proquant <- data_pq[, c("PG.ProteinGroups", "PG.Genes", sample_names)]

        write.csv(sampleinfo, file.path(output_folder, "SampleInfo_Filtered.csv"), row.names = FALSE)

        # ---------- 重现性筛选 ----------
        if ("Type" %in% colnames(sampleinfo)) {{
            keep_rows <- rep(TRUE, nrow(proquant))

            for (cell_type in unique(sampleinfo$Type)) {{
                cell_samples <- sampleinfo$FileName[sampleinfo$Type == cell_type]
                cell_samples <- cell_samples[!is.na(cell_samples)]
                cell_samples <- intersect(cell_samples, colnames(proquant))

                if (length(cell_samples) > 0) {{
                    reproducibility <- rowSums(
                        !is.na(proquant[, cell_samples, drop = FALSE])
                    ) / length(cell_samples)

                    keep_rows <- keep_rows & (reproducibility >= repro_cutoff)
                }}
            }}
            proquant <- proquant[keep_rows, ]
        }} else {{
            proquant <- subset(proquant, rowSums(!is.na(proquant[, -3])) / ncol(proquant[, -3]) >= repro_cutoff)
        }}

        write.csv(proquant, file.path(output_folder, "ProteinQuant_Filtered.csv"), row.names = FALSE)

        # ---------- Protein–Gene 映射 ----------
        protein_gene_map <- proquant[, c("PG.ProteinGroups", "PG.Genes")]
        protein_gene_map$PG.ProteinGroups <- sapply(strsplit(as.character(protein_gene_map$PG.ProteinGroups), ";"), head, 1)
        protein_gene_map$PG.Genes <- sapply(strsplit(as.character(protein_gene_map$PG.Genes), ";"), head, 1)

        write.csv(protein_gene_map, file.path(output_folder, "Protein_Gene_Map.csv"), row.names = FALSE)

        # ---------- 定量矩阵 ----------
        proquant <- proquant[, c("PG.ProteinGroups", sample_names)]
        proquant$PG.ProteinGroups <- sapply(strsplit(as.character(proquant$PG.ProteinGroups), ";"), head, 1)

        # ---------- half-min 缺失值填补 ----------
        mat <- as.matrix(proquant[, -1])
        mat_half <- t(apply(mat, 1, function(x) {{
            ifelse(is.na(x), min(x, na.rm = TRUE) / 2, x)
        }}))

        # ---------- MBQN + log2 ----------
        mat_mbqn <- mbqn(mat_half, FUN = NULL)
        if (min(mat_mbqn, na.rm = TRUE) >= 0) {{
            mat <- log2(mat_mbqn + 1)
        }} else {{
            mat <- as.matrix(proquant[, -1])
            mat <- t(apply(mat, 1, function(x) {{
                # ifelse(is.na(x), 0.0, x)
                ifelse(is.na(x), min(x, na.rm = TRUE) / 2, x)
            }}))
        }}

        write.csv(cbind(PG.ProteinGroups = proquant$PG.ProteinGroups, PG.Genes = protein_gene_map$PG.Genes, mat), file.path(output_folder, "ProQuant_Normalized.csv"), row.names = FALSE)


        # ---------- ComBat ----------
        combat_success <- TRUE

        if (!(need_calibration)) {{
            #    mat <- as.matrix(proquant[, -1])
            #    proquant_out <- cbind(PG.ProteinGroups = proquant$PG.ProteinGroups, mat)
            # proquant_out <- proquant
            # proquant_out <- cbind(PG.ProteinGroups = proquant$PG.ProteinGroups, mat_half)

            mat <- as.matrix(proquant[, -1])
            mat_mean <- t(apply(mat, 1, function(x) {{
                mean_val <- mean(x, na.rm = TRUE)
                if (!is.finite(mean_val)) {{
                    mean_val <- 0
                }}
                x[is.na(x)] <- mean_val
                return(x)
            }}))
            proquant_out <- cbind(PG.ProteinGroups = proquant$PG.ProteinGroups, PG.Genes = protein_gene_map$PG.Genes, mat_mean)

        }} else {{

            if (!(batch_col %in% colnames(sampleinfo))) {{
                sampleinfo[[batch_col]] <- 1
            }}
            batch <- as.factor(sampleinfo[[batch_col]])
            tryCatch({{
                mat_corrected <-  ComBat(
                        dat = mat,
                        batch = batch,
                        mod = NULL,
                        par.prior = TRUE,
                        prior.plots = FALSE
                    )
                    combat_success <- TRUE
                }}, error = function(e) {{
                    mat_corrected <<- mat
                    combat_success <<- FALSE
            }})
            a <- table(batch)
            proquant_out <- cbind(PG.ProteinGroups = proquant$PG.ProteinGroups, PG.Genes = protein_gene_map$PG.Genes, mat_corrected)

        }}

        write.csv(proquant_out, file.path(output_folder, "ProteinQuant_ComBat.csv"), row.names = FALSE)

    ''')


    proquant_out = robjects.r["proquant_out"]
    combat_success = bool(robjects.r["combat_success"][0])

    llm_summary = {
    "analysis_type": "protein_quant_combat_calibration",

    "input": {
        "protein_quant_file": protein_quant_path,
        "sampleinfo_file": sampleinfo_path,
        "initial_protein_rows": int(data_pq.shape[0]),
        "initial_samples": int(sampleinfo.shape[0])
    },


    "protein_filtering": {
        "reproducibility_cutoff": reproducibility_cutoff,
        "filtering_strategy": (
            "by_cell_type" if "Type" in sampleinfo.columns else "global"
        ),
        "number_of_proteins_after_filtering": int(proquant_out.nrow)
    },

    "batch_correction": {
        "method": "ComBat (sva)",
        "batch_column": batch_col,
        "parametric_prior": True,
        "combat_success": combat_success,
        "matrix_state": matrix_state,
        "statistical_backend": "r_sva_combat",
        "fallback_behavior": (
            "used_uncorrected_matrix" if not combat_success else "not_triggered"
        )
    },

    "outputs": {
        "output_folder": output_folder,
        "files": {
            "filtered_sampleinfo": "SampleInfo_Filtered.csv",
            "filtered_protein_quant": "ProteinQuant_Filtered.csv",
            "protein_gene_map": "Protein_Gene_Map.csv",
            "normalized_matrix": "ProQuant_Normalized.csv",
            "combat_matrix": "ProteinQuant_ComBat.csv"
        }
    },
    }

    parameters["combat_calibration"] = {
        "files": {
            "filtered_sampleinfo": os.path.join(output_folder, "SampleInfo_Filtered.csv"),
            "filtered_protein_quant": os.path.join(output_folder, "ProteinQuant_Filtered.csv"),
            "protein_gene_map": os.path.join(output_folder, "Protein_Gene_Map.csv"),
            "normalized_matrix": os.path.join(output_folder, "ProQuant_Normalized.csv"),
            "combat_matrix": os.path.join(output_folder, "ProteinQuant_ComBat.csv")
        },
        "batch_correction": llm_summary["batch_correction"],
    }

    save_json(parameters, parameters_path)
    append_evidence(
        this_run_folder,
        source="current_matrix",
        tool="combat_calibration",
        claim="Protein matrix normalization and R ComBat batch correction step completed.",
        files=[
            os.path.join(output_folder, "ProQuant_Normalized.csv"),
            os.path.join(output_folder, "ProteinQuant_ComBat.csv"),
        ],
        metrics={
            "matrix_state": matrix_state,
            "statistical_backend": "r_sva_combat",
            "combat_success": bool(combat_success),
            "n_proteins": int(proquant_out.nrow),
        },
        confidence=_confidence_from_design_and_backend("r_sva_combat"),
        limitations=[] if combat_success else ["R ComBat failed and uncorrected matrix was used."],
    )

    return llm_summary


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    自动推断 CSV 中各类字段的语义
    """
    schema = {
        "protein_col": None,
        "gene_col": None,
        "logfc_col": None,
        "pval_col": None,
        "detection_cols": [],
        "cluster_names": []
    }

    for col in df.columns:
        cl = col.lower()

        if schema["protein_col"] is None and any(k in cl for k in ["protein", "uniprot", "pg."]):
            schema["protein_col"] = col

        if schema["gene_col"] is None and "gene" in cl:
            schema["gene_col"] = col

        if schema["logfc_col"] is None and any(k in cl for k in ["logfc", "log_fc", "fold"]):
            schema["logfc_col"] = col

        if schema["pval_col"] is None and any(k in cl for k in ["pval", "p_value", "adj"]):
            schema["pval_col"] = col

        if "detection" in cl or "rate" in cl:
            schema["detection_cols"].append(col)

    # 推断 cluster 名称
    for col in schema["detection_cols"]:
        parts = col.split("_")
        if len(parts) > 1:
            schema["cluster_names"].append(parts[-1])

    schema["cluster_names"] = list(set(schema["cluster_names"]))

    return schema


# ========== 2. 基础统计（不给 LLM 算数） ==========

def compute_basic_stats(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    stats = {}
    stats["n_proteins"] = len(df)
    if schema["logfc_col"]:
        stats["upregulated"] = int((df[schema["logfc_col"]] > 0).sum())
        stats["downregulated"] = int((df[schema["logfc_col"]] < 0).sum())

    det_summary = {}
    for col in schema["detection_cols"]:
        det_summary[col] = {
            "median": float(df[col].median()),
            "mean": float(df[col].mean()),
            "high_conf_rate": float((df[col] > 0.5).mean())
        }

    stats["detection_summary"] = det_summary

    return stats


# ========== 4. 主函数（给 Agent 用） ==========
@tool
def evaluate_sc_proteomics_csv(
    *args, **kwargs
    ):
    """
    对单细胞蛋白质组学差异分析结果进行质量评估总结。

    1. 自动解析差异蛋白表中各字段的统计语义结构；
    2. 计算关键数值字段的基础统计特征（如分布、范围、缺失情况等）；
    3. 汇总每个 contrast 下的代表性差异蛋白 / 基因及其 logFC 信息；
    4. 将上述结构化信息输入至质量评估分析器（LLM），
       生成中立、描述性的单细胞蛋白质组学数据质量评估文本。

    该函数的职责仅限于“评估与总结现有差异分析结果”，
    不执行差异分析本身，也不提出任何后续分析建议或操作决策。

    """

    this_run_folder = resolve_path('this_run_folder_path')

    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')

    differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv")]
    if len(differential_proteins_path_list) == 0:
        print("No differential protein files found in processed_proteins folder. Try to run limma first.")
        _ = run_pairwise_limma.func()
        differential_proteins_path_list = [fn for fn in os.listdir(processed_proteins_folder) if fn.startswith("differential_") and fn.endswith(".csv")]
    if len(differential_proteins_path_list) == 0:
        return "No differential protein files were available after attempting pairwise differential analysis; matrix-level QC should be read from group_composition_qc and visualization outputs."
    response_text = ""
    contrast_col = "contrast"
    schema_dict = {}
    stats_dict = {}
    proteins_dict = {}
    for fn in differential_proteins_path_list:
        csv_path = os.path.join(processed_proteins_folder, fn)
        df = pd.read_csv(csv_path)
        for contrast, subdf in df.groupby(contrast_col):
            safe_contrast = contrast.replace(" ", "")  # ✅ 统一安全命名
            schema_dict[safe_contrast] = infer_schema(subdf)
            stats_dict[safe_contrast] = compute_basic_stats(subdf, schema_dict[safe_contrast])
            proteins_dict[safe_contrast] = proteins_basic_values(subdf)

    planner_prompt = ChatPromptTemplate.from_messages([
            ("system", QC_EVAL_PROMPT),
            ("human", """请基于以下信息，进行单细胞蛋白质组学数据质量评估：
            原始差异蛋白中各类字段的语义: {schema},
            基础统计结果: {basic_stats},
            差异蛋白/基因列表及其对应LogFC: {proteins}
            """),

            ])

    planner_chain = planner_prompt | LLM

    try:
        response = planner_chain.invoke({
            "schema": json.dumps(schema_dict, ensure_ascii=False),
            "basic_stats": json.dumps(stats_dict, ensure_ascii=False),
            "proteins": json.dumps(proteins_dict, ensure_ascii=False)
            })
        response_text += response.content
    except Exception as e:
        response_text += "error: " + str(e)

    return response_text

def record_report(file, summary="", mode='a'):
    print(summary)
    os.makedirs(os.path.dirname(file), exist_ok=True) if os.path.dirname(file) else None
    with open(file, mode, encoding='utf-8') as f:
        f.write(summary.strip() + "\n\n")


def proteins_basic_values(df):

    col_gene = "PG.Genes"
    col_logfc = "logFC"
    col_pval = "adj.P.Val"
    df = df[[col_gene, col_logfc, col_pval]]
    temp_list = []
    for idx, row in df.iterrows():
        gene = row[col_gene]
        logfc = round(row[col_logfc], 2)
        pval = row[col_logfc]
        if abs(logfc) > 1.0 and pval < 0.05:
            temp_list.append(str(gene) + ": " + str(logfc))
    results = '; '.join(temp_list)

    return results



@tool
def run_wgcna(
    *args, **kwargs
):

    """
    WGCNA 模块识别算法
    """
    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')
    sampleinfo_path = resolve_path('sampleinfo_path')
    output_folder = os.path.join(this_run_folder, 'wgcna_results')
    os.makedirs(output_folder, exist_ok=True)

    if not R_AVAILABLE or conversion is None or converter is None or robjects is None:
        skip_payload = {
            "status": "skipped",
            "reason": "R/rpy2-dependent WGCNA is unavailable in this environment.",
            "r_import_error": str(R_IMPORT_ERROR) if R_IMPORT_ERROR else "",
            "evidence_boundary": (
                "No WGCNA modules, eigengenes, module membership, or module-trait "
                "associations were generated for this run."
            ),
        }
        skip_path = os.path.join(output_folder, "wgcna_skip.json")
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skip_payload, f, ensure_ascii=False, indent=2)
        append_evidence(
            this_run_folder,
            source="current_matrix",
            tool="run_wgcna",
            claim="WGCNA was skipped because the R/rpy2-dependent backend is unavailable.",
            metrics=skip_payload,
            files=[skip_path],
            limitations=[skip_payload["evidence_boundary"]],
        )
        return skip_payload

    if os.path.exists(protein_quant_path):
        data_pq = pd.read_csv(protein_quant_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    output_folder = os.path.join(this_run_folder, 'wgcna_results')
    os.makedirs(output_folder, exist_ok=True)

    # ---------- 传递数据到 R ----------
    with conversion.localconverter(converter):
        r_data_pq= conversion.py2rpy(data_pq)
        r_sampleinfo = conversion.py2rpy(sampleinfo)
    robjects.globalenv["data_pq"] = r_data_pq
    robjects.globalenv["sampleinfo"] = r_sampleinfo
    robjects.globalenv["output_folder"] = output_folder


    robjects.r(f'''
        suppressMessages(library(WGCNA))
        suppressMessages(library(jsonlite))

        options(stringsAsFactors = FALSE)
        allowWGCNAThreads()

        # ========= 2. 读取数据 =========

        # 处理蛋白ID（去重）
        data_pq <- data_pq[!is.na(data_pq[,1]) & data_pq[,1] != "", ]
        rownames(data_pq) <- make.unique(as.character(data_pq[,1]))
        data_pq <- data_pq[,-1]

        # 对齐样本
        colnames(data_pq) <- trimws(colnames(data_pq))
        sampleinfo$FileName <- trimws(sampleinfo$FileName)

        common_samples <- intersect(colnames(data_pq), sampleinfo$FileName)

        data_pq <- data_pq[, common_samples, drop = FALSE]
        sampleinfo <- sampleinfo[sampleinfo$FileName %in% common_samples, ]

        # 按 sampleinfo 排序
        data_pq <- data_pq[, sampleinfo$FileName, drop = FALSE]

        # ========= 3. 数据预处理 =========
        data_pq <- log2(data_pq + 1)

        datExpr <- t(data_pq)
        datExpr <- as.data.frame(datExpr)

        # NA处理
        datExpr[is.na(datExpr)] <- 0

        # 去低方差蛋白
        variance <- apply(datExpr, 2, var)
        datExpr <- datExpr[, variance > quantile(variance, 0.25)]

        # 质量控制
        gsg <- goodSamplesGenes(datExpr, verbose = 3)
        if (!gsg$allOK) {{
            datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
        }}

        # ========= 4. 选择软阈值 =========
        powers <- c(1:20)
        sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

        # 自动选择softPower（简单策略）
        fit <- sft$fitIndices
        softPower <- fit$Power[which.max(fit$SFT.R.sq > 0.8)]
        if (is.na(softPower)) softPower <- 6

        # ========= 5. 构建网络 =========
        adjacency <- adjacency(datExpr, power = softPower)

        TOM <- TOMsimilarity(adjacency)
        dissTOM <- 1 - TOM

        # ========= 6. 模块识别 =========
        geneTree <- hclust(as.dist(dissTOM), method = "average")

        dynamicMods <- cutreeDynamic(
        dendro = geneTree,
        distM = dissTOM,
        deepSplit = 2,
        pamRespectsDendro = FALSE,
        minClusterSize = 30
        )

        moduleColors <- labels2colors(dynamicMods)

        # ========= 7. 模块合并 =========
        MEList <- moduleEigengenes(datExpr, colors = moduleColors)
        MEs <- MEList$eigengenes

        merge <- mergeCloseModules(
        datExpr,
        moduleColors,
        cutHeight = 0.25
        )

        moduleColors <- merge$colors
        MEs <- merge$newMEs

        # ========= 8. Hub蛋白 =========
        hub_list <- list()
        MM_all <- data.frame()

        for (module in unique(moduleColors)) {{

        moduleGenes <- moduleColors == module
        if (sum(moduleGenes) < 10) next

        ME <- MEs[, paste0("ME", module)]
        MM <- cor(datExpr[, moduleGenes], ME)

        proteins <- colnames(datExpr)[moduleGenes]

        # Hub筛选
        hub <- proteins[abs(MM) > 0.8]
        hub_list[[module]] <- hub

        # 保存MM
        tmp <- data.frame(
            protein = proteins,
            module = module,
            MM = as.numeric(MM)
        )

        MM_all <- rbind(MM_all, tmp)
        }}

        # ========= 9. 模块-蛋白映射 =========
        module2genes <- split(colnames(datExpr), moduleColors)

        module_gene_df <- stack(module2genes)
        colnames(module_gene_df) <- c("protein", "module")

        # ========= 10. 模块大小 =========
        module_size <- as.data.frame(table(moduleColors))
        colnames(module_size) <- c("module", "size")

        # ========= 11. Hub蛋白整理 =========
        hub_df <- data.frame()

        for (module in names(hub_list)) {{
        hubs <- hub_list[[module]]
        if (length(hubs) == 0) next

        tmp <- data.frame(
            module = module,
            protein = hubs
        )

        hub_df <- rbind(hub_df, tmp)
        }}

        # ========= 12. 模块-表型关系 =========
        trait_df <- data.frame()

        if ("Condition" %in% colnames(sampleinfo)) {{
        trait <- data.frame(
            Condition = as.numeric(factor(sampleinfo$Condition))
        )
        rownames(trait) <- sampleinfo$FileName
        trait <- trait[rownames(datExpr), , drop = FALSE]
        moduleTraitCor <- cor(MEs, trait, use = "p")
        trait_df <- as.data.frame(moduleTraitCor)
        }}

        # ========= 13. 输出（LLM专用） =========

        write.csv(module_gene_df, file.path(output_folder, "module_gene_mapping.csv"), row.names = FALSE)
        write.csv(module_size, file.path(output_folder, "module_size.csv"), row.names = FALSE)
        write.csv(MEs, file.path(output_folder, "module_eigengenes.csv"))
        write.csv(hub_df, file.path(output_folder, "hub_proteins.csv"), row.names = FALSE)
        write.csv(MM_all, file.path(output_folder, "module_membership.csv"), row.names = FALSE)

        if (nrow(trait_df) > 0) {{
            write.csv(trait_df, file.path(output_folder, "module_trait_correlation.csv"))
        }}


        # ========= 获取模块列表 =========
        modules <- unique(module_gene_df$module)

        # ========= 构建字典 =========
        result <- list()

        for (m in modules) {{

        # 所有蛋白
        proteins <- module_gene_df$protein[module_gene_df$module == m]

        # hub蛋白
        hubs <- hub_df$protein[hub_df$module == m]

        # trait相关性（假设只有一列）
        trait_value <- NA
        if (nrow(trait_df) > 0 && paste0("ME", m) %in% rownames(trait_df)) {{
            trait_value <- trait_df[paste0("ME", m), 1]
        }}

        result[[m]] <- list(
            hub_proteins = as.character(hubs),
            proteins = as.character(proteins),
            trait_correlation = as.numeric(trait_value)
        )

        json_out <- toJSON(result, pretty = TRUE, auto_unbox = TRUE)

        # 保存
        write(json_out, file.path(output_folder, "wgcna_llm.json"))
        }}

    ''')

    results_path = os.path.join(output_folder, "wgcna_llm.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    return result



@tool
def run_protein_trajectory(
    *args, **kwargs
):
    """
    完整蛋白质轨迹分析
    """
    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    if os.path.exists(protein_quant_path):
        data_pq = pd.read_csv(protein_quant_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_path} is not found.")

    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    output_folder = os.path.join(this_run_folder, 'trajectory_results')
    os.makedirs(output_folder, exist_ok=True)

    try:
        import igraph  # noqa: F401
    except ModuleNotFoundError:
        llm_summary = {
            "status": "skipped_missing_optional_dependency",
            "evidence_source": "not_used",
            "missing_dependency": "igraph",
            "note": "Protein trajectory/PAGA analysis requires igraph; skipped instead of failing the run.",
        }
        save_json(llm_summary, os.path.join(output_folder, "trajectory_summary.json"))
        return llm_summary

    # ========== 蛋白矩阵 ==========
    data_pq = data_pq.set_index("PG.Genes")

    # ========== 清洗 sampleinfo ==========
    sampleinfo = sampleinfo.dropna(subset=["Cluster"])

    # ========== 对齐 ==========
    common_cells = list(set(data_pq.columns) & set(sampleinfo["FileName"]))

    data_pq = data_pq[common_cells]
    sampleinfo = sampleinfo[sampleinfo["FileName"].isin(common_cells)]
    sampleinfo = sampleinfo.set_index("FileName").loc[data_pq.columns]

    # ========== NA处理 ==========
    data_pq = data_pq.fillna(0)
    # imputer = SimpleImputer(strategy="mean")
    # data_pq[:] = imputer.fit_transform(data_pq)

    cell_sums = data_pq.sum(axis=0)
    valid_cells = cell_sums > 0
    data_pq = data_pq.loc[:, valid_cells]
    sampleinfo = sampleinfo.loc[valid_cells]

    # ========== 构建 AnnData ==========
    adata = sc.AnnData(data_pq.T)

    adata.obs = sampleinfo.copy()
    adata.obs["Cluster"] = adata.obs["Cluster"].astype("category")

    # ========== 预处理 ==========
    sc.pp.normalize_total(adata)
    adata.X = np.nan_to_num(adata.X)
    min_value = np.min(adata.X)
    if min_value > -1:
        sc.pp.log1p(adata)

    # ========== PCA ==========
    sc.tl.pca(adata)

    # ========== 邻接图 ==========
    sc.pp.neighbors(adata, n_neighbors=10)

    # ========== PAGA ==========
    sc.tl.paga(adata, groups="Cluster")

    # ❗❗关键修复：生成 PAGA 坐标（否则 UMAP 会报错）
    sc.pl.paga(adata, show=False)

    # ========== UMAP ==========
    sc.tl.umap(adata, init_pos="paga")

    # ========== Diffusion pseudotime ==========
    sc.tl.diffmap(adata)

    # 自动 root（最大 Cluster）
    root_cluster = adata.obs["Cluster"].value_counts().idxmax()
    root_cell = adata.obs[adata.obs["Cluster"] == root_cluster].index[0]

    adata.uns["iroot"] = np.where(adata.obs_names == root_cell)[0][0]

    sc.tl.dpt(adata)


    # pseudotime
    pt = adata.obs["dpt_pseudotime"]
    pt_data = pt.reset_index()
    pt_data.columns = ["cell", "pseudotime"]
    pt_data.to_csv(f"{output_folder}/pseudotime.csv", index=False)

    # sampleinfo
    adata.obs.to_csv(f"{output_folder}/cell_sampleinfodata.csv")

    # UMAP
    umap = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names,
        columns=["UMAP1", "UMAP2"]
    )
    umap.to_csv(f"{output_folder}/umap.csv")

    # PAGA graph
    paga = adata.uns["paga"]["connectivities"].toarray()
    pd.DataFrame(paga).to_csv(f"{output_folder}/paga_graph.csv")


    llm_summary = {}

    # ===== 1. 基本信息 =====
    llm_summary["basic_info"] = {
        "n_cells": int(adata.n_obs),
        "n_proteins": int(adata.n_vars),
        "n_clusters": int(adata.obs["Cluster"].nunique()),
        "clusters": list(map(str, adata.obs["Cluster"].cat.categories))
    }

    # ===== 2. cluster 分布 =====
    cluster_counts = adata.obs["Cluster"].value_counts().to_dict()
    llm_summary["cluster_distribution"] = {
        str(k): int(v) for k, v in cluster_counts.items()
    }

    # ===== 3. root 信息 =====
    llm_summary["trajectory_root"] = {
        "root_cluster": str(root_cluster),
        "root_cell": str(root_cell)
    }

    # ===== 4. pseudotime 分布 =====
    pt = adata.obs["dpt_pseudotime"]

    llm_summary["pseudotime"] = {
        "min": float(pt.min()),
        "max": float(pt.max()),
        "mean": float(pt.mean()),
        "std": float(pt.std())
    }

    # ===== 5. 每个 cluster 的 pseudotime =====
    cluster_pt = (
        adata.obs
        .groupby("Cluster")["dpt_pseudotime"]
        .mean()
        .sort_values()
    )

    llm_summary["cluster_pseudotime_order"] = [
        {"cluster": str(k), "mean_pseudotime": float(v)}
        for k, v in cluster_pt.items()
    ]

    # ===== 6. PAGA connectivity（精简版）=====
    paga_conn = adata.uns["paga"]["connectivities"].toarray()
    clusters = list(adata.obs["Cluster"].cat.categories)

    edges = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            weight = paga_conn[i, j]
            if weight > 0.1:   # 阈值过滤（避免太密）
                edges.append({
                    "source": str(clusters[i]),
                    "target": str(clusters[j]),
                    "weight": float(weight)
                })

    llm_summary["paga_graph"] = edges

    # ===== 7. UMAP 范围（用于理解结构）=====
    umap = adata.obsm["X_umap"]

    llm_summary["umap"] = {
        "x_range": [float(umap[:,0].min()), float(umap[:,0].max())],
        "y_range": [float(umap[:,1].min()), float(umap[:,1].max())]
    }

    output_path = os.path.join(output_folder, "trajectory_summary.json")
    save_json(llm_summary, output_path)

    return llm_summary


@tool
def capture_protein_markers_on_trajectory(
    top_n=20,
    n_permutations=500,
    log_transform=False,
    random_state=42,
    *args, **kwargs
):
    """
    蛋白质组发育轨迹动态模式识别（最终版）

    Returns
    -------
    dict:
    {
      "trajectory": [...],
      "transient_IPC-EN": [...],
      "late_EN": [...]
    }
    """

    top_n = max(1, min(int(top_n), 200))
    n_permutations = max(1, int(n_permutations))

    np.random.seed(random_state)

    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')
    sampleinfo_path = resolve_path('sampleinfo_path')

    output_folder = os.path.join(this_run_folder, 'trajectory_results')
    os.makedirs(output_folder, exist_ok=True)


    # ========= 读取 =========
    expr = pd.read_csv(protein_quant_path, index_col="PG.Genes").fillna(0)
    meta = pd.read_csv(sampleinfo_path)
    meta = meta.copy()

    # 去掉真正的 NaN
    meta = meta.dropna(subset=["Cluster"])

    # ========= 2. 对齐 =========
    common = list(set(expr.columns) & set(meta["FileName"]))
    expr = expr[common]
    meta = meta[meta["FileName"].isin(common)]

    # ========= 3. log =========
    if log_transform:
        expr = np.log2(expr + 1)

    # ========= 4. cluster mean =========
    cluster_mean = {}

    for c in meta["Cluster"].unique():
        samples = meta.loc[meta["Cluster"] == c, "FileName"]
        cluster_mean[c] = expr[samples].mean(axis=1)
    df = pd.DataFrame(cluster_mean)

    # ========= 5. trajectory推断 =========

    dist = pdist(df.T, metric="euclidean")
    Z = linkage(dist, method="average")
    order = leaves_list(Z)
    trajectory = df.columns[order].tolist()

    df = df[trajectory]

    # ========= 6. z-score =========
    # z = df.sub(df.mean(axis=1), axis=0)
    # z = z.div(df.std(axis=1) + 1e-6, axis=0)
    z = df.copy()

    required_stages = ["RG", "IPC-EN", "EN"]
    missing_stages = [stage for stage in required_stages if stage not in z.columns]
    if missing_stages:
        llm_summary = {
            "status": "not_applicable",
            "reason": "This trajectory marker tool requires RG, IPC-EN, and EN stage labels.",
            "missing_stages": missing_stages,
            "available_stages": [str(col) for col in z.columns.tolist()],
            "trajectory": [str(col) for col in trajectory],
            "evidence_source": "current_matrix",
            "recommended_interpretation": (
                "Do not report RG/IPC-EN/EN dynamic marker claims for this dataset. "
                "Use run_protein_trajectory, differential tables, or dataset-specific contrasts instead."
            ),
        }
        output_path = os.path.join(output_folder, 'trajectory_marker_summary.json')
        save_json(llm_summary, output_path)
        return llm_summary

    # ========= 7. 找关键index =========
    stage_idx = {c: i for i, c in enumerate(trajectory)}
    rg_idx = z.columns.get_loc("RG")
    en_idx = z.columns.get_loc("EN")
    ipcen_idx = z.columns.get_loc("IPC-EN")

    # ========= 8. permutation函数 =========
    def permutation_pvalue(v, observed_score):
        null = []
        for _ in range(n_permutations):
            perm = np.random.permutation(v)
            score = perm[-1] - np.mean(perm[:-1])
            null.append(score)
        null = np.array(null)
        return (np.sum(null >= observed_score) + 1) / (len(null) + 1)

    transient_results = []
    late_results = []

    # ========= 9. 主循环 =========
    for gene, row in z.iterrows():
        v = row.values
        # ---------- A. transient IPC-EN ----------
        if ipcen_idx is not None:

            ipcen_val = v[ipcen_idx]
            rg_val = v[rg_idx]
            en_val = v[en_idx]
            p_trend = abs(rg_val - en_val)

            # 判定条件
            # if (
            #     ipcen_val > rg_val + 0.5 and
            #     ipcen_val > en_val + 0.5 and
            #     ipcen_val > np.mean(v) + 0.5
            # ):
            if ipcen_val > rg_val and ipcen_val > en_val and rg_val > 0 and en_val > 0:
                score = 2 * ipcen_val - (rg_val + en_val)
                pval = permutation_pvalue(v, score)

                transient_results.append({
                    "gene": gene,
                    "ipcen_val": float(ipcen_val),
                    "rg_val": float(rg_val),
                    "en_val": float(en_val),
                    "log2FC_vs_prev": float(ipcen_val - rg_val),
                    "log2FC_vs_next": float(ipcen_val - en_val),
                    "score": float(score),
                    "p_trend": float(p_trend)
                })

        # ---------- B. late EN ----------
        if en_idx is not None:

            en_val = v[en_idx]
            prev_mean = np.mean(np.delete(v, en_idx))
            # rg_val = v[en_i - 1]
            # ipcen_val = v[en_i - 2]



            rg_val = v[rg_idx]
            ipcen_val = v[ipcen_idx]

            # score = en_val - prev_mean
            # score = 2 * en_val - rg_val - ipcen_val

            score1 = en_val - ipcen_val
            score2 = en_val - rg_val
            score3 = ipcen_val - rg_val
            score_trend = ipcen_val - rg_val
            # score = en_val / (ipcen_val + 1e-10)  + en_val / (rg_val + 1e-10)
            score = score1

            # 单调性
            trend = np.all(np.diff(v) >= -0.2)

            # if en_val > prev_mean + 0.8:
            if score1 > 0 and score2 > 0 and score3 >=0 :
                pval = permutation_pvalue(v, score1)
                late_results.append({
                    "gene": gene,
                    "en_val": float(en_val),
                    "rg_val": float(rg_val),
                    "ipcen_val": float(ipcen_val),
                    "en_vs_ipcen": float(score1),
                    "en_vs_rg": float(score2),
                    "score": float(score),
                    "trend": "increasing" if trend else "non-monotonic",
                    "score_trend": float(score_trend)
                })

    # ========= 10. 排序 =========
    transient_results = sorted(
        transient_results,
        key=lambda x: (-x["rg_val"], x["log2FC_vs_next"]),
        reverse=True
    )[:top_n]

    transient_results_summary = {
        "genes": ';'.join(list(item['gene'] for item in reversed(transient_results))),
    }

    late_results = sorted(
        late_results,
        key=lambda x: (-x["rg_val"], x["en_vs_ipcen"]),
        reverse=True
    )[:top_n]

    late_results_summary = {
        "genes": ';'.join(list(item['gene'] for item in reversed(late_results))),
    }

    # transient_results_new = []
    # for idx, temp in enumerate(transient_results):
        # temp["idx"] = idx
        # transient_results_new.append(temp)

    # late_results_new = []
    # for idx, temp in enumerate(late_results):
        # temp["idx"] = idx
        # late_results_new.append(temp)
    # transient_results = [temp["idx"]=idx for idx, temp in enumerate(transient_results)]
    # ========= 11. 输出 =========

    transient_results_new = []
    for idx, temp in enumerate(reversed(transient_results)):
        transient_results_new.append({
            "gene": temp["gene"],
            "ipcen_val": temp["ipcen_val"],
            "rank": idx,
        })

    late_results_new = []
    for idx, temp in enumerate(reversed(late_results)):
        late_results_new.append({
            "gene": temp["gene"],
            "en_val": temp["en_val"],
            "rank": idx,
        })

    llm_summary = {
        "trajectory": trajectory,
        # "transient_IPC-EN": transient_results_summary,
        # "late_EN": late_results_summary
        "transient_IPC-EN": transient_results_new,
        "late_EN": late_results_new,
        "description": "The key 'rank' represents importance, with smaller values indicating greater importance, the report should contain gene with small rank."
    }

    llm_summary = _round_dict(llm_summary)
    output_path = os.path.join(output_folder, 'trajectory_summary.json')
    save_json(llm_summary, output_path)

    return llm_summary


def build_pli_reference(output_path="pli_reference.csv"):
    """
    从 gnomAD 构建 pLI 参考文件
    """
    url = "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
    local_file = os.path.join("./kb_source", "gnomad_lof_metrics.bgz")
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    if not os.path.exists(local_file):
        r = _request_with_retry("GET", url, stream=True, timeout=60)
        with open(local_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    df = pd.read_csv(local_file, sep="\t", compression="gzip")
    pli_df = df[["gene", "pLI"]].dropna()
    pli_df.to_csv(output_path, index=False)
    return pli_df

@tool
def add_pli_to_protein(
        *args, **kwargs
):
    """
    构建所有蛋白的 pLI 并合并到蛋白数据
    """

    this_run_folder = resolve_path('this_run_folder_path')
    protein_quant_path = resolve_path('protein_quant_path')

    output_folder = os.path.join(this_run_folder, 'pil_results')
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "protein_with_pLI.csv")

    if os.path.exists(protein_quant_path):
        data_pq = pd.read_csv(protein_quant_path)
    else:
        raise FileNotFoundError(f"The file of {protein_quant_path} is not found.")

    pil_reference_path = os.path.join(output_folder, "pli_reference.csv")
    pli_df = build_pli_reference(output_path=pil_reference_path)

    # 假设格式：TP53 或 P04637;Q9Y6X9
    data_pq["gene"] = data_pq["PG.Genes"].str.split(";").str[0]

    merged = data_pq.merge(
        pli_df,
        on="gene",
        how="left"
    )

    merged.to_csv(output_path, index=False)

    top_n = 100
    # 1️⃣ 基本统计
    total_proteins = merged.shape[0]
    with_pli = merged["pLI"].notna().sum()
    high_pli = (merged["pLI"] > 0.9).sum()

    # 2️⃣ 高 pLI 蛋白（关键）
    high_pli_df = merged[merged["pLI"] > 0.9] \
        .sort_values("pLI", ascending=False) \
        .head(top_n)

    # 3️⃣ 中等 pLI
    mid_pli_df = merged[
        (merged["pLI"] > 0.1) & (merged["pLI"] <= 0.9)
    ].head(top_n)

    # 4️⃣ 低 pLI
    low_pli_df = merged[
        (merged["pLI"] <= 0.1)
    ].head(top_n)

    # 5️⃣ 构建结构化输出
    llm_output = {
        "summary": {
            "total_proteins": int(total_proteins),
            "proteins_with_pLI": int(with_pli),
            "high_pLI_proteins": int(high_pli),
            "coverage_ratio": float(with_pli / total_proteins)
        },
        "top_high_pLI": high_pli_df[["gene", "pLI"]].to_dict(orient="records"),
        "top_mid_pLI": mid_pli_df[["gene", "pLI"]].to_dict(orient="records"),
        "top_low_pLI": low_pli_df[["gene", "pLI"]].to_dict(orient="records")
    }

    return llm_output


@tool
def cluster_distribution(
        *args, **kwargs,
):
    """
    统计输入数据中不同 Cluster 的分布情况，包括全局分布和按分组的分布。
    """

    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo_path = resolve_path('sampleinfo_path')
    sampleinfo = pd.read_csv(sampleinfo_path)

    if "Cluster" not in sampleinfo.columns:
        raise ValueError("sampleinfo.csv must contain 'Cluster' column")

    results = {}

    # =========================
    # 1️⃣ 全局 Cluster 分布
    # =========================
    total_count = len(sampleinfo)

    cluster_counts = sampleinfo["Cluster"].value_counts().sort_index()
    cluster_ratio = cluster_counts / total_count
    overall_df = pd.DataFrame({
        "count": cluster_counts,
        "ratio": cluster_ratio
    })
    results["overall"] = overall_df.to_dict(orient="index")
    type_cols = [col for col in sampleinfo.columns if col.startswith("Type")]

    for t in sampleinfo["Cluster"].dropna().unique():
        culster_result = {}
        for type_col in type_cols:
            sub_df = sampleinfo[sampleinfo["Cluster"] == t]
            type_counts = sub_df[type_col].value_counts().sort_index()
            type_ratio = type_counts / len(sub_df)

            type_df = pd.DataFrame({
                "count": type_counts,
                "ratio": type_ratio
            })
            culster_result[type_col] = type_df.to_dict(orient="index")
        results[t] = culster_result

    # =========================
    # 3️⃣ 分组统计
    # =========================
    for type_col in type_cols:
        group_result = {}
        for t in sampleinfo[type_col].dropna().unique():
            sub_df = sampleinfo[sampleinfo[type_col] == t]

            count = sub_df["Cluster"].value_counts().sort_index()
            ratio = count / len(sub_df)

            group_df = pd.DataFrame({
                "count": count,
                "ratio": ratio
            })
            group_result[t] = group_df.to_dict(orient="index")
        results[type_col] = group_result

    # =========================
    # 4️⃣ 多 Type 组合下的 Cluster 分布
    # =========================
    if len(type_cols) > 1:
        combo_result = {}

        # 去掉 NA（避免组合错误）
        sub_df = sampleinfo.dropna(subset=type_cols + ["Cluster"])

        # 统计 count
        combo_counts = (
            sub_df
            .groupby(type_cols + ["Cluster"])
            .size()
            .reset_index(name="count")
        )

        # 计算每个组合的总数（用于算比例）
        combo_counts["total"] = combo_counts.groupby(type_cols)["count"].transform("sum")
        combo_counts["ratio"] = combo_counts["count"] / combo_counts["total"]

        # 转成嵌套 dict
        for _, row in combo_counts.iterrows():
            key = ' and '.join([row[col] for col in type_cols])
            cluster = row["Cluster"]

            if key not in combo_result:
                combo_result[key] = {}

            combo_result[key][cluster] = {
                "count": int(row["count"]),
                "ratio": float(row["ratio"])
            }

        results["type_combination"] = combo_result

    results = _round_dict(results, 4)

    output_folder = os.path.join(this_run_folder, 'cluster_distribution')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'distribution_summary.json')
    save_json(results, output_path)

    return results



def _run_membrane_status_leakage(eps: float = 1e-8, corr_threshold: float = 0.6) -> Dict[str, Any]:
    this_run_folder = resolve_path('this_run_folder_path')
    output_folder = os.path.join(this_run_folder, 'proteins_leakage')
    os.makedirs(output_folder, exist_ok=True)

    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    protein_quant_combat_path = os.path.join(processed_proteins_folder, "ProteinQuant_ComBat.csv")
    if not os.path.exists(protein_quant_combat_path):
        _ = combat_calibration.func()
    protein_quant_raw = pd.read_csv(protein_quant_combat_path)

    sampleinfo_path = resolve_path('sampleinfo_path')
    sampleinfo = pd.read_csv(sampleinfo_path)
    sample_col = _sample_id_col(sampleinfo)
    if sample_col != "FileName":
        sampleinfo = sampleinfo.copy()
        sampleinfo["FileName"] = sampleinfo[sample_col].astype(str)

    feature_col = protein_quant_raw.columns[0]
    gene_lookup: Dict[str, str] = {}
    if "PG.Genes" in protein_quant_raw.columns:
        gene_lookup.update(
            protein_quant_raw.assign(_protein=protein_quant_raw[feature_col].astype(str).str.split(";").str[0])
            .drop_duplicates("_protein")
            .set_index("_protein")["PG.Genes"]
            .astype(str)
            .to_dict()
        )
    protein_gene_map_path = os.path.join(processed_proteins_folder, "Protein_Gene_Map.csv")
    if os.path.exists(protein_gene_map_path):
        try:
            protein_gene_map = pd.read_csv(protein_gene_map_path)
            if {"PG.ProteinGroups", "PG.Genes"}.issubset(protein_gene_map.columns):
                gene_lookup.update(
                    protein_gene_map.assign(_protein=protein_gene_map["PG.ProteinGroups"].astype(str).str.split(";").str[0])
                    .drop_duplicates("_protein")
                    .set_index("_protein")["PG.Genes"]
                    .astype(str)
                    .to_dict()
                )
        except Exception:
            pass

    sample_names = [s for s in sampleinfo["FileName"].dropna().astype(str).tolist() if s in protein_quant_raw.columns]
    if not sample_names:
        raise ValueError("No SampleInfo FileName values matched ProteinQuant_ComBat sample columns.")

    protein_quant = protein_quant_raw.set_index(feature_col)
    numeric_matrix = protein_quant.loc[:, sample_names].apply(pd.to_numeric, errors="coerce")
    max_val = float(np.nanmax(numeric_matrix.to_numpy(dtype=float)))
    data_mode = "log" if max_val < 100 else "linear"

    def _status_columns(df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        fallback_type_cols: List[str] = []
        for col in df.columns:
            if col == "FileName":
                continue
            values = [v for v in df[col].dropna().astype(str).str.strip().unique().tolist() if v]
            lower = {v.lower() for v in values}
            if {"intact", "permeable"}.issubset(lower):
                cols.append(col)
            elif re.match(r"^Type[0-9]+$", str(col)) and 2 <= len(values) <= 8:
                fallback_type_cols.append(col)
        return list(dict.fromkeys(cols or fallback_type_cols))

    status_cols = _status_columns(sampleinfo)
    strat_col = "CellType" if "CellType" in sampleinfo.columns else "Cluster" if "Cluster" in sampleinfo.columns else None
    results: Dict[str, Any] = {}
    module_rows: List[Dict[str, Any]] = []
    module_sets = {
        "protein_leakage_cytosol_nucleus": ["GAPDH", "LDHA", "ENO1", "TUBA1B", "ACTB", "LMNB1", "HIST1H1A", "HNRNPA1"],
        "protein_leakage_mito_membrane": ["VDAC1", "VDAC2", "ATP5F1A", "ATP5F1B", "COX4I1", "TOMM20", "SLC25A3"],
    }

    def _norm_gene(value: Any) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

    def _split_gene(value: Any) -> List[str]:
        if pd.isna(value):
            return []
        return [t.strip() for t in re.split(r"[;,|/\s]+", str(value)) if t.strip()]

    canonical_index = pd.Series(numeric_matrix.index.astype(str), index=numeric_matrix.index).str.split(";").str[0]
    row_gene_tokens: Dict[Any, set[str]] = {}
    for row_id, protein_id in canonical_index.items():
        gene_text = gene_lookup.get(str(protein_id), "")
        row_gene_tokens[row_id] = {_norm_gene(t) for t in _split_gene(gene_text)}

    def _module_row_ids(genes: List[str]):
        wanted = {_norm_gene(g) for g in genes}
        row_ids = []
        matched = set()
        for row_id, tokens in row_gene_tokens.items():
            hits = wanted.intersection(tokens)
            if hits:
                row_ids.append(row_id)
                for gene in genes:
                    if _norm_gene(gene) in hits:
                        matched.add(gene)
        return row_ids, sorted(matched)

    for status_col in status_cols:
        groups = [g for g in sampleinfo[status_col].dropna().astype(str).str.strip().unique().tolist() if g]
        groups = sorted(groups, key=lambda x: (str(x).lower() != "intact", str(x).lower() != "permeable", str(x)))
        for g1, g2 in combinations(groups, 2):
            strata = sampleinfo[strat_col].dropna().astype(str).unique().tolist() if strat_col else ["all_samples"]
            logfc_dict: Dict[str, pd.Series] = {}
            for stratum in strata:
                sub_info = sampleinfo if not strat_col else sampleinfo[sampleinfo[strat_col].astype(str) == str(stratum)]
                g1_samples = [s for s in sub_info.loc[sub_info[status_col].astype(str) == str(g1), "FileName"].astype(str) if s in numeric_matrix.columns]
                g2_samples = [s for s in sub_info.loc[sub_info[status_col].astype(str) == str(g2), "FileName"].astype(str) if s in numeric_matrix.columns]
                if len(g1_samples) == 0 or len(g2_samples) == 0:
                    continue
                g1_mean = numeric_matrix.loc[:, g1_samples].mean(axis=1)
                g2_mean = numeric_matrix.loc[:, g2_samples].mean(axis=1)
                log2fc = g1_mean - g2_mean if data_mode == "log" else np.log2((g1_mean + eps) / (g2_mean + eps))
                logfc_dict[str(stratum)] = log2fc

            logfc_df = pd.DataFrame(logfc_dict).replace([np.inf, -np.inf], np.nan).dropna()
            if logfc_df.shape[0] >= 10 and logfc_df.shape[1] >= 2:
                corr_matrix = logfc_df.corr(method="pearson")
                high_corr_pairs = []
                for i in corr_matrix.columns:
                    for j in corr_matrix.columns:
                        if str(i) >= str(j):
                            continue
                        r = corr_matrix.loc[i, j]
                        if pd.notna(r) and r >= corr_threshold:
                            high_corr_pairs.append([str(i), str(j), round(float(r), 4)])
                conclusion = "Global protein leakage across cell types" if high_corr_pairs else "No strong global leakage pattern"
                corr_payload = corr_matrix.round(4).to_dict()
            else:
                high_corr_pairs = []
                conclusion = "Insufficient stratified data for leakage-correlation test"
                corr_payload = {}

            pair_key = f"{g1}_vs_{g2}"
            results[pair_key] = {
                "status_column": status_col,
                "stratification_col": strat_col or "all_samples",
                "data_mode": data_mode,
                "logfc_matrix_shape": list(logfc_df.shape),
                "high_correlation_pairs": high_corr_pairs,
                "conclusion": conclusion,
                "correlation_matrix": corr_payload,
            }

            all_g1 = [s for s in sampleinfo.loc[sampleinfo[status_col].astype(str) == str(g1), "FileName"].astype(str) if s in numeric_matrix.columns]
            all_g2 = [s for s in sampleinfo.loc[sampleinfo[status_col].astype(str) == str(g2), "FileName"].astype(str) if s in numeric_matrix.columns]
            for module_name, genes in module_sets.items():
                row_ids, matched_genes = _module_row_ids(genes)
                mean_logfc = None
                if row_ids and all_g1 and all_g2:
                    g1_mean = numeric_matrix.loc[row_ids, all_g1].mean(axis=1)
                    g2_mean = numeric_matrix.loc[row_ids, all_g2].mean(axis=1)
                    module_fc = g1_mean - g2_mean if data_mode == "log" else np.log2((g1_mean + eps) / (g2_mean + eps))
                    mean_logfc = round(float(module_fc.mean()), 5) if module_fc.notna().any() else None
                direction = (
                    f"higher_in_{g1}" if mean_logfc is not None and mean_logfc > 0
                    else f"higher_in_{g2}" if mean_logfc is not None and mean_logfc < 0
                    else "not_applicable"
                )
                module_rows.append({
                    "comparison": pair_key,
                    "status_column": status_col,
                    "module": module_name,
                    "matched_genes": ";".join(matched_genes),
                    "n_matched_genes": int(len(matched_genes)),
                    "mean_logFC_group_a_minus_group_b": mean_logfc,
                    "group_a": g1,
                    "group_b": g2,
                    "direction": direction,
                    "n_samples_group_a": int(len(all_g1)),
                    "n_samples_group_b": int(len(all_g2)),
                    "evidence_source": "current_matrix",
                    "confidence": "moderate" if len(matched_genes) >= 3 else "low",
                })

    module_summary_path = os.path.join(output_folder, "leakage_module_summary.csv")
    pd.DataFrame(module_rows).to_csv(module_summary_path, index=False, encoding="utf-8-sig")
    output_path = os.path.join(output_folder, 'leakage_summary.json')
    save_json(results, output_path)
    append_evidence(
        this_run_folder,
        source="current_matrix",
        tool="run_protein_leakage_analysis",
        claim="Membrane-status protein leakage analysis completed using current matrix contrasts and stratified correlation checks.",
        files=[output_path, module_summary_path],
        metrics={
            "status_columns": status_cols,
            "stratification_col": strat_col,
            "n_status_comparisons": len(results),
            "n_module_rows": len(module_rows),
            "corr_threshold": corr_threshold,
        },
        confidence=_confidence_from_design_and_backend("python_fallback" if not R_AVAILABLE else "r_limma"),
        limitations=[],
    )
    return results

@tool
def run_protein_leakage_analysis(
    eps: float = 1e-8,
    corr_threshold: float = 0.6,
    *args, **kwargs
):
    """
    蛋白质跨细胞类型一致性（泄漏）分析

    参数：
    ----------
    eps : float
        防止除0
    corr_threshold : float
        判定“高相关”的阈值

    返回：
    ----------
    results : dict
        每个 Type 对比的分析结果
    """
    return _run_membrane_status_leakage(eps=eps, corr_threshold=corr_threshold)


    this_run_folder = resolve_path('this_run_folder_path')

    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")
    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    protein_quant_combat_path = os.path.join(processed_proteins_folder, "ProteinQuant_ComBat.csv")
    protein_quant = pd.read_csv(protein_quant_combat_path)

    sampleinfo_path = resolve_path('sampleinfo_path')
    sampleinfo = pd.read_csv(sampleinfo_path)


    protein_quant = protein_quant.set_index(protein_quant.columns[0])

    # ========= 1. 自动判断数据类型 =========
    max_val = protein_quant.max().max()

    if max_val < 100:
        data_mode = "log"     # log2空间（含ComBat）
    else:
        data_mode = "linear"  # 原始强度

    # ========= 2. 找到所有 Type 列 =========
    type_cols = [col for col in sampleinfo.columns if col.startswith("Type")]

    results = {}

    # ========= 3. 主循环 =========
    for type_col in type_cols:
        groups = sampleinfo[type_col].dropna().unique().tolist()
        pairs = list(combinations(groups, 2))
        for g1, g2 in pairs:
            clusters = sampleinfo["Cluster"].dropna().unique()
            logfc_dict = {}
            # ========= 4. 每个 Cluster 计算 log2FC =========
            for cluster in clusters:
                sub_info = sampleinfo[sampleinfo["Cluster"] == cluster]
                g1_samples = sub_info[sub_info[type_col] == g1]["FileName"]
                g2_samples = sub_info[sub_info[type_col] == g2]["FileName"]
                # 跳过样本不足情况
                if len(g1_samples) == 0 or len(g2_samples) == 0:
                    continue

                g1_mean = protein_quant[g1_samples].mean(axis=1)
                g2_mean = protein_quant[g2_samples].mean(axis=1)

                # ========= 自动选择 log2FC 计算方式 =========
                if data_mode == "log":
                    log2fc = g1_mean - g2_mean
                else:
                    log2fc = np.log2((g1_mean + eps) / (g2_mean + eps))
                logfc_dict[cluster] = log2fc
            # ========= 5. 构建矩阵 =========
            logfc_df = pd.DataFrame(logfc_dict)
            # 对齐蛋白（关键）
            logfc_df = logfc_df.replace([np.inf, -np.inf], np.nan).dropna()
            # 如果数据太少，跳过
            if logfc_df.shape[0] < 10 or logfc_df.shape[1] < 2:
                continue
            # ========= 6. 计算相关性 =========
            corr_matrix = logfc_df.corr(method="pearson")
            # ========= 7. 判断是否全局泄漏 =========
            high_corr_pairs = []
            for i in corr_matrix.columns:
                for j in corr_matrix.columns:
                    if i >= j:
                        continue
                    r = corr_matrix.loc[i, j]
                    if r >= corr_threshold:
                        high_corr_pairs.append([i, j, round(float(r), 4)])

            if len(high_corr_pairs) > 0:
                conclusion = "Global protein leakage across cell types"
            else:
                conclusion = "No strong global leakage pattern"

            # ========= 8. 保存结果 =========
            key = f"{g1}_vs_{g2}"

            results[key] = {
                "data_mode": data_mode,
                "logfc_matrix_shape": list(logfc_df.shape),
                "correlation_matrix": corr_matrix.round(4).to_dict(orient='records'),
                "high_correlation_pairs": high_corr_pairs,
                "conclusion": conclusion
            }

    output_folder = os.path.join(this_run_folder, 'proteins_leakage')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'leakage_summary.json')
    save_json(results, output_path)

    return results


def _canonical_gene_name(value):
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text:
        return "NA"
    return text.split(";")[0].strip()


def _get_feature_column(df: pd.DataFrame) -> str:
    for col in ["PG.Genes", "gene", "Gene", "Genes", "PG.ProteinGroups"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _load_sampleinfo_with_cluster():
    sampleinfo_path = resolve_path('sampleinfo_path')
    if os.path.exists(sampleinfo_path):
        sampleinfo = pd.read_csv(sampleinfo_path)
    else:
        raise FileNotFoundError(f"The file of {sampleinfo_path} is not found.")

    if "Cluster" not in sampleinfo.columns:
        this_run_folder = resolve_path('this_run_folder_path')
        processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
        sampleinfo_with_cluster_path = os.path.join(processed_proteins_folder, "sampleinfo_with_clusters.csv")
        if os.path.exists(sampleinfo_with_cluster_path):
            sampleinfo = pd.read_csv(sampleinfo_with_cluster_path)

    return sampleinfo


def _load_protein_matrix_for_analysis(use_combat: bool = True):
    this_run_folder = resolve_path('this_run_folder_path')
    sampleinfo = _load_sampleinfo_with_cluster()

    if use_combat:
        processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
        os.makedirs(processed_proteins_folder, exist_ok=True)
        matrix_path = os.path.join(processed_proteins_folder, 'ProteinQuant_ComBat.csv')
        if not os.path.exists(matrix_path):
            _ = combat_calibration.func()
    else:
        matrix_path = resolve_path('protein_quant_path')

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"The file of {matrix_path} is not found.")

    df = pd.read_csv(matrix_path)
    feature_col = _get_feature_column(df)
    if "FileName" not in sampleinfo.columns:
        raise ValueError("sampleinfo.csv must contain 'FileName' column.")

    sample_cols = [c for c in sampleinfo["FileName"].astype(str).tolist() if c in df.columns]
    if not sample_cols:
        raise ValueError("No sample columns from sampleinfo.FileName were found in the protein matrix.")

    numeric_df = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    feature_series = df[feature_col].apply(_canonical_gene_name)
    return this_run_folder, df, sampleinfo, feature_col, sample_cols, numeric_df, feature_series


def _load_differential_tables():
    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(processed_proteins_folder, exist_ok=True)

    diff_files = [
        fn for fn in os.listdir(processed_proteins_folder)
        if fn.startswith("differential_") and fn.endswith(".csv")
    ]
    if not diff_files:
        _ = run_pairwise_limma.func()
        diff_files = [
            fn for fn in os.listdir(processed_proteins_folder)
            if fn.startswith("differential_") and fn.endswith(".csv")
        ]

    if not diff_files:
        raise FileNotFoundError("No differential protein files found in processed_proteins folder.")

    tables = {}
    for fn in diff_files:
        contrast = fn.replace("differential_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(processed_proteins_folder, fn))
        tables[contrast] = df
    return this_run_folder, processed_proteins_folder, tables


def _compute_marker_specificity_table(use_combat: bool = True):
    _, _, sampleinfo, _, sample_cols, numeric_df, feature_series = _load_protein_matrix_for_analysis(use_combat=use_combat)

    if "Cluster" not in sampleinfo.columns:
        raise ValueError("sampleinfo.csv must contain 'Cluster' column or sampleinfo_with_clusters.csv must be available.")

    clusters = [c for c in sampleinfo["Cluster"].dropna().astype(str).unique().tolist() if c]
    if len(clusters) < 2:
        raise ValueError("Marker specificity analysis requires at least two clusters.")

    cluster_to_samples = {}
    for cluster in clusters:
        cols = sampleinfo.loc[sampleinfo["Cluster"].astype(str) == cluster, "FileName"].astype(str).tolist()
        cols = [c for c in cols if c in sample_cols]
        if cols:
            cluster_to_samples[cluster] = cols

    if len(cluster_to_samples) < 2:
        raise ValueError("At least two clusters with valid samples are required for marker specificity analysis.")

    cluster_mean_df = pd.DataFrame({
        cluster: numeric_df[cols].mean(axis=1, skipna=True)
        for cluster, cols in cluster_to_samples.items()
    })

    cluster_mean_df = cluster_mean_df.replace([np.inf, -np.inf], np.nan)
    if cluster_mean_df.shape[1] < 2:
        raise ValueError("Marker specificity analysis requires at least two cluster mean columns.")

    top_cluster = cluster_mean_df.idxmax(axis=1)
    top_mean = cluster_mean_df.max(axis=1)
    second_mean = cluster_mean_df.apply(
        lambda row: row.nlargest(2).iloc[-1] if row.notna().sum() >= 2 else np.nan,
        axis=1
    )
    n_cluster_means = cluster_mean_df.notna().sum(axis=1)
    mean_other = (cluster_mean_df.sum(axis=1) - top_mean) / (n_cluster_means - 1).replace(0, np.nan)
    specificity_score = top_mean - second_mean

    marker_df = pd.DataFrame({
        "gene": feature_series,
        "best_cluster": top_cluster,
        "best_cluster_mean": top_mean,
        "second_cluster_mean": second_mean,
        "mean_other_clusters": mean_other,
        "specificity_score": specificity_score,
    })
    marker_df = marker_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["specificity_score", "best_cluster"])
    marker_df = marker_df.sort_values(["best_cluster", "specificity_score"], ascending=[True, False])

    return marker_df, cluster_mean_df, cluster_to_samples


@tool
def run_missingness_analysis(
    use_combat: bool = True,
    top_n: int = 20,
    *args, **kwargs
):
    """
    分析样本与蛋白层面的缺失值模式，并评估不同分组中的缺失偏倚。
    """
    this_run_folder, _, sampleinfo, _, sample_cols, numeric_df, feature_series = _load_protein_matrix_for_analysis(use_combat=use_combat)

    missing_mask = numeric_df.isna()
    protein_missing = missing_mask.mean(axis=1)
    sample_missing = missing_mask.mean(axis=0)

    summary = {
        "matrix_source": "ProteinQuant_ComBat.csv" if use_combat else "ProteinQuant.csv",
        "n_proteins": int(numeric_df.shape[0]),
        "n_samples": int(numeric_df.shape[1]),
        "overall_missing_ratio": float(missing_mask.values.mean()),
        "protein_missing_ratio_summary": {
            "mean": float(protein_missing.mean()),
            "median": float(protein_missing.median()),
            "high_missing_ratio_proteins": int((protein_missing >= 0.5).sum()),
        },
        "sample_missing_ratio_summary": {
            "mean": float(sample_missing.mean()),
            "median": float(sample_missing.median()),
            "high_missing_ratio_samples": int((sample_missing >= 0.3).sum()),
        },
        "top_missing_samples": [
            {"sample": sample, "missing_ratio": round(float(rate), 4)}
            for sample, rate in sample_missing.sort_values(ascending=False).head(top_n).items()
        ],
        "top_missing_proteins": [
            {"gene": gene, "missing_ratio": round(float(rate), 4)}
            for gene, rate in pd.DataFrame({"gene": feature_series, "missing_ratio": protein_missing})
            .sort_values("missing_ratio", ascending=False)
            .head(top_n)
            .itertuples(index=False, name=None)
        ],
    }

    if "FileName" in sampleinfo.columns:
        sample_level = sampleinfo.copy()
        sample_level["missing_ratio"] = sample_level["FileName"].map(sample_missing).fillna(np.nan)
        for col in [c for c in sampleinfo.columns if c.startswith("Type") or c.startswith("Cluster")]:
            temp = sample_level.dropna(subset=[col, "missing_ratio"]).groupby(col)["missing_ratio"].agg(["mean", "median", "count"])
            summary[f"{col}_missing_summary"] = _round_dict(temp.to_dict(orient="index"))

    overall_missing = float(missing_mask.values.mean())
    if overall_missing >= 0.4:
        mechanism = "Missingness is heavy and may include strong MNAR effects."
    elif overall_missing >= 0.2:
        mechanism = "Missingness is moderate and may reflect both technical dropout and biological sparsity."
    else:
        mechanism = "Missingness is relatively limited and less likely to dominate downstream analysis."
    summary["heuristic_interpretation"] = mechanism

    summary = _round_dict(summary)
    output_folder = os.path.join(this_run_folder, 'missingness_results')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'missingness_summary.json')
    save_json(summary, output_path)

    return summary


@tool
def run_sample_correlation_qc(
    method: str = "spearman",
    top_n_outliers: int = 10,
    use_combat: bool = True,
    *args, **kwargs
):
    """
    评估样本间相关性，识别低相关的潜在异常样本。
    """
    this_run_folder, _, sampleinfo, _, sample_cols, numeric_df, _ = _load_protein_matrix_for_analysis(use_combat=use_combat)

    method = str(method).lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'.")

    corr_df = numeric_df.corr(method=method)
    corr_no_diag = corr_df.mask(np.eye(len(corr_df), dtype=bool))
    median_corr = corr_no_diag.median(axis=1, skipna=True)
    q1 = float(median_corr.quantile(0.25))
    q3 = float(median_corr.quantile(0.75))
    iqr = q3 - q1
    outlier_cutoff = q1 - 1.5 * iqr if iqr > 0 else q1

    sample_summary_df = pd.DataFrame({
        "sample": sample_cols,
        "median_correlation": median_corr.reindex(sample_cols).values,
    }).sort_values("median_correlation", ascending=True)
    sample_summary_df["is_low_correlation_outlier"] = sample_summary_df["median_correlation"] < outlier_cutoff

    summary = {
        "matrix_source": "ProteinQuant_ComBat.csv" if use_combat else "ProteinQuant.csv",
        "method": method,
        "n_samples": int(len(sample_cols)),
        "global_correlation_summary": {
            "median_of_sample_medians": float(median_corr.median()),
            "min_sample_median_correlation": float(median_corr.min()),
            "max_sample_median_correlation": float(median_corr.max()),
            "low_correlation_outlier_cutoff": float(outlier_cutoff),
            "n_low_correlation_outliers": int(sample_summary_df["is_low_correlation_outlier"].sum()),
        },
        "lowest_correlation_samples": _round_dict(
            sample_summary_df.head(top_n_outliers).to_dict(orient="records")
        ),
    }

    if "FileName" in sampleinfo.columns:
        enriched = sampleinfo.copy()
        enriched["median_correlation"] = enriched["FileName"].map(median_corr)
        for col in [c for c in sampleinfo.columns if c.startswith("Type") or c.startswith("Cluster")]:
            temp = enriched.dropna(subset=[col, "median_correlation"]).groupby(col)["median_correlation"].agg(["mean", "median", "count"])
            summary[f"{col}_correlation_summary"] = _round_dict(temp.to_dict(orient="index"))

    output_folder = _ensure_dir(os.path.join(this_run_folder, 'sample_correlation_qc'))
    _safe_to_csv(corr_df.round(4), os.path.join(output_folder, f"sample_correlation_matrix_{method}.csv"), encoding='utf-8-sig')
    save_json(_round_dict(summary), os.path.join(output_folder, 'sample_correlation_summary.json'))

    return _round_dict(summary)


# @tool
# def run_marker_specificity_analysis(
#     top_n: int = 20,
#     use_combat: bool = True,
#     *args, **kwargs
# ):
#     """
#     评估蛋白在不同 cluster 之间的特异性，并为每个 cluster 输出 top markers。
#     """
#     this_run_folder = resolve_path('this_run_folder_path')
#     marker_df, cluster_mean_df, cluster_to_samples = _compute_marker_specificity_table(use_combat=use_combat)

#     results = {
#         "matrix_source": "ProteinQuant_ComBat.csv" if use_combat else "ProteinQuant.csv",
#         "n_clusters": int(len(cluster_to_samples)),
#         "cluster_sample_sizes": {k: len(v) for k, v in cluster_to_samples.items()},
#         "clusters": {},
#     }

#     for cluster in sorted(cluster_to_samples.keys()):
#         sub = marker_df[marker_df["best_cluster"] == cluster].head(top_n).copy()
#         sub["rank"] = range(1, len(sub) + 1)
#         results["clusters"][cluster] = _round_dict(sub[
#             ["rank", "gene", "specificity_score", "best_cluster_mean", "second_cluster_mean", "mean_other_clusters"]
#         ].to_dict(orient="records"))

#     global_top = marker_df.sort_values("specificity_score", ascending=False).head(top_n).copy()
#     global_top["rank"] = range(1, len(global_top) + 1)
#     results["global_top_specific_markers"] = _round_dict(global_top[
#         ["rank", "gene", "best_cluster", "specificity_score", "best_cluster_mean", "second_cluster_mean"]
#     ].to_dict(orient="records"))

#     output_folder = os.path.join(this_run_folder, 'marker_specificity')
#     os.makedirs(output_folder, exist_ok=True)
#     save_json(results, os.path.join(output_folder, 'marker_specificity_summary.json'))
#     export_df = marker_df.copy()
#     export_df = export_df.join(cluster_mean_df.add_prefix("cluster_mean_"))
#     export_df.to_csv(os.path.join(output_folder, 'marker_specificity_table.csv'), index=False, encoding='utf-8-sig')

#     return results


@tool
def run_overlap_analysis_between_contrasts(
    p_thresh: float = 0.05,
    logfc_thresh: float = 0.0,
    top_n: int = 20,
    *args, **kwargs
):
    """
    统计不同 contrasts 之间差异蛋白的重叠关系，包括同向与反向重叠。
    """
    this_run_folder, _, tables = _load_differential_tables()

    contrast_sets = {}
    for contrast, df in tables.items():
        gene_col = "PG.Genes" if "PG.Genes" in df.columns else _get_feature_column(df)
        p_col = "adj.P.Val" if "adj.P.Val" in df.columns else "P.Value"
        if p_col not in df.columns or "logFC" not in df.columns:
            continue
        sub = df[[gene_col, p_col, "logFC"]].copy()
        sub[gene_col] = sub[gene_col].apply(_canonical_gene_name)
        sub = sub.dropna(subset=[gene_col, p_col, "logFC"])
        up = set(sub[(sub[p_col] < p_thresh) & (sub["logFC"] > logfc_thresh)][gene_col].tolist())
        down = set(sub[(sub[p_col] < p_thresh) & (sub["logFC"] < -logfc_thresh)][gene_col].tolist())
        contrast_sets[contrast] = {"up": up, "down": down}

    pairwise_results = {}
    strongest_same = []
    strongest_opposite = []

    def _jaccard(a, b):
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    for c1, c2 in combinations(sorted(contrast_sets.keys()), 2):
        up1, down1 = contrast_sets[c1]["up"], contrast_sets[c1]["down"]
        up2, down2 = contrast_sets[c2]["up"], contrast_sets[c2]["down"]

        same_up = sorted(up1 & up2)
        same_down = sorted(down1 & down2)
        opposite_up_down = sorted(up1 & down2)
        opposite_down_up = sorted(down1 & up2)

        pair_key = normalize_contrast(f"{c1}_vs_{c2}")
        pair_result = {
            "same_direction_overlap": {
                "up_up_count": len(same_up),
                "down_down_count": len(same_down),
                "jaccard_up_up": round(_jaccard(up1, up2), 4),
                "jaccard_down_down": round(_jaccard(down1, down2), 4),
                "shared_up_genes_top": same_up[:top_n],
                "shared_down_genes_top": same_down[:top_n],
            },
            "opposite_direction_overlap": {
                "up_down_count": len(opposite_up_down),
                "down_up_count": len(opposite_down_up),
                "shared_up_down_genes_top": opposite_up_down[:top_n],
                "shared_down_up_genes_top": opposite_down_up[:top_n],
            },
        }
        pairwise_results[pair_key] = pair_result

        strongest_same.append((pair_key, len(same_up) + len(same_down)))
        strongest_opposite.append((pair_key, len(opposite_up_down) + len(opposite_down_up)))

    strongest_same = sorted(strongest_same, key=lambda x: x[1], reverse=True)[:top_n]
    strongest_opposite = sorted(strongest_opposite, key=lambda x: x[1], reverse=True)[:top_n]

    results = {
        "n_contrasts": int(len(contrast_sets)),
        "contrast_summary": {
            contrast: {
                "n_up": len(info["up"]),
                "n_down": len(info["down"]),
            }
            for contrast, info in contrast_sets.items()
        },
        "strongest_same_direction_pairs": [{"pair": k, "overlap_count": v} for k, v in strongest_same],
        "strongest_opposite_direction_pairs": [{"pair": k, "overlap_count": v} for k, v in strongest_opposite],
        "pairwise_overlap": pairwise_results,
    }

    output_folder = os.path.join(this_run_folder, 'contrast_overlap')
    os.makedirs(output_folder, exist_ok=True)
    save_json(_round_dict(results), os.path.join(output_folder, 'contrast_overlap_summary.json'))

    return _round_dict(results)


@tool
def rank_candidate_biomarkers(
    top_n: int = 30,
    p_thresh: float = 0.05,
    logfc_thresh: float = 1.0,
    use_combat: bool = True,
    *args, **kwargs
):
    """
    综合差异显著性、效应量、cluster 特异性与跨 contrast 稳定性，对候选 biomarker 排序。
    """
    this_run_folder = resolve_path('this_run_folder_path')
    _, _, tables = _load_differential_tables()
    marker_df, _, _ = _compute_marker_specificity_table(use_combat=use_combat)

    specificity_map = marker_df.groupby("gene")["specificity_score"].max().to_dict()
    best_cluster_map = marker_df.sort_values("specificity_score", ascending=False).drop_duplicates("gene").set_index("gene")["best_cluster"].to_dict()

    aggregated_rows = []
    for contrast, df in tables.items():
        gene_col = "PG.Genes" if "PG.Genes" in df.columns else _get_feature_column(df)
        p_col = "adj.P.Val" if "adj.P.Val" in df.columns else "P.Value"
        if p_col not in df.columns or "logFC" not in df.columns:
            continue
        sub = df[[gene_col, p_col, "logFC"]].copy()
        sub["gene"] = sub[gene_col].apply(_canonical_gene_name)
        sub["contrast"] = contrast
        aggregated_rows.append(sub[["gene", "contrast", "logFC", p_col]].rename(columns={p_col: "p_value"}))

    if not aggregated_rows:
        raise ValueError("No differential tables with usable p-value and logFC columns were found.")

    all_diff_df = pd.concat(aggregated_rows, ignore_index=True)
    all_diff_df = all_diff_df.dropna(subset=["gene", "logFC", "p_value"])
    all_diff_df["is_significant"] = (all_diff_df["p_value"] < p_thresh) & (all_diff_df["logFC"].abs() >= logfc_thresh)

    gene_stats = all_diff_df.groupby("gene").agg(
        max_abs_logFC=("logFC", lambda x: float(np.max(np.abs(x)))),
        min_p_value=("p_value", "min"),
        n_significant_contrasts=("is_significant", "sum"),
        n_total_contrasts=("contrast", "nunique"),
    ).reset_index()

    gene_stats["specificity_score"] = gene_stats["gene"].map(specificity_map).fillna(0.0)
    gene_stats["best_cluster"] = gene_stats["gene"].map(best_cluster_map).fillna("NA")
    gene_stats["neg_log10_p"] = -np.log10(gene_stats["min_p_value"].clip(lower=1e-300))

    def _minmax(series):
        smin = series.min()
        smax = series.max()
        if pd.isna(smin) or pd.isna(smax) or smin == smax:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - smin) / (smax - smin)

    gene_stats["score_logFC"] = _minmax(gene_stats["max_abs_logFC"])
    gene_stats["score_pvalue"] = _minmax(gene_stats["neg_log10_p"])
    gene_stats["score_specificity"] = _minmax(gene_stats["specificity_score"])
    gene_stats["score_recurrence"] = _minmax(gene_stats["n_significant_contrasts"])

    gene_stats["biomarker_score"] = (
        0.35 * gene_stats["score_logFC"] +
        0.35 * gene_stats["score_pvalue"] +
        0.20 * gene_stats["score_specificity"] +
        0.10 * gene_stats["score_recurrence"]
    )

    gene_stats = gene_stats.sort_values(
        ["biomarker_score", "max_abs_logFC", "specificity_score"],
        ascending=[False, False, False]
    ).head(top_n).copy()
    gene_stats["rank"] = range(1, len(gene_stats) + 1)

    result_table = gene_stats[
        ["rank", "gene", "biomarker_score", "best_cluster", "max_abs_logFC", "min_p_value", "specificity_score", "n_significant_contrasts", "n_total_contrasts"]
    ].to_dict(orient="records")

    results = {
        "scoring_rule": {
            "logFC_weight": 0.35,
            "pvalue_weight": 0.35,
            "specificity_weight": 0.20,
            "recurrence_weight": 0.10,
            "p_thresh": p_thresh,
            "logfc_thresh": logfc_thresh,
        },
        "top_candidate_biomarkers": _round_dict(result_table),
    }

    output_folder = os.path.join(this_run_folder, 'candidate_biomarkers')
    os.makedirs(output_folder, exist_ok=True)
    save_json(results, os.path.join(output_folder, 'candidate_biomarkers_summary.json'))
    gene_stats.to_csv(os.path.join(output_folder, 'candidate_biomarkers_table.csv'), index=False, encoding='utf-8-sig')

    return results


@tool
def find_key_proteins(
    contrast_type: str = "ALL",
    top_n: int = 20,
    *args, **kwargs
):
    """
    基于 extract_differential_proteins 的输出，按 safe_contrast 返回上调和下调关键蛋白及其统计结果。
    """
    this_run_folder = resolve_path('this_run_folder_path')
    processed_proteins_folder = os.path.join(this_run_folder, "processed_proteins")

    if not os.path.exists(processed_proteins_folder):
        print("Processed proteins folder not found, extracting differential proteins first.")
        _ = extract_differential_proteins.func()

    contrast_list = [
        f for f in os.listdir(processed_proteins_folder)
        if os.path.isdir(os.path.join(processed_proteins_folder, f))
        and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())
    ]
    if not contrast_list:
        contrast_list = [
            f for f in os.listdir(processed_proteins_folder)
            if os.path.isdir(os.path.join(processed_proteins_folder, f))
        ]

    if not contrast_list:
        _ = extract_differential_proteins.func()
        contrast_list = [
            f for f in os.listdir(processed_proteins_folder)
            if os.path.isdir(os.path.join(processed_proteins_folder, f))
            and (contrast_type.lower() in f.lower() or normalize_contrast(contrast_type) in f.lower())
        ]
        if not contrast_list:
            contrast_list = [
                f for f in os.listdir(processed_proteins_folder)
                if os.path.isdir(os.path.join(processed_proteins_folder, f))
            ]

    if not contrast_list:
        raise FileNotFoundError("No processed contrast folders found after running extract_differential_proteins.")

    top_n = max(1, int(top_n))
    results = {}

    def _select_sort_columns(df_part: pd.DataFrame):
        if "adj.P.Val" in df_part.columns and df_part["adj.P.Val"].notna().any():
            return ["adj.P.Val", "logFC"], [True, False]
        if "P.Value" in df_part.columns and df_part["P.Value"].notna().any():
            return ["P.Value", "logFC"], [True, False]
        return ["logFC"], [False]

    def _normalize_output_table(df_part: pd.DataFrame, safe_contrast: str):
        if df_part.empty:
            return []

        frame = df_part.copy()
        gene_col = "PG.Genes" if "PG.Genes" in frame.columns else _get_feature_column(frame)
        protein_col = "PG.ProteinGroups" if "PG.ProteinGroups" in frame.columns else gene_col

        frame["gene"] = frame[gene_col].apply(_canonical_gene_name)
        frame["protein"] = frame[protein_col].astype(str).str.split(";").str[0].str.strip()
        frame["safe_contrast"] = safe_contrast

        for col in ["logFC", "adj.P.Val", "P.Value", "AveExpr", "t", "B"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        sort_cols, ascending = _select_sort_columns(frame)
        frame = frame.sort_values(by=sort_cols, ascending=ascending).head(top_n).copy()

        output_cols = [
            col for col in [
                "gene",
                "protein",
                # "safe_contrast",
                "logFC",
                "adj.P.Val",
                "P.Value",
                # "AveExpr",
                # "t",
                # "B",
            ]
            if col in frame.columns
        ]

        records = []
        for rank, (_, row) in enumerate(frame[output_cols].iterrows(), start=1):
            # item = {"rank": rank}
            item = {}
            for col in output_cols:
                value = row[col]
                if pd.isna(value):
                    item[col] = None
                elif isinstance(value, (np.floating, float)):
                    item[col] = round(float(value), 4)
                elif isinstance(value, (np.integer, int)):
                    item[col] = int(value)
                else:
                    item[col] = value
            records.append(item)
        return records

    for contrast in contrast_list:
        safe_contrast = contrast.replace(" ", "")
        up_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_up.csv")
        down_path = os.path.join(processed_proteins_folder, safe_contrast, f"{contrast}_down.csv")

        if not os.path.exists(up_path) or not os.path.exists(down_path):
            _ = extract_differential_proteins.func()
            if not os.path.exists(up_path) or not os.path.exists(down_path):
                raise FileNotFoundError(f"Processed protein files not found for contrast {contrast}.")

        up_df = pd.read_csv(up_path)
        down_df = pd.read_csv(down_path)

        results[safe_contrast] = {
            "upregulated_proteins": _normalize_output_table(up_df, safe_contrast),
            "downregulated_proteins": _normalize_output_table(down_df, safe_contrast),
        }

    output_folder = os.path.join(this_run_folder, "key_proteins")
    os.makedirs(output_folder, exist_ok=True)
    output_name = f"{normalize_contrast(str(contrast_type)) if str(contrast_type).upper() != 'ALL' else 'ALL'}_key_proteins.json"
    save_json(_round_dict(results), os.path.join(output_folder, output_name))

    return _round_dict(results)


@tool
def extract_differential_proteins(
    p_thresh=None,
    logfc_thresh=None,
    top_n=10,
    min_direction_hits=10,
    *args, **kwargs
):
    """
    提取差异蛋白，并返回适用于 LLM 的 summary（包含 top N 基因及其 p 值）。

    选择规则：
    1. 默认优先使用 adj.P.Val。
    2. 当多重校正后几乎无命中、或显著少于原始 P.Value 命中数时，
       认为当前 contrast 可能处于低样本量/低功效场景，自动回退到 P.Value。
    3. 上调和下调方向分别判断，尽量让两侧都保留不少于 min_direction_hits 个候选蛋白。
    """
    design_diff = _get_analysis_design().get("differential", {})
    requested_p_thresh = p_thresh
    requested_logfc_thresh = logfc_thresh
    if p_thresh is None:
        p_thresh = design_diff.get("p_thresh", 1.0)
    if logfc_thresh is None:
        logfc_thresh = design_diff.get("logfc_thresh", 0.0)
    elif design_diff.get("logfc_thresh") is not None:
        design_logfc = float(design_diff.get("logfc_thresh", logfc_thresh))
        if float(logfc_thresh) > design_logfc:
            logfc_thresh = design_logfc
    p_thresh = float(p_thresh)
    logfc_thresh = float(logfc_thresh)
    top_n = min(int(top_n), 100)
    min_direction_hits = max(1, int(min_direction_hits))

    this_run_folder = resolve_path('this_run_folder_path')
    parameters_path = resolve_path('parameters_path')
    parameters = {}

    pval_col = 'P.Value'
    adjpval_col = 'adj.P.Val'
    logfc_col = 'logFC'
    contrast_col = 'contrast'
    gene_col = 'PG.Genes'

    proteins_folder = os.path.join(this_run_folder, 'processed_proteins')
    os.makedirs(proteins_folder, exist_ok=True)

    differential_proteins_path_list = [
        fn for fn in os.listdir(proteins_folder)
        if fn.startswith("differential_") and fn.endswith(".csv")
    ]

    if not differential_proteins_path_list:
        print("No differential protein files found. Running limma...")
        _ = run_pairwise_limma.func()
        differential_proteins_path_list = [
            fn for fn in os.listdir(proteins_folder)
            if fn.startswith("differential_") and fn.endswith(".csv")
        ]

    if not differential_proteins_path_list:
        raise FileNotFoundError("Still no differential protein files found.")

    def _build_sig_masks(frame: pd.DataFrame, p_col: str):
        sig_mask = frame[p_col] < p_thresh
        up_mask = sig_mask & (frame[logfc_col] > logfc_thresh)
        down_mask = sig_mask & (frame[logfc_col] < -logfc_thresh)
        effect_mask = sig_mask & (frame[logfc_col].abs() > logfc_thresh)
        return up_mask, down_mask, effect_mask

    def _count_sig_hits(frame: pd.DataFrame, p_col: str):
        up_mask, down_mask, effect_mask = _build_sig_masks(frame, p_col)
        return {
            "n_up": int(up_mask.sum()),
            "n_down": int(down_mask.sum()),
            "n_total": int(effect_mask.sum()),
        }

    def _choose_significance_columns(frame: pd.DataFrame):
        available_p_cols = [
            col for col in (adjpval_col, pval_col)
            if col in frame.columns and frame[col].notna().any()
        ]
        if not available_p_cols:
            raise ValueError(
                f"No usable p-value columns found. Expected at least one of {adjpval_col} or {pval_col}."
            )

        if adjpval_col not in available_p_cols:
            raw_counts = _count_sig_hits(frame, pval_col)
            return {
                "up": pval_col,
                "down": pval_col,
            }, {
                "selected_pvalue_column": pval_col,
                "selected_pvalue_column_by_direction": {"up": pval_col, "down": pval_col},
                "selection_reason": f"{adjpval_col} missing or empty; fallback to {pval_col}.",
                "selection_reason_by_direction": {
                    "up": f"{adjpval_col} missing or empty; fallback to {pval_col}.",
                    "down": f"{adjpval_col} missing or empty; fallback to {pval_col}.",
                },
                "adj_summary": None,
                "raw_summary": raw_counts,
            }

        adj_counts = _count_sig_hits(frame, adjpval_col)
        adj_min = float(frame[adjpval_col].min()) if frame[adjpval_col].notna().any() else float("nan")

        if pval_col not in available_p_cols:
            return {
                "up": adjpval_col,
                "down": adjpval_col,
            }, {
                "selected_pvalue_column": adjpval_col,
                "selected_pvalue_column_by_direction": {"up": adjpval_col, "down": adjpval_col},
                "selection_reason": f"{pval_col} missing or empty; using {adjpval_col}.",
                "selection_reason_by_direction": {
                    "up": f"{pval_col} missing or empty; using {adjpval_col}.",
                    "down": f"{pval_col} missing or empty; using {adjpval_col}.",
                },
                "adj_summary": adj_counts,
                "raw_summary": None,
            }

        raw_counts = _count_sig_hits(frame, pval_col)
        min_adj_hits = max(2, min(top_n, 5))
        raw_hit_floor = max(4, min(top_n, 8))

        def _pick_direction(direction: str):
            count_key = "n_up" if direction == "up" else "n_down"
            adj_n = adj_counts[count_key]
            raw_n = raw_counts[count_key]

            if adj_n >= min_direction_hits:
                return adjpval_col, (
                    f"{adjpval_col} retained {adj_n} {direction} proteins, meeting the target of "
                    f"{min_direction_hits}."
                )

            if raw_n >= min_direction_hits:
                return pval_col, (
                    f"{adjpval_col} retained only {adj_n} {direction} proteins, while {pval_col} retained "
                    f"{raw_n}, meeting the target of {min_direction_hits}."
                )

            if adj_n == 0 and raw_n > 0:
                return pval_col, (
                    f"{adjpval_col} retained 0 {direction} proteins, while {pval_col} retained {raw_n}."
                )

            if adj_n <= min_adj_hits and raw_n >= max(raw_hit_floor, adj_n * 3):
                return pval_col, (
                    f"{adjpval_col} was too conservative for {direction} proteins "
                    f"({adj_n} vs {raw_n} under {pval_col})."
                )

            if np.isfinite(adj_min) and adj_min > p_thresh and raw_n >= raw_hit_floor:
                return pval_col, (
                    f"All {adjpval_col} values stayed above threshold, but {pval_col} retained "
                    f"{raw_n} {direction} proteins."
                )

            if raw_n > adj_n:
                return pval_col, (
                    f"Neither threshold reached the target of {min_direction_hits} {direction} proteins, "
                    f"but {pval_col} retained more candidates ({raw_n} vs {adj_n})."
                )

            return adjpval_col, (
                f"Keeping {adjpval_col} for {direction} proteins ({adj_n} vs {raw_n} under {pval_col})."
            )

        selected_by_direction = {}
        reason_by_direction = {}
        for direction in ("up", "down"):
            selected_col, direction_reason = _pick_direction(direction)
            selected_by_direction[direction] = selected_col
            reason_by_direction[direction] = direction_reason

        unified_col = (
            selected_by_direction["up"]
            if selected_by_direction["up"] == selected_by_direction["down"]
            else "mixed"
        )
        unified_reason = (
            f"Up: {reason_by_direction['up']} Down: {reason_by_direction['down']}"
        )

        return selected_by_direction, {
            "selected_pvalue_column": unified_col,
            "selected_pvalue_column_by_direction": selected_by_direction,
            "selection_reason": unified_reason,
            "selection_reason_by_direction": reason_by_direction,
            "adj_summary": adj_counts,
            "raw_summary": raw_counts,
        }

    def _build_gene_info_summary(df_part: pd.DataFrame, selected_p_col: str):
        if df_part.empty:
            return {
                "average_logFC": 0.0,
                "average_p_value": 1.0,
                "gene_names": "",
            }

        return {
            "average_logFC": round(float(df_part[logfc_col].mean()), 4),
            "average_p_value": round(float(df_part[selected_p_col].mean()), 4),
            "gene_names": ';'.join(df_part[gene_col].astype(str).tolist()),
        }

    llm_summary = {}

    for fn in differential_proteins_path_list:
        df = pd.read_csv(os.path.join(proteins_folder, fn))

        required_cols = [contrast_col, logfc_col, gene_col]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns in {fn}: {missing_required}")

        for col in (pval_col, adjpval_col, logfc_col):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        usable_p_cols = [col for col in (pval_col, adjpval_col) if col in df.columns]
        if not usable_p_cols:
            raise ValueError(f"Missing p-value columns in {fn}: expected {pval_col} or {adjpval_col}.")

        df_filtered = df.dropna(subset=[contrast_col, logfc_col]).copy()
        df_filtered = df_filtered[df_filtered[usable_p_cols].notna().any(axis=1)].copy()

        for contrast, subdf in df_filtered.groupby(contrast_col):
            selected_cols, selection_meta = _choose_significance_columns(subdf)
            up_col = selected_cols["up"]
            down_col = selected_cols["down"]
            up = subdf[(subdf[up_col] < p_thresh) & (subdf[logfc_col] > logfc_thresh)].copy()
            down = subdf[(subdf[down_col] < p_thresh) & (subdf[logfc_col] < -logfc_thresh)].copy()

            safe_contrast = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(contrast)).strip("_") or "contrast"
            contrast_folder = os.path.join(proteins_folder, safe_contrast)
            os.makedirs(_windows_long_path(contrast_folder), exist_ok=True)

            up_path = os.path.join(contrast_folder, f"{safe_contrast}_up.csv")
            down_path = os.path.join(contrast_folder, f"{safe_contrast}_down.csv")

            parameters[safe_contrast] = {
                "up": up_path,
                "down": down_path,
                "p_thresh": p_thresh,
                "logfc_thresh": logfc_thresh,
                "min_direction_hits": min_direction_hits,
                # "selected_pvalue_column": selection_meta["selected_pvalue_column"],
                # "selected_pvalue_column_by_direction": selection_meta["selected_pvalue_column_by_direction"],
                # "selection_reason": selection_meta["selection_reason"],
                # "selection_reason_by_direction": selection_meta["selection_reason_by_direction"],
            }

            _safe_to_csv(up, up_path, index=False)
            _safe_to_csv(down, down_path, index=False)

            up_sorted = up.sort_values(by=up_col, ascending=True).head(top_n)
            down_sorted = down.sort_values(by=down_col, ascending=True).head(top_n)

            llm_summary[safe_contrast] = {
                "p_thresh": p_thresh,
                "logfc_thresh": logfc_thresh,
                "min_direction_hits": min_direction_hits,
                # "selected_pvalue_column": selection_meta["selected_pvalue_column"],
                # "selected_pvalue_column_by_direction": selection_meta["selected_pvalue_column_by_direction"],
                # "selection_reason": selection_meta["selection_reason"],
                # "selection_reason_by_direction": selection_meta["selection_reason_by_direction"],
                "significance_summary": {
                    "adj.P.Val": selection_meta["adj_summary"],
                    "P.Value": selection_meta["raw_summary"],
                },
                "n_up": len(up),
                "n_down": len(down),
                "top_up_genes": _build_gene_info_summary(up_sorted, up_col),
                "top_down_genes": _build_gene_info_summary(down_sorted, down_col),
            }

    save_json(parameters, parameters_path)
    append_evidence(
        this_run_folder,
        source="current_matrix",
        tool="extract_differential_proteins",
        claim="Differential protein candidates were extracted using explicit or analysis_design thresholds.",
        files=[os.path.join(proteins_folder, contrast, f"{contrast}_up.csv") for contrast in llm_summary.keys()],
        metrics={
            "p_thresh": p_thresh,
            "logfc_thresh": logfc_thresh,
            "requested_p_thresh": requested_p_thresh,
            "requested_logfc_thresh": requested_logfc_thresh,
            "n_contrasts": len(llm_summary),
            "contrast_counts": {
                contrast: {"n_up": payload.get("n_up"), "n_down": payload.get("n_down")}
                for contrast, payload in llm_summary.items()
            },
        },
        confidence=_confidence_from_design_and_backend("python_fallback" if not R_AVAILABLE else "r_limma"),
        limitations=[],
    )
    return llm_summary


AVAIABLE_TOOLS = [extract_differential_proteins, search_pubmed_for_information,
         search_uniprot_for_protein_knowledge, search_kegg_for_information, search_DGIdb_for_drug,
         search_local_database, run_gsea_enrichment, visualize, run_enrichment, load_original_differential_proteins,
         run_pairwise_limma, detect_batch_and_doublets, prepare_pseudobulk, run_umap,
         load_ori_proteinquant_for_llm, load_proteinquant_combat_for_llm, build_ppi_network, evaluate_sc_proteomics_csv,
         align_samples_from_paths, report_replication_and_power, search_google_information, combat_calibration, run_wgcna,
         run_protein_trajectory, cluster_distribution, run_protein_leakage_analysis, capture_protein_markers_on_trajectory,
         run_sample_correlation_qc, run_overlap_analysis_between_contrasts, rank_candidate_biomarkers, find_key_proteins,
         prepare_scoring_evidence_pack]

# run_missingness_analysis, run_marker_specificity_analysis, run_overlap_analysis_between_contrasts,

if __name__ == "__main__":

    input_file_path = r'./examples/20260501/Cell_TurnoverDynamics_2025'
    this_run_folder = r"./runs/0501/Agent-gpt-5-mini-test"
    config = {
        "sampleinfo_path": os.path.join(input_file_path, 'SampleInfo.csv'),
        "protein_quant_path": os.path.join(input_file_path, 'ProteinQuant.csv'),
        "ground_truth_path": os.path.join(input_file_path, 'ground_truth.txt'),
        "grading_standard_path": os.path.join(input_file_path, 'grading_standard.txt'),
        "user_input_path": os.path.join(input_file_path, 'user_input.txt'),
        "this_run_folder_path": this_run_folder,
        "record_file_path": os.path.join(this_run_folder, 'record_file.md'),
        "memory_path": os.path.join(this_run_folder, 'memory.md'),
        "parameters_path": os.path.join(this_run_folder, 'parameters.json')
    }

    set_config(config)
    arg = {}

    # results = evaluate_sc_proteomics_csv.func()
    # results = combat_calibration.func(need_calibration=True)
    # results = run_pairwise_limma.func()
    results = extract_differential_proteins.func(logfc_thresh=1.0)
    results = find_key_proteins.func()
    # results = run_umap.func()
    # results = visualize.func()
    # results = cluster_distribution.func()
    # results = run_enrichment.func(enrich_types= ['GO_BP', 'KEGG', 'Reactome'])
    # results = run_wgcna.func()
    # results = run_protein_trajectory.func()
    # results = add_pli_to_protein.func()
    # results = build_ppi_network.func(species="mouse")
    # results = search_uniprot_for_protein_knowledge.func()
    # results = search_kegg_for_information.func(species="mus")
    # results = search_DGIdb_for_drug.func()
    # results = run_gsea_enrichment.func(species="mouse")
    # results = load_original_differential_proteins.func()
    # results = run_protein_leakage_analysis.func()
    # results = capture_protein_markers_on_trajectory.func()
    # results = run_missingness_analysis.func()
    # results = run_sample_correlation_qc.func()
    # results = run_marker_specificity_analysis.func()
    # results = run_overlap_analysis_between_contrasts.func()
    # results = rank_candidate_biomarkers.func()
    # results = search_google_information.func(queries=["D-1553 drug resistance mechanisms", "UCL 1684 drug resistance mechanisms"])
    print(results)
    # print(len(json.dumps(results, ensure_ascii=False)))


    # out = run_gsea_enrichment.func()
    #
    # out = search_pubmed_for_information()
    # out = search_local_database()
    # load_proteinquant_combat_for_llm.func()
    # out = search_DGIdb_for_drug()
    # out = load_original_differential_proteins()
    # out = detect_batch_and_doublets()
    # out = prepare_pseudobulk.func()
    # arg = {"queries":["D-1553 drug resistance mechanisms", "UCL 1684 drug resistance mechanisms"]}
    # out = search_google_information()
