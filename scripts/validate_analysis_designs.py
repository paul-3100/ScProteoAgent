import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from analysis_design import infer_analysis_design, write_design_file
from path_config import resolve_examples_root


def validate_dataset(folder: Path) -> dict:
    sampleinfo = folder / "SampleInfo.csv"
    protein_quant = folder / "ProteinQuant.csv"
    triplet = {
        "user_input": (folder / "user_input.txt").exists(),
        "ground_truth": (folder / "ground_truth.txt").exists(),
        "grading_standard": (folder / "grading_standard.txt").exists(),
    }
    if not sampleinfo.exists() or not protein_quant.exists():
        return {
            "dataset": folder.name,
            "ok": False,
            "reason": "missing SampleInfo.csv or ProteinQuant.csv",
            "triplet": triplet,
        }
    try:
        design = infer_analysis_design(str(folder), str(sampleinfo), str(protein_quant))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis_design.inferred.yaml"
            write_design_file(design, out)
            writable = out.exists() and out.stat().st_size > 0
        ok = bool(design.get("group_col")) and writable and all(triplet.values())
        return {
            "dataset": folder.name,
            "ok": ok,
            "triplet": triplet,
            "group_col": design.get("group_col"),
            "batch_col": design.get("batch_col"),
            "matrix_state": design.get("matrix", {}).get("matrix_state"),
            "n_contrasts": len(design.get("differential", {}).get("contrasts", [])),
            "warnings": design.get("warnings", []),
        }
    except Exception as exc:
        return {
            "dataset": folder.name,
            "ok": False,
            "reason": str(exc),
            "triplet": triplet,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    args.root = resolve_examples_root(args.root)
    rows = [validate_dataset(folder) for folder in sorted(args.root.iterdir()) if folder.is_dir()]
    result = {
        "root": str(args.root),
        "dataset_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "rows": rows,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    raise SystemExit(0 if result["dataset_count"] > 0 and result["ok_count"] == result["dataset_count"] else 1)


if __name__ == "__main__":
    main()
