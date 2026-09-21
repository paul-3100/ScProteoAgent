


# class Config:
#     def __init__(self, this_run_folder: str,
#                  differential_proteins_path: str,
#                  gene_map_path: str,
#                  sampleinfo_path: str,
#                  protein_quant_path: str,
#                  record_file_path: str):
#         self.this_run_folder = this_run_folder
#         self.differential_proteins_path = differential_proteins_path
#         self.gene_map_path = gene_map_path
#         self.sampleinfo_path = sampleinfo_path
#         self.protein_quant_path = protein_quant_path
#         self.record_file_path = record_file_path

from typing import TypedDict

class Config(TypedDict):
    this_run_folder_path: str
    protein_gene_map_path: str
    sampleinfo_path: str
    protein_quant_path: str
    protein_quant_combat_path: str
    record_file_path: str
    ground_truth_path: str
    grading_standard_path: str
    memory_path: str
    parameters_path: str
    analysis_design: dict
    analysis_design_path: str
    analysis_design_inferred_path: str
    input_folder: str
    project_dir: str
    evaluation_requirements_path: str
    publication_mode: bool
    evaluator_mode: str
    enable_internal_scorer: bool
    evaluator_root: str
    publication_eligible_initial: bool

CONFIG = {}

def set_config(cfg: dict):
    global CONFIG
    CONFIG = cfg

def get_config():
    if not CONFIG:
        raise RuntimeError("Config has not been initialized")
    return CONFIG
