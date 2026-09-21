def _function_schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


_STRING_OR_STRING_ARRAY = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
    ]
}


TOOLS_PROMPTS = [
    _function_schema(
        "prepare_scoring_evidence_pack",
        "Generate dataset-specific scoring evidence tables, including evaluation requirements, group composition/QC, candidate protein evidence, curated module scores, and scoring-standard coverage.",
    ),
    _function_schema(
        "extract_differential_proteins",
        "Extract significantly changed proteins for each contrast using explicit thresholds or analysis_design defaults, and return an LLM-friendly summary of upregulated and downregulated proteins.",
        {
            "p_thresh": {
                "type": "number",
                "description": "Adjusted p-value threshold for significance filtering. Omit to use analysis_design.",
            },
            "logfc_thresh": {
                "type": "number",
                "description": "Absolute logFC threshold for differential protein filtering. Omit to use analysis_design.",
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of top proteins to summarize for each direction.",
                "default": 10,
            },
        },
    ),
    _function_schema(
        "search_pubmed_for_information",
        "Search PubMed for evidence related to genes or keywords and return short relevant abstract sentences.",
        {
            "genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbol or list of gene symbols to search.",
                "default": [],
            },
            "keywords": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Optional extra search keywords as a string or list of strings.",
                "default": [],
            },
            "max_hits_per_gene": {
                "type": "integer",
                "description": "Maximum number of PubMed hits to fetch for each query target.",
                "default": 5,
            },
            "max_sentences": {
                "type": "integer",
                "description": "Maximum number of relevant sentences to keep from each abstract.",
                "default": 5,
            },
        },
    ),
    _function_schema(
        "search_uniprot_for_protein_knowledge",
        "Query UniProt for protein function or subcellular location annotations for differential proteins.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude from annotation lookup.",
                "default": [],
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of proteins to query per contrast.",
                "default": 500,
            },
            "category": {
                "type": "string",
                "description": "Annotation category to retrieve. Accepted forms are case-insensitive and support synonyms such as function, functional, location, subcellular_location, subcellular localization, or all/both.",
                "default": "function",
            },
        },
    ),
    _function_schema(
        "search_kegg_for_information",
        "Query local KEGG pathway annotations for differential genes or proteins; online KEGG is used only when explicitly enabled.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude before KEGG lookup.",
                "default": [],
            },
            "species": {
                "type": "string",
                "description": "KEGG organism code, for example hsa for human or mus for mouse.",
                "default": "hsa",
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of genes to send for each contrast.",
                "default": 100,
            },
        },
    ),
    _function_schema(
        "search_DGIdb_for_drug",
        "Query DGIdb to find known drug-gene interactions for differential proteins.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude before drug lookup.",
                "default": [],
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of genes to query per contrast.",
                "default": 100,
            },
        },
    ),
    _function_schema(
        "search_local_database",
        "Search the local annotation database and return structured protein features for differential proteins.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude before local lookup.",
                "default": [],
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of records to return for each gene or protein.",
                "default": 100,
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Feature columns to return from the local database.",
                "default": ["Protein Function", "Disorder"],
            },
        },
    ),
    _function_schema(
        "run_gsea_enrichment",
        "Run GSEA on ranked differential protein results and summarize significant pathways.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "species": {
                "type": "string",
                "description": "Species name such as human or mouse.",
                "default": "human",
            },
            "min_size": {
                "type": "integer",
                "description": "Minimum gene set size allowed in GSEA.",
                "default": 15,
            },
            "max_size": {
                "type": "integer",
                "description": "Maximum gene set size allowed in GSEA.",
                "default": 2000,
            },
            "fdr_threshold": {
                "type": "number",
                "description": "FDR cutoff used to keep enriched pathways.",
                "default": 0.25,
            },
            "permutation_num": {
                "type": "integer",
                "description": "Number of permutations used in GSEA.",
                "default": 500,
            },
            "n_jobs": {
                "type": "integer",
                "description": "Parallel worker count. Use -1 to use all CPUs.",
                "default": -1,
            },
        },
    ),
    _function_schema(
        "visualize",
        "Generate visual summaries for the current proteomics analysis, including differential and enrichment plots.",
        {
            "plot_set": {
                "type": "string",
                "description": "Static figure set to generate. Use full for all standard and mechanism evidence figures.",
                "default": "full",
            },
            "formats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Figure output formats. PNG is always generated.",
                "default": ["png"],
            },
            "top_n_proteins": {
                "type": "integer",
                "description": "Maximum number of top proteins used in mechanism evidence heatmaps.",
                "default": 50,
            },
            "top_n_terms": {
                "type": "integer",
                "description": "Maximum number of enrichment terms used per contrast/direction.",
                "default": 15,
            },
            "max_labels": {
                "type": "integer",
                "description": "Maximum number of protein labels drawn on crowded plots.",
                "default": 12,
            },
            "include_enrichment": {
                "type": "boolean",
                "description": "Whether to draw enrichment-derived figures when enrichment CSV files are available.",
                "default": True,
            },
        },
    ),
    _function_schema(
        "run_enrichment",
        "Run local-first over-representation enrichment analysis such as GO, KEGG, and Reactome for differential proteins.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude before enrichment analysis.",
                "default": [],
            },
            "enrich_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Enrichment namespaces to run, such as GO, KEGG, or Reactome.",
                "default": ["GO", "KEGG", "Reactome"],
            },
            "species": {
                "type": "string",
                "description": "Species name such as human or mouse.",
                "default": "human",
            },
            "fdr_cutoff": {
                "type": "number",
                "description": "FDR cutoff for selecting enriched terms.",
                "default": 0.05,
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of top enriched results to keep per contrast and direction.",
                "default": 10,
            },
        },
    ),
    _function_schema(
        "load_original_differential_proteins",
        "Load original differential protein tables and summarize top hits for each contrast.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of top proteins to keep for each contrast.",
                "default": 20,
            },
            "fdr_cutoff": {
                "type": "number",
                "description": "Adjusted p-value threshold used to define significant hits.",
                "default": 0.05,
            },
        },
    ),
    _function_schema(
        "run_pairwise_limma",
        "Run design-driven differential analysis on the processed protein matrix. Use R/limma when available; otherwise use transparent Python fallback. Contrasts come from analysis_design when provided.",
    ),
    _function_schema(
        "detect_batch_and_doublets",
        "Assess QC patterns, detect possible batch effects, and identify suspicious outlier or doublet-like samples.",
    ),
    _function_schema(
        "prepare_pseudobulk",
        "Build pseudobulk protein expression profiles from grouped samples and summarize their structure.",
        {
            "top_var_proteins": {
                "type": "integer",
                "description": "Number of highly variable proteins to summarize in the output.",
                "default": 20,
            },
        },
    ),
    _function_schema(
        "run_umap",
        "Run UMAP on the batch-corrected protein matrix and summarize cluster structure and marker proteins.",
        {
            "random_state": {
                "type": "integer",
                "description": "Random seed used for UMAP reproducibility.",
                "default": 42,
            },
            "logfc_cut": {
                "type": "number",
                "description": "logFC threshold used when selecting cluster markers.",
                "default": 1.0,
            },
            "pval_cut": {
                "type": "number",
                "description": "P-value threshold used when selecting cluster markers.",
                "default": 0.05,
            },
            "top_n_markers": {
                "type": "integer",
                "description": "Maximum number of marker proteins to keep for each cluster.",
                "default": 50,
            },
        },
    ),
    _function_schema(
        "load_ori_proteinquant_for_llm",
        "Load the original ProteinQuant matrix and return an LLM-friendly dataset summary.",
    ),
    _function_schema(
        "load_proteinquant_combat_for_llm",
        "Load the ComBat-corrected ProteinQuant matrix and return an LLM-friendly dataset summary.",
    ),
    _function_schema(
        "build_ppi_network",
        "Build a protein-protein interaction network for differential proteins and identify hub proteins.",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "exclude_genes": {
                **_STRING_OR_STRING_ARRAY,
                "description": "Gene symbols or prefixes to exclude before network construction.",
                "default": [],
            },
            "species": {
                "type": "string",
                "description": "Species name such as human or mouse.",
                "default": "human",
            },
            "score_threshold": {
                "type": "integer",
                "description": "Minimum STRING confidence score used to keep edges.",
                "default": 700,
            },
        },
    ),
    _function_schema(
        "evaluate_sc_proteomics_csv",
        "Evaluate the quality and statistical structure of existing single-cell proteomics differential analysis results.",
    ),
    _function_schema(
        "align_samples_from_paths",
        "Align sample identifiers across ProteinQuant, ComBat output, and SampleInfo files and summarize the mapping.",
    ),
    _function_schema(
        "report_replication_and_power",
        "Summarize observable sample replication structure and discuss statistical power limitations.",
    ),
    _function_schema(
        "search_google_information",
        "Search Google for background knowledge related to one or more queries and return concise summaries.",
        {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search queries to run independently.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to keep for each query.",
                "default": 5,
            },
        },
        ["queries"],
    ),
    _function_schema(
        "combat_calibration",
        "Standardize the protein quantification matrix and optionally run ComBat batch correction. Respect analysis_design matrix_state and never repeat batch correction for already processed/batch-corrected inputs unless explicitly requested.",
        {
            "reproducibility_cutoff": {
                "type": "number",
                "description": "Minimum within-group reproducibility required to keep a protein.",
                "default": 0.1,
            },
            "need_calibration": {
                "type": "boolean",
                "description": "Whether ComBat batch correction should be applied.",
                "default": True,
            },
        },
    ),
    _function_schema(
        "run_wgcna",
        "Run WGCNA to identify co-expression modules in the proteomics dataset.",
    ),
    _function_schema(
        "run_protein_trajectory",
        "Infer a protein-expression trajectory and summarize state transitions across clusters or groups.",
    ),
    _function_schema(
        "cluster_distribution",
        "Summarize how samples are distributed across clusters and group labels.",
    ),
    _function_schema(
        "run_protein_leakage_analysis",
        "Measure cross-cell-type consistency of protein fold-change patterns to detect possible protein leakage.",
        {
            "eps": {
                "type": "number",
                "description": "Small positive constant used to avoid division by zero.",
                "default": 1e-8,
            },
            "corr_threshold": {
                "type": "number",
                "description": "Correlation threshold used to flag highly similar leakage-like patterns.",
                "default": 0.6,
            },
        },
    ),
    _function_schema(
        "capture_protein_markers_on_trajectory",
        "Identify dynamic marker proteins along the inferred trajectory, including transient and late-up patterns.",
        {
            "top_n": {
                "type": "integer",
                "description": "Maximum number of proteins to return for each dynamic marker category.",
                "default": 20,
            },
            "n_permutations": {
                "type": "integer",
                "description": "Number of permutations used for significance estimation.",
                "default": 500,
            },
            "log_transform": {
                "type": "boolean",
                "description": "Whether to apply log2(x+1) transformation before analysis.",
                "default": False,
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed used by stochastic steps in the analysis.",
                "default": 42,
            },
        },
    ),
    _function_schema(
        "run_sample_correlation_qc",
        "Assess sample-to-sample correlation structure and detect low-correlation outlier samples.",
        {
            "method": {
                "type": "string",
                "description": "Correlation method. Supported values are pearson and spearman.",
                "default": "spearman",
            },
            "top_n_outliers": {
                "type": "integer",
                "description": "Number of lowest-correlation samples to highlight.",
                "default": 10,
            },
            "use_combat": {
                "type": "boolean",
                "description": "Whether to use the ComBat-corrected protein matrix.",
                "default": True,
            },
        },
    ),
    # _function_schema(
    #     "run_marker_specificity_analysis",
    #     "Quantify protein specificity across clusters and return the most cluster-specific markers.",
    #     {
    #         "top_n": {
    #             "type": "integer",
    #             "description": "Number of top markers to return for each cluster.",
    #             "default": 20,
    #         },
    #         "use_combat": {
    #             "type": "boolean",
    #             "description": "Whether to use the ComBat-corrected protein matrix.",
    #             "default": True,
    #         },
    #     },
    # ),
    _function_schema(
        "run_overlap_analysis_between_contrasts",
        "Measure same-direction and opposite-direction overlap of differential proteins across contrasts.",
        {
            "p_thresh": {
                "type": "number",
                "description": "P-value threshold used to define significant proteins.",
                "default": 0.05,
            },
            "logfc_thresh": {
                "type": "number",
                "description": "Absolute logFC threshold used to define significant proteins.",
                "default": 0.0,
            },
            "top_n": {
                "type": "integer",
                "description": "Maximum number of overlapping genes to return per overlap list.",
                "default": 20,
            },
        },
    ),
    _function_schema(
        "rank_candidate_biomarkers",
        "Rank candidate biomarkers by combining differential effect size, significance, cluster specificity, and recurrence across contrasts.",
        {
            "top_n": {
                "type": "integer",
                "description": "Number of top candidate biomarkers to return.",
                "default": 30,
            },
            "p_thresh": {
                "type": "number",
                "description": "P-value threshold used when counting significant contrasts.",
                "default": 0.05,
            },
            "logfc_thresh": {
                "type": "number",
                "description": "Absolute logFC threshold used when counting significant contrasts.",
                "default": 1.0,
            },
            "use_combat": {
                "type": "boolean",
                "description": "Whether to use the ComBat-corrected protein matrix when computing specificity.",
                "default": True,
            },
        },
    ),
    _function_schema(
        "find_key_proteins",
        "Find the key proteins of different contrasts, including upregulated proteins and downregulated proteins, the logFC, the P value and the adj.P.Value, etc",
        {
            "contrast_type": {
                "type": "string",
                "description": "Contrast name to analyze. Use ALL to cover every available contrast.",
                "default": "ALL",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of key proteins.",
                "default": 20,
            },
        },
    ),
]
