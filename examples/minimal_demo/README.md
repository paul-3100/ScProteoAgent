# minimal_demo: input format demonstration

This directory is a **software demonstration**, not study data. The matrix was written by hand
to show the input contract of the agent:

```
ProteinQuant.csv   proteins in rows; leading annotation columns; one column per sample
SampleInfo.csv     one row per sample; first column is the sample key (here FileName);
                   grouping column Cluster; experimental-unit column Donor; Batch column
user_input.txt     the natural-language request, including the evidence-boundary requirement
```

Values are arbitrary numbers, several cells are empty on purpose to show that missing values
are represented as empty fields, and the file header of `user_input.txt` states that the data
are synthetic. Do not cite anything derived from this folder.

Validate it offline (no key, no network, no model call):

```bash
python scripts/validate_dataset_inputs.py examples/minimal_demo
```

Expected result: `PASS` for the dataset, `12 protein rows x 11 columns (8 sample columns
matched)`, `missing: 5 of 96 sample cells`, grouping field `Cluster`, and the experimental-unit
candidates `Donor(2), Batch(2)`. The validator exits 0.

Running the agent on this directory requires your own model credentials, sends the request to
the endpoint you configure, and may be charged by that provider:

```bash
python main_agent.py -i examples/minimal_demo -rf runs
```

That live run was **not** executed while assembling this release, so no output is shipped for
it. The point of this folder is that the format can be checked without any model access.
