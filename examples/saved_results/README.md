# saved_results: reading reported results without running the agent

This is the third workflow in the README table: nothing is computed, a frozen table is read.

```bash
python examples/saved_results/read_saved_table.py examples/saved_results/SYNTHETIC_rule_scores_sample.tsv --group-by method
python examples/saved_results/read_saved_table.py examples/saved_results/SYNTHETIC_rule_scores_sample.tsv --filter method=SystemA --value rule_total
```

`SYNTHETIC_rule_scores_sample.tsv` is a hand-written sample with the column layout of the frozen
rule-score table and invented numbers. It exists so the reader can be exercised offline; it is
**not** a result and must not be cited.

The real tables live in the data archive, and the same reader works on them:

```bash
# the 88-cell frozen rule table
python examples/saved_results/read_saved_table.py ../data/intervals/benchmark_rule_scores.tsv --filter mode=with_run_dir --group-by method --value rule_total
# the two GO branches
python examples/saved_results/read_saved_table.py ../data/go_background/tables/go_results_all.tsv --group-by branch --value q
# a case source table
python examples/saved_results/read_saved_table.py ../data/case_tables/liver_zonation_unit_contrasts.tsv --group-by contrast
```

For multi-gigabyte tables keep `--limit` at a value that is large enough for the summary you
need, or filter first; the reader streams rows but a full pass over a 20 MB table takes a few
seconds.
