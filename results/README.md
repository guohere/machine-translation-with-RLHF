# Results

Finished results, one folder per phase. Each folder holds what the run produced, so every number in the README can be traced back.

- `p2/`: Phase 2, pushed by `notebooks/p2_training.ipynb`: `outputs/<system>/<testset>.jsonl` (translations), `quick_eval.md`, `training/<run>/` (log history and GRPO samples), `candidates.jsonl.gz` (the 40k scored samples). Adapters live in a private Hugging Face repo, not here.
- `p3/`: Phase 3 analysis, rebuilt from `p2/` at the end of every Phase 2 run (`scripts/p3_analyze.py`): `report.md` (sections A–G), `results.json` (every number in it), figures, and `<model>_suspicious_wins_<system>.jsonl`. `human_eval/<model>/` holds the blinded annotation sheets, their keys and the analysed `summary.md`.
- `p1/`: Phase 1 reward benchmark. Unzip `p1_results.zip` from the Kaggle/Colab run here. It contains `report/` (`summary.md`, CSVs, heatmap) and `scores/` (raw scores and timings).

The Phase 1 run (Kaggle, Tesla T4, November 2026) was pushed by the notebook itself; see `p1/report/summary.md`. One display fix was applied by hand afterwards: in the last line of the summary (and in `choice.notes` in `results.json`), the margin of MetricX-24-QE over CometKiwi-22 was printed with the wrong sign. The numbers in the tables were correct; the code is fixed for future runs.
