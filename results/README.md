# Results

Finished results, one folder per phase. Each folder holds what the run produced, so every number in the README can be traced back.

- `p1/`: Phase 1 reward benchmark. Unzip `p1_results.zip` from the Kaggle/Colab run here. It contains `report/` (`summary.md`, CSVs, heatmap) and `scores/` (raw scores and timings).

The Phase 1 run (Kaggle, Tesla T4, November 2026) was pushed by the notebook itself; see `p1/report/summary.md`. One display fix was applied by hand afterwards: in the last line of the summary (and in `choice.notes` in `results.json`), the margin of MetricX-24-QE over CometKiwi-22 was printed with the wrong sign. The numbers in the tables were correct; the code is fixed for future runs.
