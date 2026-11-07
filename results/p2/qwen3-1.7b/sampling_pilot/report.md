# Sampling pilot

Model `Qwen/Qwen3-1.7B` (system `sft`), 200 held-out sentences (`data/p2/sft_val.jsonl`), 8 samples each, scored with `metricx24_qe`. Broken = MetricX error > 10. Spread = mean within-group std of the reward.

| Temperature | top-p | Broken % | Guard flags % | Spread | Identical groups % | Unique / group | Mean | Best-of-N |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.95 | 0.2 | 0.9 | 0.97 | 0.0 | 7.8 | -1.64 | -0.66 |
| 0.6 | 1.0 | 0.4 | 1.4 | 1.20 | 0.0 | 7.9 | -1.89 | -0.68 |
| 0.8 | 0.95 | 1.4 | 2.1 | 1.48 | 0.0 | 7.9 | -2.25 | -0.78 |
| 0.8 | 1.0 | 3.2 | 2.6 | 2.08 | 0.0 | 8.0 | -2.97 | -0.90 **← chosen** |
| 1.0 | 0.95 | 10.0 | 4.9 | 3.02 | 0.0 | 8.0 | -4.45 | -1.28 |
| 1.0 | 1.0 | 27.8 | 6.7 | 4.40 | 0.0 | 8.0 | -7.06 | -1.92 |

**Chosen: temperature 0.8, top-p 1.0** — largest within-group reward spread (2.08) among the 4 settings with <= 5.0% broken samples.
