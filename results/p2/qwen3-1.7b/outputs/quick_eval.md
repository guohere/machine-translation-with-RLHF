# Phase 2 quick evaluation

Means over each test set. Guard columns: % of outputs flagged. This is a sanity check; the held-out evaluation is Phase 3.

| System | Test set | n | wrong_language % | number_mismatch % | length_ratio % | copy % | chrf | metricx24_qe | comet22 |
|---|---|---|---|---|---|---|---|---|---|
| base | flores_dev.en-de | 997 | 0.301 | 0.903 | 0.201 | 0.100 | 53.750 | -1.677 | 0.806 |
| dpo | flores_dev.en-de | 997 | 0 | 1.505 | 0 | 0 | 56.500 | -1.320 | 0.817 |
| opus-mt | flores_dev.en-de | 997 | 0.100 | 0.903 | 0 | 0 | 62.879 | -1.291 | 0.843 |
| rsft | flores_dev.en-de | 997 | 0.301 | 1.605 | 0 | 0 | 54.978 | -1.215 | 0.832 |
| sft | flores_dev.en-de | 997 | 0.201 | 1.505 | 0.100 | 0 | 54.277 | -1.292 | 0.827 |
| sft+bo8 | flores_dev.en-de | 997 | 0.100 | 2.708 | 0 | 0 | 50.476 | -0.834 | 0.810 |
| base | flores_devtest.de-en | 1012 | 0.296 | 0.296 | 0 | 0.296 | 63.731 | -2.432 | 0.877 |
| dpo | flores_devtest.de-en | 1012 | 100 | 0.988 | 0 | 94.565 | 24.294 | -6.311 | 0.739 |
| rsft | flores_devtest.de-en | 1012 | 99.308 | 0.692 | 0 | 96.739 | 24.558 | -8.332 | 0.750 |
| sft | flores_devtest.de-en | 1012 | 98.913 | 0.692 | 0 | 96.739 | 24.740 | -8.845 | 0.751 |
| base | flores_devtest.en-de | 1012 | 0.099 | 0.692 | 0.198 | 0 | 53.278 | -1.610 | 0.805 |
| dpo | flores_devtest.en-de | 1012 | 0 | 1.285 | 0 | 0 | 56.684 | -1.346 | 0.820 |
| opus-mt | flores_devtest.en-de | 1012 | 0.099 | 1.383 | 0 | 0.099 | 63.663 | -1.285 | 0.847 |
| rsft | flores_devtest.en-de | 1012 | 0.099 | 1.482 | 0.099 | 0 | 54.356 | -1.327 | 0.828 |
| sft | flores_devtest.en-de | 1012 | 0 | 1.383 | 0 | 0 | 54.077 | -1.411 | 0.822 |
| sft+bo8 | flores_devtest.en-de | 1012 | 0 | 1.680 | 0 | 0 | 50.203 | -0.886 | 0.805 |
| base | flores_devtest.en-zh | 1012 | 0.198 | 2.273 | 98.814 | 0.099 | 33.692 | -1.735 | 0.872 |
| dpo | flores_devtest.en-zh | 1012 | 0.692 | 1.383 | 96.146 | 0.099 | 34.047 | -1.799 | 0.866 |
| rsft | flores_devtest.en-zh | 1012 | 3.261 | 1.877 | 93.182 | 1.581 | 32.668 | -2.129 | 0.861 |
| sft | flores_devtest.en-zh | 1012 | 3.755 | 1.877 | 93.577 | 2.174 | 32.266 | -2.260 | 0.858 |
| base | wmt24.en-de | 998 | 1.403 | 1.303 | 0.401 | 2.104 | 48.078 | -3.347 | 0.734 |
| dpo | wmt24.en-de | 998 | 0.301 | 3.307 | 1.102 | 1.403 | 49.351 | -2.899 | 0.740 |
| opus-mt | wmt24.en-de | 998 | 0.401 | 1.403 | 1.102 | 1.904 | 53.718 | -3.502 | 0.733 |
| rsft | wmt24.en-de | 998 | 0.802 | 3.006 | 0.501 | 2.004 | 48.804 | -3.013 | 0.756 |
| sft | wmt24.en-de | 998 | 0.802 | 3.006 | 0.401 | 2.104 | 48.506 | -3.087 | 0.750 |
| sft+bo8 | wmt24.en-de | 998 | 0.401 | 4.208 | 0.902 | 1.202 | 43.701 | -2.593 | 0.711 |
