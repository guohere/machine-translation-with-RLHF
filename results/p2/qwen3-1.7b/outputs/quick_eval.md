# Phase 2 quick evaluation

Means over each test set. Guard columns: % of outputs flagged. This is a sanity check; the held-out evaluation is Phase 3.

| System | Test set | n | wrong_language % | number_mismatch % | length_ratio % | copy % | chrf | metricx24_qe | comet22 |
|---|---|---|---|---|---|---|---|---|---|
| base | flores_dev.en-de | 997 | 0.301 | 0.903 | 0.201 | 0.100 | 53.750 | -1.677 | 0.806 |
| dpo | flores_dev.en-de | 997 | 0.100 | 1.404 | 0.301 | 0 | 56.368 | -1.231 | 0.818 |
| opus-mt | flores_dev.en-de | 997 | 0.100 | 0.903 | 0 | 0 | 62.879 | -1.291 | 0.843 |
| rsft | flores_dev.en-de | 997 | 0.201 | 1.304 | 0 | 0.100 | 54.880 | -1.244 | 0.831 |
| sft | flores_dev.en-de | 997 | 0.100 | 1.505 | 0.100 | 0.100 | 54.547 | -1.315 | 0.826 |
| sft+bo8 | flores_dev.en-de | 997 | 0.100 | 2.006 | 0 | 0.100 | 49.815 | -0.825 | 0.805 |
| base | flores_devtest.de-en | 1012 | 0.296 | 0.296 | 0 | 0.296 | 63.731 | -2.432 | 0.877 |
| dpo | flores_devtest.de-en | 1012 | 0 | 0.988 | 0 | 0 | 63.097 | -2.450 | 0.877 |
| rsft | flores_devtest.de-en | 1012 | 0 | 1.581 | 0.099 | 0 | 61.653 | -2.704 | 0.875 |
| sft | flores_devtest.de-en | 1012 | 0 | 1.680 | 0.296 | 0 | 61.652 | -2.770 | 0.874 |
| base | flores_devtest.en-de | 1012 | 0.099 | 0.692 | 0.198 | 0 | 53.278 | -1.610 | 0.805 |
| dpo | flores_devtest.en-de | 1012 | 0 | 1.285 | 0.099 | 0 | 56.444 | -1.315 | 0.818 |
| opus-mt | flores_devtest.en-de | 1012 | 0.099 | 1.383 | 0 | 0.099 | 63.663 | -1.285 | 0.847 |
| rsft | flores_devtest.en-de | 1012 | 0 | 1.087 | 0.099 | 0 | 54.609 | -1.313 | 0.830 |
| sft | flores_devtest.en-de | 1012 | 0.296 | 1.186 | 0.198 | 0.296 | 54.228 | -1.433 | 0.825 |
| sft+bo8 | flores_devtest.en-de | 1012 | 0 | 1.482 | 0 | 0 | 50.126 | -0.925 | 0.803 |
| base | flores_devtest.en-zh | 1012 | 0.198 | 2.273 | 98.814 | 0.099 | 33.692 | -1.735 | 0.872 |
| dpo | flores_devtest.en-zh | 1012 | 19.565 | 1.680 | 77.866 | 19.170 | 28.630 | -5.070 | 0.821 |
| rsft | flores_devtest.en-zh | 1012 | 90.020 | 0.791 | 9.585 | 88.735 | 5.863 | -18.623 | 0.639 |
| sft | flores_devtest.en-zh | 1012 | 94.269 | 0.494 | 5.336 | 93.281 | 4.581 | -19.775 | 0.628 |
| base | wmt24.en-de | 998 | 1.403 | 1.303 | 0.401 | 2.104 | 48.078 | -3.347 | 0.734 |
| dpo | wmt24.en-de | 998 | 0.401 | 3.607 | 1.503 | 1.603 | 49.653 | -2.853 | 0.743 |
| opus-mt | wmt24.en-de | 998 | 0.401 | 1.403 | 1.102 | 1.904 | 53.718 | -3.502 | 0.733 |
| rsft | wmt24.en-de | 998 | 1.403 | 3.006 | 0.501 | 2.705 | 48.448 | -3.099 | 0.752 |
| sft | wmt24.en-de | 998 | 2.204 | 2.705 | 0.601 | 3.006 | 48.269 | -3.291 | 0.747 |
| sft+bo8 | wmt24.en-de | 998 | 0.701 | 5.611 | 0.401 | 1.703 | 44.206 | -2.621 | 0.714 |
