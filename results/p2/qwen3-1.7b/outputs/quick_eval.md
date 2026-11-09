# Phase 2 quick evaluation

Means over each test set. Guard columns: % of outputs flagged. This is a sanity check; the held-out evaluation is Phase 3.

| System | Test set | n | wrong_language % | number_mismatch % | length_ratio % | copy % | chrf | metricx24_qe | comet22 |
|---|---|---|---|---|---|---|---|---|---|
| base | flores_dev.en-de | 997 | 0.301 | 0.903 | 0.201 | 0.100 | 53.750 | -1.677 | 0.806 |
| dpo | flores_dev.en-de | 997 | 0.100 | 1.404 | 0.301 | 0 | 56.368 | -1.231 | 0.818 |
| grpo-step1000 | flores_dev.en-de | 997 | 0.100 | 1.404 | 0.100 | 0 | 54.996 | -1.010 | 0.840 |
| grpo-step1125 | flores_dev.en-de | 997 | 0.100 | 1.705 | 0 | 0 | 53.593 | -0.941 | 0.837 |
| grpo-step125 | flores_dev.en-de | 997 | 0.201 | 1.505 | 0 | 0 | 55.717 | -1.217 | 0.831 |
| grpo-step1250 | flores_dev.en-de | 997 | 0.100 | 1.605 | 0 | 0 | 54.136 | -0.969 | 0.839 |
| grpo-step250 | flores_dev.en-de | 997 | 0.100 | 1.304 | 0 | 0 | 55.993 | -1.146 | 0.833 |
| grpo-step375 | flores_dev.en-de | 997 | 0.201 | 1.404 | 0 | 0 | 55.757 | -1.124 | 0.835 |
| grpo-step500 | flores_dev.en-de | 997 | 0.100 | 1.304 | 0 | 0 | 55.595 | -1.093 | 0.837 |
| grpo-step625 | flores_dev.en-de | 997 | 0.100 | 1.404 | 0.100 | 0 | 54.762 | -0.990 | 0.839 |
| grpo-step750 | flores_dev.en-de | 997 | 0.100 | 1.404 | 0 | 0 | 55.374 | -1.012 | 0.842 |
| grpo-step875 | flores_dev.en-de | 997 | 0.100 | 1.505 | 0.100 | 0 | 54.405 | -0.964 | 0.841 |
| grpo_guarded-step1000 | flores_dev.en-de | 997 | 0.100 | 1.103 | 0 | 0 | 54.116 | -1.007 | 0.838 |
| grpo_guarded-step1125 | flores_dev.en-de | 997 | 0.100 | 1.003 | 0 | 0 | 54.300 | -0.966 | 0.839 |
| grpo_guarded-step125 | flores_dev.en-de | 997 | 0.201 | 1.304 | 0 | 0 | 55.551 | -1.210 | 0.831 |
| grpo_guarded-step1250 | flores_dev.en-de | 997 | 0.100 | 1.304 | 0 | 0 | 53.828 | -0.980 | 0.840 |
| grpo_guarded-step250 | flores_dev.en-de | 997 | 0.100 | 1.505 | 0 | 0 | 55.770 | -1.149 | 0.833 |
| grpo_guarded-step375 | flores_dev.en-de | 997 | 0.100 | 1.505 | 0 | 0 | 55.630 | -1.123 | 0.835 |
| grpo_guarded-step500 | flores_dev.en-de | 997 | 0.100 | 1.404 | 0 | 0 | 55.290 | -1.025 | 0.839 |
| grpo_guarded-step625 | flores_dev.en-de | 997 | 0.100 | 1.204 | 0 | 0 | 54.668 | -1.028 | 0.839 |
| grpo_guarded-step750 | flores_dev.en-de | 997 | 0.100 | 1.304 | 0 | 0 | 54.876 | -0.986 | 0.841 |
| grpo_guarded-step875 | flores_dev.en-de | 997 | 0.100 | 1.204 | 0 | 0 | 54.826 | -0.987 | 0.841 |
| opus-mt | flores_dev.en-de | 997 | 0.100 | 0.903 | 0 | 0 | 62.879 | -1.291 | 0.843 |
| rsft | flores_dev.en-de | 997 | 0.201 | 1.304 | 0 | 0.100 | 54.880 | -1.244 | 0.831 |
| sft | flores_dev.en-de | 997 | 0.100 | 1.505 | 0.100 | 0.100 | 54.547 | -1.315 | 0.826 |
| sft+bo8 | flores_dev.en-de | 997 | 0.100 | 2.006 | 0 | 0.100 | 49.815 | -0.825 | 0.805 |
| base | flores_devtest.de-en | 1012 | 0.296 | 0.296 | 0 | 0.296 | 63.731 | -2.432 | 0.877 |
| dpo | flores_devtest.de-en | 1012 | 0 | 0.988 | 0 | 0 | 63.097 | -2.450 | 0.877 |
| grpo-step625 | flores_devtest.de-en | 1012 | 0 | 1.482 | 0 | 0 | 61.878 | -2.394 | 0.877 |
| grpo_guarded-step625 | flores_devtest.de-en | 1012 | 0 | 1.482 | 0 | 0 | 61.819 | -2.425 | 0.877 |
| rsft | flores_devtest.de-en | 1012 | 0 | 1.581 | 0.099 | 0 | 61.653 | -2.704 | 0.875 |
| sft | flores_devtest.de-en | 1012 | 0 | 1.680 | 0.296 | 0 | 61.652 | -2.770 | 0.874 |
| base | flores_devtest.en-de | 1012 | 0.099 | 0.692 | 0.198 | 0 | 53.278 | -1.610 | 0.805 |
| dpo | flores_devtest.en-de | 1012 | 0 | 1.285 | 0.099 | 0 | 56.444 | -1.315 | 0.818 |
| grpo-step625 | flores_devtest.en-de | 1012 | 0 | 1.087 | 0 | 0 | 54.786 | -1.092 | 0.839 |
| grpo_guarded-step625 | flores_devtest.en-de | 1012 | 0.099 | 0.988 | 0 | 0 | 55.006 | -1.061 | 0.838 |
| opus-mt | flores_devtest.en-de | 1012 | 0.099 | 1.383 | 0 | 0.099 | 63.663 | -1.285 | 0.847 |
| rsft | flores_devtest.en-de | 1012 | 0 | 1.087 | 0.099 | 0 | 54.609 | -1.313 | 0.830 |
| sft | flores_devtest.en-de | 1012 | 0.296 | 1.186 | 0.198 | 0.296 | 54.228 | -1.433 | 0.825 |
| sft+bo8 | flores_devtest.en-de | 1012 | 0 | 1.482 | 0 | 0 | 50.126 | -0.925 | 0.803 |
| base | flores_devtest.en-zh | 1012 | 0.198 | 2.273 | 98.814 | 0.099 | 33.692 | -1.735 | 0.872 |
| dpo | flores_devtest.en-zh | 1012 | 19.565 | 1.680 | 77.866 | 19.170 | 28.630 | -5.070 | 0.821 |
| grpo-step625 | flores_devtest.en-zh | 1012 | 34.091 | 2.075 | 63.834 | 32.016 | 23.190 | -6.688 | 0.790 |
| grpo_guarded-step625 | flores_devtest.en-zh | 1012 | 41.008 | 1.383 | 57.213 | 38.538 | 21.447 | -7.434 | 0.772 |
| rsft | flores_devtest.en-zh | 1012 | 90.020 | 0.791 | 9.585 | 88.735 | 5.863 | -18.623 | 0.639 |
| sft | flores_devtest.en-zh | 1012 | 94.269 | 0.494 | 5.336 | 93.281 | 4.581 | -19.775 | 0.628 |
| base | wmt24.en-de | 998 | 1.403 | 1.303 | 0.401 | 2.104 | 48.078 | -3.347 | 0.734 |
| dpo | wmt24.en-de | 998 | 0.401 | 3.607 | 1.503 | 1.603 | 49.653 | -2.853 | 0.743 |
| grpo-step625 | wmt24.en-de | 998 | 0.601 | 3.607 | 0.802 | 1.603 | 48.240 | -2.546 | 0.769 |
| grpo_guarded-step625 | wmt24.en-de | 998 | 0.501 | 3.106 | 0.701 | 1.804 | 48.616 | -2.485 | 0.772 |
| opus-mt | wmt24.en-de | 998 | 0.401 | 1.403 | 1.102 | 1.904 | 53.718 | -3.502 | 0.733 |
| rsft | wmt24.en-de | 998 | 1.403 | 3.006 | 0.501 | 2.705 | 48.448 | -3.099 | 0.752 |
| sft | wmt24.en-de | 998 | 2.204 | 2.705 | 0.601 | 3.006 | 48.269 | -3.291 | 0.747 |
| sft+bo8 | wmt24.en-de | 998 | 0.701 | 5.611 | 0.401 | 1.703 | 44.206 | -2.621 | 0.714 |
