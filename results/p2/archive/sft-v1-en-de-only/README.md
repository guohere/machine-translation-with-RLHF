# Archived: SFT v1 (English→German only, one fixed instruction)

The first Qwen3-1.7B Part A, trained with SFT on 20,000 English→German pairs, all with the
identical instruction. Kept because it is a finding, not because it is used further.

- English→German: every trained system beat the base model (COMET-22 +0.015 to +0.023 on FLORES
  devtest), see `p3/report.md` and `qwen3-1.7b/outputs/quick_eval.md`.
- Other directions collapsed: German→English answers were German (chrF 64.0 → 25.1), French→English
  answers were French, German→Chinese answers were German, while English→Chinese still worked.
- `qwen3-1.7b/diagnostics/direction.md` shows it is learned, not a bug: with the adapter switched off
  the model reproduces the base outputs (98% identical); scaling the adapter from 0 to 1 moves
  German→English output smoothly from 100% to 2% English; putting the instruction after the text
  brings English back to 37%. The model learned "English input → translate, any other input →
  repeat it", and stopped reading the fixed instruction.

Fix (SFT v2, `configs/p2/sft.yaml` recipe `both-directions-v2`): 20,000 English→German plus 10,000
German→English pairs, and 30% of prompts in other wordings. Base and OPUS-MT outputs do not depend on
SFT and stay in `results/p2/qwen3-1.7b/outputs/`.
