# Project Proposal

## Trustworthy Rewards for RL-Tuned Machine Translation
### Validating the reward, optimizing it, and finding where it breaks: English→German text and cascaded speech translation

**Author:** Xusheng Guo · **Version:** draft v0.1 · **Date:** November 2026 · **Status:** proposal

---

## 1. Summary

RL-based fine-tuning methods (DPO, GRPO) can improve LLM translation by optimizing an automatic quality estimate instead of imitating a single reference. A policy can only be as good as its reward, though. **Whatever the reward cannot see, the policy is free to exploit.**

This project studies that loop end to end. It uses one high-resource language pair (English→German), a small open model (Qwen2.5-1.5B with LoRA) and a single 24 GB GPU:

1. **Validate the reward.** Score WMT system outputs that experts have rated (MQM) with candidate quality-estimation (QE) reward models. Measure how well each one agrees with the humans *when comparing translations of the same source sentence*, and probe its blind spots with controlled perturbations.
2. **Optimize it.** Start every method from the same SFT checkpoint and compare best-of-N reranking, rejection-sampling fine-tuning, DPO and GRPO on the same prompts with the same reward budget.
3. **Close the loop.** Judge the final models with a held-out reward from a different model family, reference-based metrics and a small human check. Find the cases where the training reward went up but quality did not, and test whether Phase 1 predicted them.
4. **Extend to speech.** Feed Whisper transcripts into the same MT models (a cascaded speech-translation system) and measure whether RL-tuned models are more or less robust to recognition errors. Report WER, translation quality and latency.

The project delivers a controlled comparison of alignment methods for MT, a reusable blind-spot probe set for en→de rewards, and an analysis of reward hacking grounded in a measured reward-model study.

## 2. Background and motivation

**From imitation to optimization.** Standard NMT and LLM fine-tuning use maximum likelihood on reference translations. A sentence has many valid translations, and references contain errors, so MLE rewards one arbitrary choice and imitates its noise. Learned metrics now track expert judgement far better than BLEU. COMET-style models compare against a reference, and quality-estimation models such as CometKiwi and MetricX in QE mode need no reference at all. That makes them usable as *rewards* for preference optimization and reinforcement learning, as in ReST, CPO and QE-reward RLHF for MT (He et al., 2024).

**The risk is over-optimization.** A learned reward is an imperfect proxy. When a policy is optimized against a proxy, the proxy score keeps rising after true quality has stopped improving or has started to fall (Gao et al., 2023). For MT the known exploits include omissions, off-target language, length effects and copying of the source.

**The gap this project addresses.** Rewards are usually chosen by their *global* correlation with human scores on WMT. Alignment methods use the reward differently, though. Best-of-N, DPO and GRPO only ever compare **translations of the same source sentence**. The property that matters is therefore *within-source ranking ability* plus *blind spots*, and these are rarely measured before training. The chain "measured blind spot → exploited failure" is rarely shown end to end.

**Why speech.** Many deployed speech-translation systems are cascades: speech recognition (ASR) followed by MT. The MT model then receives inputs with recognition errors, missing or wrong punctuation, and disfluencies. How RL-tuned MT models behave on such inputs is less studied. RL could make them more robust, since they are trained on their own sampled outputs. It could also make them less robust, by making them more literal or more tuned to clean input. Cascades are also the baseline that end-to-end systems have to beat, so knowing where they break is useful in its own right.

## 3. Research questions and hypotheses

| | Research question | Hypothesis |
|---|---|---|
| **RQ1** Reward validation | How well do candidate QE rewards agree with expert judgements for en→de *when ranking translations of the same source*, and what are their blind spots? | **H1.** Ranking the rewards by within-source accuracy gives a different order than ranking them by global correlation. Each candidate fails at least one probe category (e.g. omission, number change) at a high rate. |
| **RQ2** Method comparison | With the same start, prompts and reward-call budget, how do best-of-N, RSFT, DPO and GRPO compare on held-out quality, and at what inference cost? | **H2.** GRPO produces the largest gain in *training* reward, but its held-out gain is smaller. This over-optimization gap grows with training steps and with KL divergence from the SFT policy. |
| **RQ3** Reward hacking | Which failure modes does the policy exploit, and did Phase 1 predict them? | **H3.** The error categories that increase most under RL are the ones with the lowest probe detection rates in Phase 1. KL control and rule-based penalties reduce them at a small cost in held-out quality. |
| **RQ4** Speech robustness | In a Whisper→MT cascade, are RL-tuned MT models more or less robust to ASR errors than the SFT model? | **H4** (two-sided). The quality drop from gold to ASR transcripts differs between SFT and RL-tuned models. Training GRPO on ASR transcripts reduces the drop. |

Secondary question: does RL fine-tuning on en→de cause **forgetting** on untrained directions, measured as quality loss or off-target output?

**Success criteria.** The project does not need RL to "win". It succeeds if it produces:

1. a reward benchmark with confidence intervals;
2. a matched-budget comparison of alignment methods;
3. at least one documented exploit traced back to a measured blind spot, or clear evidence that the probes did *not* predict the exploits, which is also informative;
4. robustness curves for the speech cascade.

## 4. Approach

The mechanics of each component are in [`workflow.md`](workflow.md). This section summarizes the design.

### Phase 1: Validate the reward (QE study, scoring only)

- **Human judgements:** WMT22, WMT23 and WMT24 English→German system outputs with expert **MQM** scores, loaded via Google's [`mt-metrics-eval`](https://github.com/google-research/mt-metrics-eval). WMT22 and WMT23 (sentence-level) decide the reward choice. WMT24, the newest en→de set with expert scores, consists mostly of LLM-based systems, closer to what the policy will produce. It is paragraph-level, so it is reported separately as a check and does not influence the choice.
- **Candidate rewards (core):** CometKiwi-22 (≈0.5B, reference-free) and MetricX-24-hybrid-large in QE mode (≈1.2B, reference-free).
  - **Reference-based comparisons:** COMET-22, MetricX-24 with references, chrF and BLEU.
  - **Stretch candidates:** CometKiwi-23-XL, xCOMET-XL in QE mode, and an LLM judge.
- **Statistics:**
  - **within-source pairwise accuracy:** pairs of translations of the *same* source, the quantity RL actually uses;
  - **segment-level Kendall τ and tie-calibrated pairwise accuracy** (*acc_eq*; Deutsch et al., 2023);
  - **system-level pairwise accuracy;**
  - bootstrap confidence intervals and significance tests between rewards.
- **Sanity checks:** our CometKiwi, COMET-22 and chrF scores must reproduce the scores stored with the WMT data for the same public models, and our statistics must match mt-metrics-eval's own implementation.
- **Blind-spot probes:** take about 500 good en→de translations and apply controlled perturbations. The categories are omission, addition, number change, named-entity swap, negation flip, untranslated copy, wrong target language, word-order damage, repetition and formality flip (*Sie*/*du*). Measure how often each reward scores the perturbed version lower (the detection rate). This follows the ideas of ACES and DEMETR.
- **Decision:** the training reward **R_train** is the candidate with the best within-source accuracy that is also cheap enough to run inside GRPO on 24 GB. The held-out reward **R_eval** is the strongest candidate from a *different model family*: if R_train is CometKiwi, R_eval is MetricX-24 with references, and vice versa.

### Phase 2: Optimize the reward (four methods, one budget)

- **Policy:** Qwen2.5-1.5B-Instruct with LoRA; Qwen2.5-0.5B for debugging.
- **SFT:** about 30k filtered en→de pairs from WMT training data (News Commentary, Europarl). Only heuristic filters are used: language ID, length ratio and deduplication. There is *no* QE filtering, so the baseline is not already tuned towards either reward.
- **Prompt pool:** 5,000 English sentences from WMT News Crawl, deduplicated against every test set. All methods use this pool.
- **Methods:**

| Method | How it uses R_train | Where the cost falls |
|---|---|---|
| Best-of-N (N=8) | Sample 8 translations at test time and return the one R_train scores highest | Every request |
| RSFT (rejection-sampling FT) | Fine-tune on the best of 8 samples per prompt | One-off, offline |
| DPO | Train on (best, worst) pairs from the 8 samples, kept only if the score margin is large enough | One-off, offline |
| GRPO | Online: 8 samples per prompt, group-normalized reward, KL to the SFT policy | One-off, online |
| GRPO + guards | Same as GRPO, with penalties for wrong language, extreme length and copying, plus a KL sweep | One-off, online |

- **Matched budget:** the RSFT, DPO and GRPO comparison uses 5k prompts × 8 samples = **40k reward calls**. GRPO continues past this point only to trace the over-optimization curve.
- **Anchors:** OPUS-MT en-de (classic NMT), Qwen2.5-1.5B zero-shot, and Qwen2.5-7B-Instruct zero-shot (a 5× larger model).
- **Stretch:** CPO; MBR decoding; a "cross-reward" 2×2 that trains on each candidate reward and evaluates with the other.

### Phase 3: Close the loop

- **Evaluation:** R_eval (primary), COMET-22, chrF and BLEU, plus R_train reported separately to expose the gap.
- **Over-optimization curves:** during GRPO, log R_train, R_eval and KL on FLORES dev at regular intervals. Do the same for best-of-N with N ∈ {1, 2, 4, 8, 16}.
- **Hacking diagnostics** (rule-based, on every output):
  - off-target language rate;
  - length-ratio distribution;
  - source-copy rate;
  - mismatched numbers;
  - repetition;
  - an omission proxy based on cross-lingual similarity.
- **Suspicious wins:** segments where R_train(RL) ≫ R_train(SFT) but R_eval(RL) < R_eval(SFT). These are sampled and categorized by hand.
- **Predicted vs. observed:** compare the exploit categories found here with the Phase 1 blind-spot profile.
- **Human check:**
  - 60 random FLORES devtest segments, blind A/B between SFT and the best RL model, to get an unbiased win rate;
  - 50 suspicious-win segments, error-categorized, for the hacking taxonomy;
  - a native-speaker check on 30 segments to estimate agreement, since the main annotator is a non-native (B1) German reader.
- **Forgetting:** FLORES devtest de→en and en→zh, before and after alignment. Report chrF, COMET-22 and off-target rate.

### Phase 4: Cascaded speech translation

- **ASR:** Whisper tiny, base, small, medium and large-v3-turbo on English audio. The five sizes give a natural range of WER.
- **Test sets:** FLEURS English test (gold English transcripts, German references via the shared FLORES sentence IDs), MuST-C v1 en→de tst-COMMON (TED talks), and optionally CoVoST 2 en→de. The gold sentence segmentation is used, so no resegmentation is needed.
- **Conditions:** each MT system translates the gold transcript and every Whisper transcript.
- **Metrics:**
  - **WER**, computed with Whisper's English text normalizer;
  - **translation quality**: COMET-22, chrF and R_eval against the gold German reference. QE scores use the *gold* transcript as source;
  - **latency**: per-utterance ASR and MT time at batch size 1, median and p90.
- **Robustness:** the drop from gold to ASR transcripts, and the slope of quality vs. WER, for each MT method.
- **Stretch, noise-aware GRPO:** train on Whisper transcripts of English utterances from an ASR corpus that does not overlap any test set (e.g. LibriSpeech). The ablation asks *which source the reward should see*: the noisy transcript (faithful to the input) or the gold transcript (faithful to the intended meaning, at the risk of rewarding guesses).
- **Out of scope:** end-to-end, simultaneous and speech-to-speech translation (see §11).

## 5. Experimental design

### 5.1 Controlled variables
Every alignment method shares:
- the base model and the SFT checkpoint it starts from;
- the prompt template;
- the 5k-prompt pool and the reward-call budget;
- decoding at evaluation (greedy, except best-of-N and MBR);
- the LoRA configuration.

Only the method changes. Checkpoints are selected on FLORES-200 *dev*. Test sets are scored once per final system.

### 5.2 Systems benchmarked

| ID | System | Role |
|---|---|---|
| B0 | OPUS-MT en-de | Classic NMT anchor |
| B1 | Qwen2.5-1.5B-Instruct, zero-shot | Lower bound |
| B2 | Qwen2.5-7B-Instruct, zero-shot | "Can a small aligned model catch a 5× larger one?" |
| S0 | SFT (LoRA) | Starting point for all methods |
| A1 | S0 + best-of-8 with R_train | Inference-time alignment |
| A2 | RSFT | Simplest training-time alignment |
| A3 | DPO | Offline preference optimization |
| A4 | GRPO | Online RL |
| A5 | GRPO + guards (KL sweep, penalties) | Hacking mitigation |
| *opt.* | CPO · MBR · cross-reward GRPO · noise-aware GRPO | Stretch |

### 5.3 Evaluation data

| Set | Use |
|---|---|
| WMT22–23 en→de + MQM | Phase 1 only (reward choice) |
| FLORES-200 dev | Checkpoint selection, over-optimization curves |
| **FLORES-200 devtest en→de** | **Primary text test set** |
| WMT24 en→de | Secondary text test set (newer, harder). Also scored in Phase 1 as a separate check on LLM-era systems, never used for the reward choice |
| FLORES-200 devtest de→en, en→zh | Forgetting check |
| FLEURS en (test), MuST-C tst-COMMON | Speech test sets |

### 5.4 Metrics
- **Primary held-out metric:** R_eval.
- **Also reported:** COMET-22, chrF and BLEU (sacreBLEU).
- **Shown for transparency:** R_train.
- **Hacking diagnostics:** rates for each failure category.
- **Human check:** A/B win rate and error counts.
- **Speech:** WER and latency.

### 5.5 Statistics
- Paired bootstrap resampling (1,000 resamples) for every system difference, with 95% CIs.
- Permutation tests for differences between rewards in Phase 1.
- Three seeds for DPO and GRPO if the budget allows; otherwise one seed, stated explicitly.

## 6. Data, models and licenses

| Resource | Use | License / terms |
|---|---|---|
| mt-metrics-eval (WMT22–24 en→de) | Human judgements for Phase 1 | Apache-2.0 code; WMT data terms |
| WMT en-de training data (News Commentary, Europarl) and News Crawl | SFT data, prompt pool | Per-corpus terms (research use) |
| FLORES-200 | Text dev/test | CC BY-SA 4.0 |
| FLEURS | Speech test | CC BY 4.0 |
| MuST-C v1 en-de | Speech test | CC BY-NC-ND 4.0 (registration) |
| Qwen2.5-0.5B / 1.5B / 7B-Instruct | Policy and anchor | Apache-2.0 |
| CometKiwi-22 (`Unbabel/wmt22-cometkiwi-da`) | Reward candidate | **Gated, CC BY-NC-SA 4.0: non-commercial use only** |
| MetricX-24 (`google/metricx-24-hybrid-*`) | Reward candidate / evaluator | Apache-2.0 |
| COMET-22 (`Unbabel/wmt22-comet-da`) | Evaluator | Apache-2.0 |
| Whisper | ASR | MIT |
| OPUS-MT en-de | Anchor | CC BY 4.0 |

This is a non-commercial research project. Gated models require accepting their license on Hugging Face before download.

## 7. Compute plan and budget

| Step | Hardware | Rough GPU-hours |
|---|---|---|
| Phase 1: score WMT22–24 en→de with 2–4 rewards; probes | Free T4 (Kaggle / Colab) | 3–8 |
| SFT (1.5B, LoRA, ~30k pairs) | 24 GB (RTX 4090 / L4 / A10) | 1–3 |
| Sample and score 5k × 8 candidates (vLLM) | 24 GB | 1–2 |
| RSFT and DPO | 24 GB | 2–4 |
| GRPO main run, KL sweep and guarded run (~4 runs) | 24 GB | 12–24 |
| Evaluation of all systems, including best-of-N | 24 GB | 2–4 |
| Phase 4: Whisper × 5 sizes × 2 test sets, cascades | T4 or 24 GB | 2–4 |
| **Total** | | **≈ 25–45 GPU-hours** |

At typical marketplace rates for 24 GB cards (RunPod, Vast.ai; usually under $1/hour, but check current prices), this is **tens of dollars**.

Cost discipline:
- Debug everything on free T4s with tiny subsets. T4s lack bf16, so they are good for debugging and Phase 1 scoring but not for final GRPO runs.
- Rent a GPU only for final runs, and checkpoint often.
- Use vLLM generation inside TRL's GRPO trainer.
- Check whether university HPC access is available first.

## 8. Timeline and milestones

| Week | Work | Milestone (something showable) |
|---|---|---|
| 1 | Reward interface; mt-metrics-eval loader; score candidates; reproduce one official number; probe set v1 | **M1:** reward benchmark and blind-spot profile → choose R_train and R_eval |
| 2 | Data preparation; SFT; evaluation harness; anchors; best-of-N curve | **M2:** baseline table and best-of-N curve |
| 3 | Sampling and scoring; RSFT; DPO | **M3:** offline methods in the benchmark ← *minimum viable project* |
| 4 | GRPO main run, KL sweep, guarded run; over-optimization curves | **M4:** full method benchmark and over-optimization plot |
| 5 | Hacking diagnostics; predicted vs. observed; human check; forgetting | **M5:** reward-hacking report |
| 6 | Whisper sweep; cascade evaluation; latency; write-up | **M6:** speech robustness results and final report |

M1–M3 already form a coherent story: the reward study plus offline alignment. Later phases add depth but don't block it. The weeks can stretch around other commitments.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GRPO with vLLM and the reward model doesn't fit in 24 GB | Use the 0.5B policy, group size 4, max 128 tokens, reward model in fp16 and gradient checkpointing; or rent a 40–48 GB card only for these runs |
| RL gives no held-out gain | Still a result. Report it with the over-optimization analysis. Sanity-check the pipeline first: best-of-N must raise R_train, and normally R_eval too |
| Test-set contamination in Qwen pretraining | It affects all systems equally because they share a base. Report two test sets and state it as a limitation |
| Limited German for the human check | Focus on error types checkable with the source and reference (omissions, additions, numbers, entities, wrong language). Add a native-speaker check on 30 segments and report agreement |
| Noisy or tied human scores | Use tie-calibrated accuracy and bootstrap confidence intervals |
| Gated or non-commercial models | Accept licenses early. MetricX (Apache-2.0) is a fallback |
| Limited time | The minimum viable project is defined at M3, and each phase stands alone |

## 10. Deliverables

1. Open-source code and configs that reproduce every reported number.
2. **Reward benchmark** for en→de and a released **blind-spot probe set** (JSONL).
3. LoRA adapters for all trained systems on the Hugging Face Hub.
4. A technical report or blog post covering the benchmarks, over-optimization curves, a gallery of exploits, and the speech-robustness analysis.
5. Human-check annotations and guidelines.

## 11. Scope and future work

**In scope:** one pair (en→de), sentence-level, a small model, cascaded speech translation.

**Out of scope, as future work:**
- low-resource pairs, where QE rewards are themselves less reliable;
- document-level translation;
- learned reward models trained on preference data;
- reward ensembles and uncertainty-aware rewards;
- **end-to-end, simultaneous and speech-to-speech translation**, which need far more compute and data.

One sentence states the trade-off: *I chose one high-resource pair and a cascaded setup to isolate the effect of the training method and the reward; extending to low-resource pairs and end-to-end speech translation is future work.*

## References

- Amrhein, Moghe & Guillou (2022). *ACES: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics.* WMT.
- Conneau et al. (2023). *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.* SLT.
- Deutsch, Foster & Freitag (2023). *Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration.* EMNLP.
- Di Gangi et al. (2019). *MuST-C: a Multilingual Speech Translation Corpus.* NAACL.
- Dong et al. (2023). *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.*
- Fernandes et al. (2022). *Quality-Aware Decoding for Neural Machine Translation.* NAACL.
- Freitag et al. (2021). *Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation.* TACL.
- Gao, Schulman & Hilton (2023). *Scaling Laws for Reward Model Overoptimization.* ICML.
- Guerreiro et al. (2024). *xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection.* TACL.
- Gulcehre et al. (2023). *Reinforced Self-Training (ReST) for Language Modeling.*
- He et al. (2024). *Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model.* NAACL.
- Juraska et al. (2024). *MetricX-24: The Google Submission to the WMT 2024 Metrics Shared Task.* WMT.
- Karpinska et al. (2022). *DEMETR: Diagnosing Evaluation Metrics for Translation.* EMNLP.
- Kocmi & Federmann (2023). *Large Language Models Are State-of-the-Art Evaluators of Translation Quality.* EAMT.
- Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML.
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
- Rei et al. (2022). *COMET-22* and *CometKiwi*: Unbabel-IST submissions to the WMT22 Metrics and QE shared tasks.
- Shao et al. (2024). *DeepSeekMath* (introduces GRPO).
- Sperber & Paulik (2020). *Speech Translation and the End-to-End Promise: Taking Stock of Where We Are.* ACL.
- Tiedemann & Thottingal (2020). *OPUS-MT: Building Open Translation Services for the World.* EAMT.
- Xu et al. (2024). *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation.* ICML.
