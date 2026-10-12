# Model Card - Cognitive Distortion Detector

## Model Details

| Field | Value |
|---|---|
| Model type | MentalRoBERTa fine-tuned for multi-label sequence classification |
| Base model | `mental/mental-roberta-base` (Hugging Face) |
| Task | 10-label multi-label cognitive distortion detection |
| Training framework | PyTorch + HuggingFace Transformers |
| Loss | BCE with per-label `pos_weight` (capped at 3) |
| Max sequence length | 256 tokens |
| Epochs | 10 (best epoch 6, chosen on a 203-row validation split) |
| Inference threshold | 0.40 (sigmoid output, global) |
| Deployed at | https://huggingface.co/spaces/mihir-apte/cognitive-distortion-detector |

## Intended Use

**In scope:**
- Educational tool to help users recognise cognitive distortion patterns in their own written thoughts
- Portfolio demonstration of NLP + responsible AI practices
- Research and learning about CBT-adjacent NLP

**Out of scope:**
- Clinical diagnosis or assessment
- Therapeutic intervention or advice
- Crisis support or mental health treatment
- Any decision-making about patient care
- Commercial use

## Training Data

- Source: `danthareja/cognitive-distortion` (Hugging Face), derived from a therapist Q&A platform (Shreevastava & Foltz, 2021)
- Split: 1,821 train / 203 validation (carved from the original 2,024-row training set) / 506 test
- Labels: 10 cognitive distortions from Beck's CBT taxonomy (see below), multi-label
- Label schema reviewed with a licensed psychologist
- Limitation: the data is small, imbalanced and noisy; annotators disagree on many examples. This is the main driver of the limited F1.

## Label Set (10 Cognitive Distortions)

| Label | Description |
|---|---|
| All-or-nothing thinking | Seeing situations in black and white with no middle ground |
| Overgeneralization | Drawing broad conclusions from a single event |
| Mental filter | Focusing on one negative detail while ignoring the whole |
| Should statements | Rigid rules using "should", "must", "ought to" |
| Labeling | Attaching a fixed global label to self or others |
| Personalization | Taking excessive personal responsibility for external events |
| Catastrophising/Magnification | Exaggerating the importance of a problem |
| Emotional Reasoning | Treating feelings as facts about reality |
| Mind Reading | Assuming knowledge of others' thoughts without evidence |
| Fortune-telling | Predicting negative outcomes as certain fact |

## Performance (506-row test set, threshold = 0.40)

| Metric | Value |
|---|---|
| Macro F1 | 0.268 |
| Micro F1 | 0.293 |
| Macro precision | 0.239 |
| Macro recall | 0.323 |
| Hamming loss | 0.136 |

Per-label results:

| Label | F1 | Precision | Recall |
|---|---|---|---|
| All-or-nothing thinking | 0.063 | 0.143 | 0.040 |
| Overgeneralization | 0.286 | 0.231 | 0.375 |
| Mental filter | 0.171 | 0.132 | 0.241 |
| Should statements | 0.279 | 0.300 | 0.261 |
| Labeling | 0.416 | 0.362 | 0.488 |
| Personalization | 0.234 | 0.183 | 0.325 |
| Catastrophising/Magnification | 0.356 | 0.304 | 0.429 |
| Emotional Reasoning | 0.184 | 0.151 | 0.235 |
| Mind Reading | 0.414 | 0.358 | 0.492 |
| Fortune-telling | 0.275 | 0.227 | 0.349 |

**Calibration:** not yet measured for this model. The earlier DistilBERT baseline had an overall ECE of 0.077 (`results/calibration_report.json`).

## Comparisons (same test set unless noted)

| Model | Threshold | Macro F1 | Notes |
|---|---|---|---|
| DistilBERT fine-tuned | 0.40 | 0.172 | Baseline, previously deployed |
| DistilBERT fine-tuned | 0.20 | 0.229 | Its original evaluation threshold; recall 0.49, precision 0.15 |
| **MentalRoBERTa fine-tuned** | **0.40** | **0.268** | **Deployed** |
| LLM zero-shot (qwen3.8-27b) | - | 0.211 | 50-row subset only, not directly comparable |
| LLM few-shot (qwen3.8-27b) | - | 0.239 | 50-row subset only, not directly comparable |

## Limitations and Failure Modes

1. **Modest absolute performance.** Macro F1 is 0.27. Treat results as suggestions, not diagnoses.

2. **Very weak on All-or-nothing thinking** (F1 0.06, recall 0.04) and weak on Mental filter and Emotional Reasoning.

3. **Label confusion.** Overgeneralization, All-or-nothing thinking and Catastrophising share lexical signals and are frequently confused.

4. **Small, noisy, single-source data.** About 1,800 training rows from one Q&A platform; labels are subjective.

5. **Single training run, small test set.** Differences of a few points between runs or models are within noise. The test set has only 23 to 59 positives per label.

6. **English only.** Non-English input will produce unpredictable output.

7. **Input length.** Capped at 200 words in the app; the model sees at most 256 tokens.

8. **No demographic evaluation.** Performance across demographic groups, cultural contexts or communication styles has not been assessed.

9. **Threshold sensitivity.** A single global threshold of 0.40 is used. Per-label thresholds could help but were not used for the reported numbers, because they were tuned on the test set.

## Safety and Responsible AI

**Crisis routing:** All inputs are screened for crisis signals (suicidal ideation, self-harm, passive ideation) using regex matching before any model inference. Crisis inputs return a fixed, pre-written safe message with helpline numbers. The classifier and LLM explanation layer are bypassed entirely.

**Misuse detection:** Jailbreak attempts and requests for therapy or medical advice are detected and refused.

**Disclaimer:** Displayed permanently in the UI. The tool is for educational purposes only and is not medical or psychological advice. Predictions are not psychiatric diagnoses.

**Adversarial testing:** 20 adversarial inputs across four categories (direct crisis, indirect crisis, jailbreak, edge cases) were tested; all 20 route correctly.

**LLM explanations:** When enabled, explanations are grounded in LIME-identified keywords and checked by a citation verifier that flags any word claimed as influential but absent from the LIME top-word list.

## Citation and Data Source

Base model: Ji, S. et al. MentalBERT / MentalRoBERTa: Pretrained Language Models for Mental Healthcare. https://huggingface.co/mental/mental-roberta-base

Dataset: danthareja/cognitive-distortion (Hugging Face Datasets); original source Shreevastava & Foltz (2021).

CBT taxonomy: Beck, A.T. (1979). Cognitive Therapy of Depression. Guilford Press.

## Version History

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | Oct 2026 | DistilBERT, threshold 0.40, LIME, safety guardrails. |
| 2.0.0 | Oct 2026 | MentalRoBERTa, max length 256, weighted BCE, 200-word input, LLM explanation layer. |
