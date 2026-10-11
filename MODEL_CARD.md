# Model Card - Cognitive Distortion Detector (Phase 1)

## Model Details

| Field | Value |
|---|---|
| Model type | DistilBERT fine-tuned for multi-label sequence classification |
| Base model | distilbert-base-uncased (Hugging Face) |
| Task | 10-class multi-label cognitive distortion detection |
| Phase | Phase 1 (production). Phase 2 in development. |
| Training framework | PyTorch + HuggingFace Transformers |
| Inference threshold | 0.40 (sigmoid output; tuned on validation set) |
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

## Training Data

- Source: Combination of publicly available HuggingFace dataset (`danthareja/cognitive-distortion`) and synthetic data generated via GPT-4 and validated manually
- Size: ~2,800 training examples, 506 test examples
- Labels: 10 cognitive distortions from Beck's CBT taxonomy (see below)
- Annotation: Expert validation of label schema via licensed psychologist consultation
- Limitation: Synthetic data may not fully capture the linguistic diversity of real patient language. This is the primary driver of limited F1 performance.

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

## Performance Metrics (Phase 1, threshold = 0.40)

| Metric | Value |
|---|---|
| Macro F1 | 0.23 |
| Micro F1 | 0.24 |
| Hamming Loss | 0.26 |
| Overall ECE | 0.077 |

Per-label F1:

| Label | F1 | Precision | Recall |
|---|---|---|---|
| All-or-nothing thinking | 0.182 | 0.118 | 0.400 |
| Overgeneralization | 0.267 | 0.176 | 0.554 |
| Mental filter | 0.173 | 0.120 | 0.310 |
| Should statements | 0.188 | 0.114 | 0.522 |
| Labeling | 0.304 | 0.221 | 0.488 |
| Personalization | 0.185 | 0.116 | 0.450 |
| Catastrophising/Magnification | 0.279 | 0.179 | 0.633 |
| Emotional Reasoning | 0.241 | 0.159 | 0.500 |
| Mind Reading | 0.264 | 0.170 | 0.593 |
| Fortune-telling | 0.210 | 0.138 | 0.442 |

**Calibration:** ECE = 0.077 (pooled across all labels, 10 bins). Per-label ECE ranges from 0.054 (Mental filter) to 0.099 (Personalization). Lower is better; 0 = perfectly calibrated. This means the model's stated confidence is moderately reliable, despite low F1. A high-confidence prediction (>0.7) is more likely to be correct than a low-confidence one (0.40-0.50), which informs how end users should interpret results.

## Limitations and Failure Modes

1. **Low precision across all labels.** Precision ranges from 0.11 to 0.22. The model over-predicts positive labels, producing frequent false positives. Users should treat results as suggestions, not diagnoses.

2. **Label confusion.** Overgeneralization, All-or-nothing thinking, and Catastrophising are frequently co-predicted and confused with each other. This is partly a data problem (they share lexical signals) and partly a model capacity problem.

3. **Synthetic training data.** A significant portion of training data is LLM-generated. Synthetic text may lack the subtlety, code-switching, and colloquial expression of real patient language.

4. **English only.** The model was trained and validated exclusively on English text. Non-English inputs will produce unpredictable outputs.

5. **Short text optimised.** Input is capped at 150 words. The model was not evaluated on longer texts.

6. **No demographic evaluation.** The model has not been evaluated for performance differences across demographic groups, cultural contexts, or communication styles.

7. **Threshold sensitivity.** At threshold 0.40, recall is prioritised over precision. Raising the threshold reduces false positives but increases false negatives.

## Safety and Responsible AI

**Crisis routing:** All inputs are screened for crisis signals (suicidal ideation, self-harm, passive ideation) using regex pattern matching before any model inference. Crisis inputs return a fixed, pre-written safe message with helpline numbers. The LLM explanation layer is bypassed entirely for crisis inputs.

**Misuse detection:** Jailbreak attempts and requests for therapy or medical advice are detected and refused.

**Disclaimer:** Displayed permanently in the UI. The tool is for educational purposes only and does not constitute medical or psychological advice.

**Adversarial testing:** 20 adversarial inputs across four categories (direct crisis, indirect crisis, jailbreak, edge cases) were tested. All 20 routed correctly as of Phase 1 release.

**LLM explanations:** When enabled, explanations are grounded in LIME-identified keywords and verified by a citation checker that flags any word claimed as influential but absent from the LIME top-word list. Verification failures are logged.

## Comparisons

| Model | Macro F1 | Notes |
|---|---|---|
| TF-IDF + Logistic Regression | - | See results/baseline_results.json |
| DistilBERT fine-tuned (Phase 1) | 0.23 | Deployed |
| Llama 3.3 70B zero-shot | TBD | Run python -m src.llm_baseline |
| Llama 3.3 70B few-shot | TBD | Run python -m src.llm_baseline |
| DeBERTa-v3-base (Phase 2) | TBD | In development |

## Citation and Data Source

Base model: Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.

Dataset: danthareja/cognitive-distortion (HuggingFace Datasets) + proprietary synthetic data.

CBT taxonomy: Beck, A.T. (1979). Cognitive Therapy of Depression. Guilford Press.

## Version History

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | Oct 2026 | Phase 1 release. DistilBERT, threshold 0.40, LIME explainability, safety guardrails. |
