# Semantic Edge Enrichment for Inter-Community Conflict Prediction in Reddit

We use LLM-derived semantic labels to improve and explain inter-community conflict prediction in Reddit cross-community links. Our study revisits the Stanford SNAP Reddit Hyperlink Network conflict-prediction setting and asks whether simple plain-language semantic labels can add useful predictive signal beyond the original handcrafted, social, graph, and text features.

We annotate all 116,224 crosslink entries with two LLM annotators, Gemini 2.5 Flash-Lite and GPT-4o mini. Each annotator sees only the recovered post text, source subreddit, target subreddit, and a non-outcome text-quality flag. We hide the outcome label, train/validation/test split, future behavior, target-derived features, and model predictions from the annotators.

Our main conclusion is that semantic enrichment is useful, but not automatic. Semantic labels improve the simple RF500 metafeature baseline for both annotators. In feature-rich models that already use LSTM, social, graph, and metafeature representations, gains are smaller because many semantic labels are already predictable from the original features. The strongest result comes from the Gemini-based redundancy-aware semantic system.

---

## Repository Contents

This repository is intentionally simple. We keep the notebooks, the paper figures, and this single README. We do not use separate `requirements.txt`, `environment.yml`, `LICENSE`, `CITATION.cff`, `.gitignore`, `paper/`, `data/`, `results/`, or `docs/` files. All usage, data-access, reproducibility, citation, and figure/table information is kept here.

```text
.
├── README.md
├── notebooks/
│   ├── 01_semantic_annotation_gemini25_flashlite_full_dataset.ipynb
│   ├── 02_semantic_annotation_gpt4o_mini_full_dataset.ipynb
│   ├── 03_model_training_evaluation_gemini25_flashlite_semantics.ipynb
│   └── 04_model_training_evaluation_gpt4o_mini_semantics.ipynb
└── figures/
    ├── fig01_reddit_community_space.png
    ├── fig02_semantic_enrichment_pipeline.png
    ├── fig03a_gemini_pr_auc_values_and_deltas.png
    ├── fig03b_gpt4o_mini_pr_auc_values_and_deltas.png
    └── fig04_selected_precision_recall_curves_gemini.png
```

---

## Paper Summary

**Paper title:** Semantic Edge Enrichment for Inter-Community Conflict Prediction in Reddit

**Author:** Ha Viet Thang  
**Affiliation:** FPT University, Can Tho, Vietnam  
**Contact:** havietthang@acm.org

We study whether explicit semantic labels help predict whether a Reddit cross-community link will be followed by conflict or mobilization. We use the original Reddit conflict-prediction setting introduced by Kumar, Hamilton, Leskovec, and Jurafsky, and we preserve the original train/validation/test split.

We add two independent LLM semantic-enrichment layers:

1. Gemini 2.5 Flash-Lite semantic annotations.
2. GPT-4o mini semantic annotations.

For each annotator, we compare original models against semantic-augmented models under matched architectures. We also evaluate redundancy-aware semantic-aware models that use raw semantic labels, residualized semantic features, semantic-original interactions, and validation-selected gradient boosting.

---

## Data Sources

We do not redistribute the original Reddit corpus as our own data. The original Reddit conflict-prediction corpus should be obtained from the SNAP project pages.

| Resource | Use in this project |
|---|---|
| [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) | Directed subreddit-to-subreddit hyperlink network and subreddit embeddings. |
| [SNAP Community Interaction and Conflict on the Web](https://snap.stanford.edu/conflict/) | Original conflict-prediction task, target definition, and project context. |
| Generated semantic annotations and outputs | We release generated semantic labels, benchmark outputs, model predictions, figures, tables, and supporting files keyed to original identifiers. |

In the paper draft, the generated semantic annotations and experimental outputs are listed as available on Kaggle:

[semantic-enrichment-conflict-prediction-in-reddit](https://www.kaggle.com/datasets/michaelhafpt/semantic-enrichment-conflict-prediction-in-reddit)

---

## Notebook Order

Run the notebooks in this order.

| Step | Notebook | Purpose | Suggested Colab runtime |
|---:|---|---|---|
| 1 | `01_semantic_annotation_gemini25_flashlite_full_dataset.ipynb` | Generate or resume Gemini 2.5 Flash-Lite semantic annotations for the full corpus. | Standard Colab CPU is enough because annotation uses an API; High-RAM is safer for loading/merging data. |
| 2 | `02_semantic_annotation_gpt4o_mini_full_dataset.ipynb` | Generate or resume GPT-4o mini semantic annotations for the full corpus. | Standard Colab CPU is enough because annotation uses an API; High-RAM is safer for loading/merging data. |
| 3 | `03_model_training_evaluation_gemini25_flashlite_semantics.ipynb` | Train/evaluate original and Gemini-semantic models; generate Gemini tables and figures. | Colab A100 High-RAM preferred; L4/T4 can work but may be slower. |
| 4 | `04_model_training_evaluation_gpt4o_mini_semantics.ipynb` | Train/evaluate original and GPT-semantic models; generate GPT tables and figures. | Colab A100 High-RAM preferred; L4/T4 can work but may be slower. |

We recommend running the annotation notebooks first, then the model notebooks. The annotation notebooks are written to save outputs incrementally so that interrupted Colab sessions can resume instead of restarting from zero.

---

## Expected Local or Drive Layout

The notebooks were developed around a Google Drive project folder. A typical Colab layout is:

```text
/content/drive/MyDrive/csonet2026_conflict_prediction/
├── tokenized_posts.tsv
├── post_crosslinks_info.tsv
├── glove_word_embeds.txt
├── original_split_ids_extracted.csv
├── recovered_text_original_splits.parquet
├── crosslink_post_target_map.parquet
└── results/
    ├── llm_semantic_enrichment_gemini_2_5_flash_lite_boolean_v5_full_dataset/
    ├── llm_semantic_enrichment_openai_gpt4o_mini_boolean_v5_full_dataset/
    ├── model_training_gemini_semantics/
    └── model_training_gpt4o_mini_semantics/
```

If a notebook uses a different folder name, update the `PROJECT_ROOT` or equivalent path variable near the beginning of the notebook.

---

## Software and API Notes

We do not include `requirements.txt` or `environment.yml` in this minimal repository. The notebooks install or import the packages they need inside the notebook cells.

Typical packages used across the notebooks include:

| Area | Packages |
|---|---|
| Data processing | `pandas`, `numpy`, `pyarrow`, `tqdm` |
| Machine learning | `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna` |
| Neural models | `torch` |
| Interpretation | `shap` |
| Plotting | `matplotlib` |
| LLM APIs | OpenAI API client, Google Gemini API client |

For API notebooks, set keys through Colab secrets or environment variables. Do not hard-code API keys inside notebooks.

Common names:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

or:

```bash
GEMINI_API_KEY=...
```

---

## Semantic Annotation Schema

We use one primary label and seven independent boolean labels. The primary label captures the dominant semantic role. The boolean labels allow multiple mechanisms to be active at the same time.

### Primary Label Options

| Primary label |
|---|
| `neutral_information_reference` |
| `negative_grievance_reference` |
| `audience_directed_engagement_reference` |
| `norm_violation_reference` |
| `cross_community_comparison_reference` |
| `supportive_or_defensive_reference` |
| `unclear_or_other` |

### Boolean Semantic Indicators

| Boolean label | Meaning |
|---|---|
| `neutral_information_reference` | Ordinary informational reference. |
| `negative_grievance` | Complaint, hostile evaluation, accusation, ridicule, or outrage. |
| `audience_directed_engagement` | Appeal to readers to look, judge, respond, vote, report, visit, or participate. |
| `norm_violation_framing` | Claim that rules, expectations, moderation norms, or social norms were violated. |
| `cross_community_comparison` | Explicit comparison between communities, groups, or audiences. |
| `intergroup_boundary_framing` | In-group/out-group framing, sides, camps, insiders, outsiders, or identities. |
| `supportive_or_defensive_framing` | Defense or support of a group, user, moderator, position, action, or discussion. |

### Compact Annotation Prompt

```text
System role:
You are a careful research annotator for a computational social science study.
Label a Reddit cross-community post for semantic mechanisms related to community
mobilization and conflict prediction. Output ONLY valid JSON.

Leakage controls:
Use only recovered title/body text, source subreddit, target subreddit, and the
non-outcome text-quality flag. Do not use or infer outcome label, split,
future behavior, model prediction, target-derived features, confidence,
probability, intensity, severity, or any numeric judgment.

Primary label, choose exactly one:
neutral_information_reference;
negative_grievance_reference;
audience_directed_engagement_reference;
norm_violation_reference;
cross_community_comparison_reference;
supportive_or_defensive_reference;
unclear_or_other.

Independent boolean labels:
negative_grievance;
audience_directed_engagement;
norm_violation_framing;
cross_community_comparison;
intergroup_boundary_framing;
supportive_or_defensive_framing;
neutral_information_reference.

Per-row prompt:
Source community: {source_subreddit}
Target community: {target_subreddit}
Recovered text quality: {text_quality_flag}
Recovered text: {text_for_llm}
Return JSON with primary_label, seven booleans, and short_reason.
```

The `short_reason` field is used for audit and inspection only. We do not use it as a predictive feature.

---

## Models

We evaluate semantic enrichment at two levels.

### Matched Original vs Semantic Models

In matched comparisons, each original model is paired with a semantic version using the same post IDs and the same architecture. The only difference is whether semantic features are included.

| Model family | Why we include it |
|---|---|
| RF500 metafeatures | Tests whether semantic labels help when the base model has only handcrafted original metafeatures. |
| RF500 full features | Tests whether semantic labels still help after adding richer original features. |
| SocialLSTM variants | Tests whether semantic labels help recurrent text/social models. |
| Enhanced boosted trees | Tests semantic augmentation under stronger tabular learners. |

### Redundancy-Aware Semantic-Aware Models

The semantic-aware system combines original full features, raw semantic labels, semantic residuals, semantic-original interactions, and validation-selected gradient boosting.

For each semantic label `s_j`, we train an auxiliary model to predict the semantic label from the original representation `x`:

```text
s_hat_j = f_j(x)
```

The residualized semantic feature is:

```text
r_j = s_j - s_hat_j
```

We use residuals because semantic labels may already be indirectly encoded in the original text, social, and graph representations. Residualized labels are intended to preserve semantic information that is not already predictable from original features.

---

## Evaluation Protocol

We preserve the original train/validation/test split. We use validation PR-AUC for model selection and evaluate the test split only after selection. F1 thresholds are chosen on validation predictions and then applied unchanged to the test split.

| Evaluation detail | Protocol |
|---|---|
| Main selection metric | Validation PR-AUC |
| Final test metrics | ROC-AUC, PR-AUC, F1 |
| F1 threshold | Selected on validation, applied once to test |
| Deterministic model uncertainty | Paired bootstrap over held-out test examples |
| LSTM stability | Multiple random seeds |
| Leakage control | No outcome labels, future behavior, split membership, target-derived features, or model predictions are shown to annotators |

We emphasize PR-AUC because the positive conflict/mobilization class is rare.

---

## Figures

### Figure 1. Reddit Community Space

![Figure 1. Reddit community space](figures/fig01_reddit_community_space.png)

We use this figure to give a descriptive view of the Reddit community embedding space. Coral nodes indicate communities that initiate more negative cross-community links; teal nodes indicate other communities. Topic outlines are visual guides only and are not used as model features.

### Figure 2. Semantic Enrichment Pipeline

![Figure 2. Semantic enrichment pipeline](figures/fig02_semantic_enrichment_pipeline.png)

We use this figure to summarize the leakage-controlled semantic-enrichment and evaluation pipeline. Each LLM sees only text, source community, target community, and a non-outcome text-quality flag.

### Figure 3a. Gemini PR-AUC Values and Deltas

![Figure 3a. Gemini PR-AUC values and semantic-minus-original deltas](figures/fig03a_gemini_pr_auc_values_and_deltas.png)

We use this panel to show Gemini original-versus-semantic PR-AUC values and semantic-minus-original differences.

### Figure 3b. GPT-4o Mini PR-AUC Values and Deltas

![Figure 3b. GPT-4o mini PR-AUC values and semantic-minus-original deltas](figures/fig03b_gpt4o_mini_pr_auc_values_and_deltas.png)

We use this panel to show GPT-4o mini original-versus-semantic PR-AUC values and semantic-minus-original differences.

### Figure 4. Selected Test Precision-Recall Curves

![Figure 4. Selected test precision-recall curves](figures/fig04_selected_precision_recall_curves_gemini.png)

We use this figure to compare representative original and semantic models on the held-out test split. PR-AUC is the main model-selection metric because conflict examples are uncommon.

---

## Tables

### Table 1. Semantic Coverage by Annotator

Split membership follows the original train/validation/test assignment.

| Annotator | Train | Validation | Test | Total paired | Missing |
|---|---:|---:|---:|---:|---:|
| Gemini | 93,659 | 11,261 | 11,257 | 116,177 | 47 |
| GPT-4o mini | 93,696 | 11,264 | 11,264 | 116,224 | 0 |

### Table 2. Main PR-AUC Results by Annotator

RF500 rows measure matched direct semantic augmentation. Semantic-aware rows compare validation-selected original and semantic-aware systems.

| Annotator | Comparison | Original PR-AUC | Semantic PR-AUC | Delta PR-AUC |
|---|---|---:|---:|---:|
| Gemini | RF500 metafeatures | 0.2782 | 0.2846 | +0.0063 |
| GPT-4o mini | RF500 metafeatures | 0.2790 | 0.2874 | +0.0084 |
| Gemini | Semantic-aware model | 0.4408 | 0.4487 | +0.0079 |
| GPT-4o mini | Semantic-aware model | 0.4450 | 0.4429 | -0.0021 |

### Table 3. Selected Original vs Semantic-Aware Systems on Held-Out Test Split

Each pair is selected by validation PR-AUC before test evaluation.

| Annotator | Model | ROC-AUC | PR-AUC | F1 |
|---|---|---:|---:|---:|
| Gemini | Selected original model | 0.7706 | 0.4408 | 0.3391 |
| Gemini | Semantic-aware model | 0.7720 | 0.4487 | 0.4363 |
| GPT-4o mini | Selected original model | 0.7708 | 0.4450 | 0.4278 |
| GPT-4o mini | Semantic-aware model | 0.7723 | 0.4429 | 0.4308 |

### Table 4. Semantic Label Predictability from Original Full Features

High ROC-AUC here indicates that original features already encode much of the same information captured by the semantic label.

| Semantic label | Gemini ROC-AUC | GPT-4o mini ROC-AUC |
|---|---:|---:|
| Intergroup Boundary Framing | 0.894 | 0.909 |
| Norm Violation Framing | 0.900 | 0.897 |
| Negative Grievance | 0.872 | 0.885 |
| Audience Directed Engagement | 0.785 | 0.842 |
| Supportive Or Defensive Framing | 0.839 | 0.825 |
| Cross Community Comparison | 0.790 | 0.814 |
| Neutral Information | 0.814 | 0.810 |

---

## Main Findings

1. We find that semantic labels improve the simple RF500 metafeature baseline for both annotators.
2. Gemini improves RF500 metafeature test PR-AUC from 0.2782 to 0.2846.
3. GPT-4o mini improves RF500 metafeature test PR-AUC from 0.2790 to 0.2874.
4. The strongest semantic-aware result comes from the Gemini-based system, improving ROC-AUC from 0.7706 to 0.7720, PR-AUC from 0.4408 to 0.4487, and F1 from 0.3391 to 0.4363.
5. Semantic gains are smaller in feature-rich models because many semantic labels are already predictable from the original representation.
6. We interpret semantic enrichment as partially redundant signal rather than universally additive signal.

---

## Reproduction Checklist

To reproduce the reported artifacts:

1. Download the original Reddit conflict-prediction files from the SNAP project pages.
2. Place the original files in the expected project folder or update the notebook path variables.
3. Add API keys through Colab secrets or environment variables.
4. Run the Gemini annotation notebook.
5. Run the GPT-4o mini annotation notebook.
6. Run the Gemini training/evaluation notebook.
7. Run the GPT-4o mini training/evaluation notebook.
8. Compare the generated metrics against the tables in this README.
9. Compare generated figures against the files in `figures/`.

We recommend keeping large raw data, generated `.parquet`, `.csv`, `.jsonl`, model files, and intermediate outputs outside Git. GitHub blocks files larger than 100 MiB in normal Git repositories, so large artifacts should stay on Kaggle, Google Drive, or another external artifact store.

---

## What Not to Commit

Even though we do not include a separate `.gitignore`, we still recommend not committing the following files:

| Do not commit | Reason |
|---|---|
| API keys or `.env` files | They are private credentials. |
| Raw SNAP data | The original dataset should be obtained from SNAP. |
| Large generated outputs | They can exceed GitHub file limits and make the repo hard to clone. |
| Google Drive cache files | They are machine-specific. |
| Colab checkpoint folders | They are temporary notebook artifacts. |
| Model binaries or large prediction dumps | Store these on Kaggle, Drive, or release artifacts instead. |

---

## Citation Information

If you use this repository, cite the paper:

```bibtex
@inproceedings{thang2026semanticedge,
  title     = {Semantic Edge Enrichment for Inter-Community Conflict Prediction in Reddit},
  author    = {Thang, Ha Viet},
  booktitle = {Proceedings of CSoNet 2026},
  year      = {2026},
  note      = {Code and generated semantic artifacts released for reproduction}
}
```

Also cite the original Reddit conflict-prediction work:

```bibtex
@inproceedings{kumar2018community,
  title     = {Community Interaction and Conflict on the Web},
  author    = {Kumar, Srijan and Hamilton, William L. and Leskovec, Jure and Jurafsky, Dan},
  booktitle = {Proceedings of The Web Conference 2018},
  pages     = {933--943},
  year      = {2018},
  publisher = {ACM},
  doi       = {10.1145/3178876.3186141}
}
```

---

## Source Links

We use the following project and documentation sources:

| Source | Link |
|---|---|
| GitHub README guidance | https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories |
| GitHub large-file guidance | https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github |
| SNAP Reddit Hyperlink Network | https://snap.stanford.edu/data/soc-RedditHyperlinks.html |
| SNAP Community Interaction and Conflict on the Web | https://snap.stanford.edu/conflict/ |
| OpenAI GPT-4o mini documentation | https://developers.openai.com/api/docs/models/gpt-4o-mini |
| OpenAI Structured Outputs documentation | https://developers.openai.com/api/docs/guides/structured-outputs |
| Google Gemini API documentation | https://ai.google.dev/gemini-api/docs |

---

## Notes on Scope and Responsible Use

We built this repository for research reproduction and analysis. The models are intended for aggregate study of inter-community dynamics and decision support, not automatic punishment of users or communities. The semantic annotations are LLM outputs, not human gold labels or causal explanations. Results may vary with prompt wording, provider behavior, software versions, hardware, random seeds, and future changes to Reddit behavior.
