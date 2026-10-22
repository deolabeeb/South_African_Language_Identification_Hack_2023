# Language identification model card

## Purpose

Educational text-classification and reproducibility demonstration for the eleven-language label set in the ExploreAI 2023 classification hackathon. The intended audience is a reviewer who wants to inspect data preparation, feature extraction, validation, saved-pipeline inference and honest limitations.

The new experiment compares two scikit-learn pipelines under a fixed protocol. The final chosen model, actual parameters, metrics, source/data fingerprints and environment are reported in [the results](../reports/RESULTS.md) and [machine-readable aggregate evidence](../reports/metrics.json). The selected artifact is trained only on the training partition and evaluated on the holdout once. There is no claim that it improves on the historical notebook's differently prepared data or split.

## Inputs and outputs

Input: 1 to 10,000 normalized Unicode characters, treated as one text. Transformation is label-independent NFKC normalization, case folding and whitespace collapse. Output: one benchmark code from `afr, eng, nbl, nso, sot, ssw, tsn, tso, ven, xho, zul`.

No calibrated probability is returned. This is a closed-set classifier: it does not detect unsupported languages, mixed-language content, malicious input or uncertainty reliably. Blank or excessively long inputs are rejected. The CLI is a local demonstration, not an authenticated or rate-limited network service.

## Data and evaluation

The original labelled CSV is used locally without new external text. Identical normalized texts are collapsed, and conflicting-label groups are excluded before stratified 60/20/20 splitting. Vectorizers are fitted on the training partition only. Macro F1 is the predeclared selection metric, followed by one final holdout evaluation. Per-language support/precision/recall/F1 and confusion counts are reported so overall accuracy does not hide weaker classes.

The holdout was newly reserved for this 2026 protocol; the source corpus had already been explored in 2023 coursework. Exact duplicate control does not prove source independence. Document IDs, temporal provenance and near-duplicate audit are unavailable. In-distribution results can be optimistic for real-world messages. Performance on dialects, social media, short texts, names and code-switching remains unverified.

## Responsible use

Do not use this prototype for decisions about a person's identity, ethnicity, citizenship, access to services or suitability for employment. Text language is not a reliable proxy for those attributes. Per-language differences alone neither prove nor disprove fairness. Meaningful deployment would require appropriate data rights, domain validation, group coverage, unknown-language handling, privacy/security controls and ongoing error review.

No production deployment, uptime, throughput guarantee or real-user benefit is established. Software test counts and model performance are separate evidence.

## Reproducibility and ownership

Dependencies are pinned in requirements.txt. Local run markers prevent accidental reselection or repeat holdout evaluation in a run directory; source/environment/data checksums detect drift. These are workflow controls, not protection against deliberately rewritten local records. Never deserialize legacy or third-party model binaries. An artifact checksum establishes expected bytes, not safety of an untrusted pickle.

The original project is ExploreAI coursework/hackathon work. The new source, tests and reporting were developed with Codex assistance. No independent owner proficiency or teach-back assessment has been performed. Historical files and contribution history are preserved; attribution and data/model rights are discussed in [the provenance note](DATA_AND_ATTRIBUTION.md).
