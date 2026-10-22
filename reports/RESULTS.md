# Local experiment results

Run date: 22 October 2026.

These are freshly executed local results on a reserved internal holdout, not Kaggle scores. Old notebook scores use different preparation/splits and are not a comparable baseline.

## Data audit

Original labelled rows: 33,000. Retained distinct normalized texts: 29,948. Duplicate rows collapsed: 3,052. Conflicting-label groups excluded: 0 (0 rows).

Seed 42; sizes: {'train': 17968, 'validation': 5990, 'holdout': 5990}. Exact normalized-text overlap: zero. Near duplicates and original document sources were not independently identified.

## Validation comparison

Selection used validation macro F1 only; candidates were fixed beforehand.

| Candidate | Macro F1 | Accuracy | Fit seconds | Features |
|---|---:|---:|---:|---:|
| word_tfidf_nb | 0.997838 | 0.997830 | 1.816 | 60,000 |
| char_tfidf_svm | 0.999357 | 0.999332 | 25.497 | 135,365 |

Frozen choice: **char_tfidf_svm**. Fitted on training rows only; no refit or post-holdout tuning.

## One-use holdout

Rows: 5,990; macro F1: **0.999168**; weighted F1: 0.999165; accuracy: 0.999165; lowest per-language F1: 0.997438.

| Language code | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| afr | 1.000000 | 1.000000 | 1.000000 | 528 |
| eng | 0.998336 | 1.000000 | 0.999167 | 600 |
| nbl | 1.000000 | 0.995763 | 0.997877 | 472 |
| nso | 0.998264 | 1.000000 | 0.999131 | 575 |
| sot | 1.000000 | 0.998236 | 0.999117 | 567 |
| ssw | 1.000000 | 1.000000 | 1.000000 | 485 |
| tsn | 1.000000 | 1.000000 | 1.000000 | 574 |
| tso | 1.000000 | 1.000000 | 1.000000 | 551 |
| ven | 1.000000 | 1.000000 | 1.000000 | 521 |
| xho | 0.998120 | 0.998120 | 0.998120 | 532 |
| zul | 0.996587 | 0.998291 | 0.997438 | 585 |

Language-code differences are descriptive diagnostics, not proof of fairness. The confusion matrix and predeclared character-length slices are included in metrics.json.

## Reproducibility

See the README for the exact prepare, select, evaluate and predict commands.

- Dataset SHA-256: `155cd842622c677434ced8bcdd635facf39d8d139a35dbb671b1eb67bd1a6626`
- Package source SHA-256: `e912890e98479ca8f26f2f977509fc518745fd75447e2a9f967bf0aef54d5c92`
- Frozen protocol SHA-256: `3feedf09cdc4a0281951617832f67fe681455fc4cbb21ff20e77cd14bf87a991`
- Split SHA-256: `511b21bafb7b65d85f057312922abe7939f96382a08abcb27676aa7e31c60b6c`
- Environment: `{"joblib": "1.5.2", "numpy": "2.2.6", "platform": "Windows-10-10.0.19045-SP0", "python": "3.12.14", "scikit_learn": "1.7.2", "scipy": "1.16.2", "thread_limit": 1, "threadpoolctl": "3.6.0"}`

Runtime is observed on this local host, not a latency guarantee. No fresh-machine reproduction, external corpus, production traffic, out-of-distribution or fairness validation was performed. All candidates used the same frozen training/validation partitions. The final holdout informed no model changes.

## Interpretation

The result measures this deduplicated text benchmark. Shared corpus/document sources, near duplicates and a historical corpus previously explored in coursework limit generalization claims. Short text, unseen domains, unknown languages and code-switching require separate future evaluation; the current model always chooses among the eleven known labels.
