# Experiment log

22 October 2026. One fixed local comparison; no GPU, external data, tracker or submission. Candidate definitions and selection rule were frozen in the local protocol before fitting.

| Candidate | Hypothesis | Fixed configuration | Validation macro F1 | Fit seconds | Decision |
|---|---|---|---:|---:|---|
| word_tfidf_nb | A simple word baseline can capture language-specific vocabulary | Word unigrams, 60,000-feature limit, MultinomialNB alpha 1 | 0.997838 | 1.816 | Retain as measured baseline |
| char_tfidf_svm | Character patterns can help distinguish related language vocabularies without a pretrained model | char_wb 3-5 grams, min_df 2, 150,000-feature limit, sublinear TF, LinearSVC C 1, seed 42 | 0.999357 | 25.497 | Selected on validation |

The validation difference is approximately 0.152 percentage points in macro F1. It compares these two newly executed pipelines on the same partitions; it is not a gain over an old notebook or leaderboard result.

Selection command wall time was 49.942 seconds including process startup, dataset validation, both fits, validation prediction and artifact save. The reported fit times cover each fit only. Character model feature count: 135,365. All numerical-library work was limited to one thread. Requirements: Python 3.12.14, scikit-learn 1.7.2, NumPy 2.2.6, SciPy 1.16.2, joblib 1.5.2, threadpoolctl 3.6.0, Windows 10.

After selection was saved, one final internal holdout evaluation returned macro F1 0.999168, weighted F1 0.999165 and accuracy 0.999165 on 5,990 rows. Five predictions were incorrect. Lowest per-language F1 was 0.997438. Evaluation command wall time: 20.076 seconds; batch prediction portion: 9.272 seconds. These are observed local batch timings, not request latency or production service guarantees.

No model code, normalization, configuration or selection was changed after this result. The saved pipeline is trained only on the 17,968-row training partition, with 5,990 validation rows. Dataset fingerprints, exact per-language metrics, source/configuration details, partition sizes and limitations are in reports/RESULTS.md and reports/metrics.json.

Next action: owner review and scoped code/documentation publication. Do not tune against this holdout. A later generalization study would need a separately justified source/domain-independent dataset and protocol; no such study was started here.
