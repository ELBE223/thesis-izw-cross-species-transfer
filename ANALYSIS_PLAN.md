# Pre-specified re-analyses for the paper

Date: 2026-10-13

This plan was written before the results of the leave-one-individual-out (LOIO)
within-dataset evaluation and of the H3 sensitivity analyses below were available.
The commit history of this file documents the timing. Primary analyses of the
thesis remain unchanged unless stated otherwise; all analyses listed here are
reported in full, regardless of their outcome.

## 1. Statistical tests for H1 and H2

- Paired Wilcoxon signed-rank tests use exact two-sided p-values (normal
  approximation only if ties or zero differences occur; the method used is
  exported as `test_method`).
- H2 post hoc contrasts are Holm-adjusted within each classifier and are
  interpreted only for classifiers with a significant Friedman test.

## 2. Within-dataset baseline (H1)

- Within-dataset performance is re-estimated with leave-one-individual-out
  (`Python/within_loio.py`): every individual is the test unit once, the model is
  trained on the remaining individuals of the same dataset, and predictions of all
  folds are pooled, so every window is predicted once, as in the cross-dataset mode.
- Models, settings, training cap (250 windows per behavior), minimum counts
  (30 training windows per behavior), and metrics are unchanged from the model
  scripts. Seed 42; CNN and ResNet on CPU for reproducibility.
- Primary H1 re-analysis: datasets in which every individual can be the test unit
  under the unchanged minimum-count rule. Based on class counts per individual
  (not on performance), Hedgehog_dataset_1 and Raccoon_dataset_2 are expected to
  lack complete coverage, giving 10 paired datasets.
- Sensitivity: all datasets with at least one valid fold
  (`metrics_all_incl_partial.csv`).
- Optional: seeds 42-46 for CNN and ResNet, reported descriptively as seed
  variability.

## 3. H3 sensitivity analyses

The primary H3 model remains:

    Macro-F1 ~ z(core Gower distance) + classifier
               + (1 | unordered species pair) + (1 | directed species pair)
               + (1 | source species) + (1 | target species)

fitted to 432 records (72 directed species pairs x 6 classifiers), with
same-species transfers excluded. Inference for the distance effect uses a
likelihood-ratio test of ML fits; beta, SE, and 95 % CI come from the REML fit.
Each sensitivity analysis uses the same random-effects structure and inference.

Within-dataset values in S1 and S2 are the LOIO Macro-F1 of the same classifier
(section 2). Datasets without complete LOIO coverage use their partial LOIO value,
so that all 432 records are retained. Covariates and responses are computed at the
dataset-pair level and then averaged to the species-pair level in the same way as
the primary response.

### S1: Within-dataset performance as covariates

    Macro-F1 ~ z(core Gower distance) + classifier
               + z(source within Macro-F1) + z(target within Macro-F1)
               + same random effects

- Question: does trait distance explain transfer performance beyond how well the
  classifier performs within the source and the target dataset?
- The likelihood-ratio test compares this model with the same model without trait
  distance. Correlations between covariates and trait distance and variance
  inflation factors are reported.

### S2: Relative transfer (target-normalized retention)

    retention = pairwise Macro-F1 (source -> target) / within Macro-F1 (target)

- Same model structure as the primary model, with retention as the response.
- Question: is the share of the achievable in-domain performance that is retained
  under transfer associated with trait distance?
- log(retention) is reported as an additional check of distributional assumptions.

### S3: Alternative distance measure

To be specified before computation (decision pending, see commit history).

## 4. Rules that apply to all re-analyses

- No inclusion thresholds based on classification performance are used.
- Sensitivity analyses are interpreted descriptively (direction, magnitude, and
  confidence intervals relative to the primary analysis), not as additional
  confirmatory tests.
