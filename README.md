# Cross-Species Transfer in Terrestrial Mammal Behavior Classification

R and Python scripts for my thesis at IZW on cross-species transfer in terrestrial mammal behavior classification using tri-axial accelerometer data.

<img width="239" height="239" alt="image" src="https://github.com/user-attachments/assets/b571eb29-b010-4de4-b79e-93acc17463b1" />

## Overview

This repository contains R and Python scripts for analysing multiple tri-axial accelerometer datasets from terrestrial mammals.

The project investigates how well behaviour classification models generalise across datasets and species, and whether transfer performance depends on behavioural class and biological distance between species.

Included species currently comprise:

- Dog
- Fox
- Hedgehog
- Bison
- Cattle
- Horse
- Giraffe
- Raccoon

## Research question

To what extent do behaviour classification models trained on accelerometer data generalise across datasets and species, and how is this generalisation influenced by behavioural class and biological distance?

## Hypotheses

- **H1 (transfer gap):** Behaviour classification performance will be higher in within-dataset evaluations than in cross-dataset transfer evaluations.
- **H2 (behaviour-class effect):** Cross-dataset performance differs across broad behavioural classes.
- **H3 (distance effect):** Cross-species transfer performance will decline with increasing biological distance between source and target species.

## Models

We selected six models representing three complementary modelling families:

- **Feature-based tree ensembles:** Random Forest, LightGBM
- **Deep learning on raw time series:** baseline CNN, InceptionTime
- **Specialised time-series classifiers:** MultiRocket, Hydra

## Evaluation design

The evaluation is structured into five complementary settings:

- **Within-dataset:** baseline performance under minimal domain shift
- **Inter-dataset:** transfer between related datasets
- **Cross-dataset:** transfer to entirely unseen datasets
- **Pairwise:** fine-grained source-target transfer comparisons
- **Global:** pooled multi-dataset training across datasets

Together, these settings separate within-dataset accuracy from increasingly challenging transfer scenarios.

## Repository structure

- `R/` – R scripts
- `python/` – Python scripts
- `outputs/` – processed outputs
- `figures/` – plots and visual outputs
- `images/` – README images

## Data availability

Most datasets used in this project are not publicly available.

Publicly available datasets used in this project include:

- [*Horsing Around -- A Dataset Comprising Horse Movement*](https://data.4tu.nl/articles/_/12687551/1) (Horse)
- [*Japanese Black Beef Cow Behavior Classification Dataset*](https://zenodo.org/records/5849025#.ZE-y_3ZByHu) (Cattle)

## License

This repository is licensed under the MIT License.
