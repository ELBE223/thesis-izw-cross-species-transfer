
# Cross-Species Transfer in Terrestrial Mammal Behavior Classification

This repository contains the code and selected supplementary files for the master’s thesis:

**Cross-Species Transfer in Terrestrial Mammal Behavior Classification Using Tri-axial Accelerometer Data**

The project evaluates how well accelerometer-based behaviour classification models generalise across heterogeneous datasets and terrestrial mammal species.

<img width="1254" height="1254" alt="M A _Animals" src="https://github.com/user-attachments/assets/60189d8f-90ce-4210-9360-bf3efe61177d" />

## Project overview

Tri-axial accelerometer data were harmonised across multiple terrestrial mammal datasets and mapped to three broad behavioural classes:

- foraging
- locomotion
- resting

The study compares within-dataset performance, cross-dataset transfer, pairwise source–target transfer, and cross-species transfer in relation to functional-biomechanical trait distance.

## Models

Six classifiers from three complementary modelling families were compared:

- Feature-based tree ensembles: Random Forest, LightGBM
- Deep learning on raw time series: CNN, ResNet
- Time-series classifiers: HYDRA, MultiRocket

## Repository structure

- `R/` – statistical analysis and visualisation scripts
- `Python/` – preprocessing, modelling, and classification scripts
- `Traits/` – trait data and distance calculations used for the transfer analysis
- `supplementary_data/` – selected supplementary CSV files, ethogram harmonisation tables, trait-distance outputs, and sensitivity-analysis files

## Supplementary data

The folder `supplementary_data/` contains selected derived supplementary files used in the thesis, including:

- ethogram harmonisation tables
- trait profiles
- functional-biomechanical distance outputs
- cross-species transfer-analysis tables
- sensitivity-analysis files
- selected figure/table source data

These files are derived outputs and supporting metadata. They are included to improve transparency and reproducibility of the analyses.

## Data availability

Most raw accelerometer datasets used in this project are not publicly available and therefore cannot be shared in this repository.

This repository includes selected derived supplementary files only. Full reproduction of all analyses is limited by raw dataset availability.

Publicly available datasets used in this project include:

- Horsing Around -- A Dataset Comprising Horse Movement
- Japanese Black Beef Cow Behavior Classification Dataset

## Reproducibility

This repository provides the code used for preprocessing, modelling, evaluation, statistical analysis, and visualisation.

Selected derived supplementary CSV files are included in `supplementary_data/`.

Full reproduction of all analyses requires access to the original accelerometer datasets.

## Software

The analyses were implemented mainly in:

- Python 3.11
- R 4.5.2

Python was used for preprocessing, feature extraction, model training, and prediction.

R was used for statistical analysis, hypothesis testing, and visualisation.

## Thesis context

This repository accompanies a master’s thesis submitted in the M.Sc. Data Analytics programme at Justus Liebig University Giessen.

## License

Please see the `LICENSE` file for licensing information.

## AI use

The use of AI tools is documented in `AI_USE_STATEMENT.md`.
