# Beyond Adiabatic TDDFT on the Hubbard Dimer using Echo State Networks

Code and data accompanying the paper:

> **Beyond Adiabatic TDDFT on the Hubbard Dimer using Echo State Networks**  
> [Author names], [Journal], [Year]  
> [arXiv link]

## Overview

This repository contains the code and data for training Echo State Network (ESN) functionals that map time-dependent external potentials to electron densities. The Hubbard dimer and a four-site tight-binding model serve as benchmark systems.

Two ESN implementations are examined:
- **Direct ESN functional** — uses only the external potential as input
- **Frequency-augmented ESN functional** — incorporates instantaneous transition frequencies

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── ESN_data_gen_v1_4.py              # Shared utility module (data generation routines)
├── generate_figures.py               # Generates Figs. 3–9 of the paper
├── generate_tables.py                # Generates Tables II & III of the paper
├── reduce_dataset.py                 # Utility to reduce JSON file sizes
├── plots/                            # Pre-generated figures (Figs. 3–9)
│   ├── Fig3.png
│   ├── ...
│   └── Fig9.png
├── sec3a/                            # Sigmoid switch potential (Sec. III A)
│   ├── ESN_data_gen_v1_3_3.py        # Shared utility module
│   ├── data_generation.py            # Generates full training/test datasets
│   ├── direct_training.py            # Trains the direct ESN functional
│   ├── augmented_training.py         # Trains the frequency-augmented ESN functional
│   ├── test5_single_100.py           # Training script for Fig. 5 (N=100)
│   ├── test5_single_200.py           # Training script for Fig. 5 (N=200)
│   ├── test5_single_300.py           # Training script for Fig. 5 (N=300)
│   ├── test1_train_data.json         # Training data (reduced)
│   ├── test1_test_data.json          # Test data (reduced)
│   ├── test1_direct_results.json     # Direct ESN predictions
│   ├── test1_augmented_results.json  # Augmented ESN predictions
│   ├── test1_direct_study.db         # Optuna hyperparameter optimization (direct)
│   ├── test1_augmented_study.db      # Optuna hyperparameter optimization (augmented)
│   ├── test5_train_data.json         # Training data for Fig. 5 (reduced)
│   ├── test5_test_data.json          # Test data for Fig. 5 (reduced)
│   ├── test5_single_results_100.json # Fig. 5 results (N=100)
│   ├── test5_single_results_200.json # Fig. 5 results (N=200)
│   ├── test5_single_results_300.json # Fig. 5 results (N=300)
│   ├── test5_single_study_100.db     # Optuna study for Fig. 5 (N=100)
│   ├── test5_single_study_200.db     # Optuna study for Fig. 5 (N=200)
│   └── test5_single_study_300.db     # Optuna study for Fig. 5 (N=300)
├── sec3b/                            # Adiabatic and non-adiabatic regimes (Sec. III B)
│   ├── ESN_data_gen_v1_3_3.py
│   ├── data_generation.py
│   ├── direct_training.py
│   ├── augmented_training.py
│   ├── test2_train_data.json         # Training data (reduced)
│   ├── test2_test_data.json          # Test data (reduced)
│   ├── test2_direct_results.json     # Direct ESN predictions
│   ├── test2_augmented_results.json  # Augmented ESN predictions
│   ├── test2_direct_study.db
│   └── test2_augmented_study.db
├── sec3c/                            # Resonant driving, near-degenerate states (Sec. III C)
│   ├── ESN_data_gen_v1_3_3.py
│   ├── data_generation.py
│   ├── test3_direct_training.py
│   ├── test3_augmented_training.py
│   ├── test3_train_data.json         # Training data (reduced)
│   ├── test3_test_data.json          # Test data (reduced)
│   ├── test3_validation_data.json    # Cross-validation data (reduced)
│   ├── test3_direct_results.json     # Direct ESN predictions
│   ├── test3_augmented_results.json  # Augmented ESN predictions
│   ├── test3_direct_study.db
│   ├── test3_augmented_study.db
│   └── test2_augmented_study.db
├── sec3d/                            # Four-site tight-binding model (Sec. III D)
│   ├── ESN_data_gen_v1_3_3.py
│   ├── test4_datagen.py              # Generates full training/test datasets
│   ├── test4_datagen_lite.py         # Lightweight data generation variant
│   ├── test4_single_tmax.py          # Training script
│   ├── new__plots3.py                # Supplementary plotting script
│   ├── test4_train_data_1200.json    # Training data (reduced)
│   ├── test4_test_data_1200.json     # Test data (reduced)
│   ├── test4_direct_1200.json        # Direct ESN predictions
│   └── test4_direct_1200.db
└── appA/                             # Appendix A: density-to-potential mapping
    ├── ESN_data_gen_v1_3_3.py
    ├── data_generation.py
    ├── test6_training.py
    ├── test1_train_data.json         # Training data (reduced)
    ├── test1_test_data.json          # Test data (reduced)
    ├── test1_direct_results.json     # ESN predictions
    └── test1_direct_study.db
```

## Reproducing Results

### Figures and Tables

The JSON data files included in this repository are **reduced-size versions** of the full datasets, to comply with Git file size limits. They contain sufficient data to reproduce the figures and tables from the paper:

```bash
python generate_figures.py    # Generates Figs. 3–9 (saved to plots/)
python generate_tables.py     # Generates Tables II & III
```

Pre-generated figures are also available in the `plots/` directory.

### Retraining the ESN Models

To retrain the ESN models from scratch, you must first regenerate the full datasets. The reduced JSON files included in this repository **do not** contain sufficient data for training. Within each subsection folder, run the data generation script first, then the training script. For example:

```bash
# Sec. III A — Sigmoid switch potential
cd sec3a/
python data_generation.py     # Generates full training/test datasets
python direct_training.py     # Trains the direct ESN functional
python augmented_training.py  # Trains the frequency-augmented ESN functional
```

The same pattern applies to the other sections — see the repository structure above for the corresponding script names in each folder.

## Installation

### Requirements

- Python >= 3.9
- See `requirements.txt` for package dependencies


## Key Dependencies

| Package       | Purpose                                      |
|---------------|----------------------------------------------|
| `reservoirpy` | Echo State Network implementation            |
| `numpy`       | Numerical computation                        |
| `scipy`       | ODE integration (8th-order Runge-Kutta)      |
| `matplotlib`  | Figure generation                            |
| `optuna`      | Bayesian hyperparameter optimization (TPE)   |

## Citation

If you use this code or data, please cite:

```bibtex
@article{AuthorYear,
  title   = {Beyond Adiabatic TDDFT on the Hubbard Dimer using Echo State Networks},
  author  = {},
  journal = {},
  year    = {2026},
  doi     = {}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
