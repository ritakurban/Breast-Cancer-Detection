# Capstone Project
### Rita Kurban
### 25 January 2020

# Files:

1. `setup.py` - Loads all necessary public and Eva packages; select relevant
explorations and their BI-RADS values; pre-process raw temperature arrays using masks.

2. `utils.py` - Contains all functions that are used across notebooks. In progress.

3. `feature_engineering.ipynb` - Calculates HOGs, LBPs, delta-Ts, and segmentation
 features. Also implements dimensionality reducing techniques. Analyses performance.


4. `human_evaluation.ipynb` - Implements and compares two quantitative thermography
 methods from the literature.

5. `data_exploration.ipynb` -- Analyzes the distribution of BI-RADS scores, data
sources, and offers PCA and T-SNE based 2-D visualizations of the dataset.

6. `oversampling.ipynb` - Compares different oversampling methods with various `float`
values. Produces t-SNE visualizations of oversampled data. In progress.

7. `outlier_detection.ipynb` - Implements outlier detection techniques for
artificial labeling of samples with no recorded BI-RADS values. In progress.
