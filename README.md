# Capstone Project
### Rita Kurban
### 24 March 2020

# Files:

1. `setup.py` - Loads all public and Eva packages, selects relevant explorations and their BI-RADS values, pre-processes raw temperature arrays using masks.

2. `utils.py` - Contains helper functions that are used across notebooks to minimize code duplicaion.

3. `human_evaluation.ipynb` - Implements two quantitative thermography
methods from the literature and compares them to subjective evaluation.

4. `ML_pipeline.ipynb` - Calculates HOGs, LBPs, delta-Ts, and segmentation
features and implements dimensionality reducing techniques. Performs model selection, hyperparameter tuning, and feature selection. Analyzes performance.

5. `data_exploration.ipynb` -- Analyzes the distribution of BI-RADS scores, data
sources, and offers PCA and T-SNE based 2-D visualizations of the training set.

6. `anomaly_detection.ipynb` - Implements anomaly detection techniques for extracting information from samples with no recorded BI-RADS values.

7. `sampling_methods.ipynb` - Compares undersampling and oversampling methods with various `float` values.  Produces t-SNE visualizations of sampled data.

