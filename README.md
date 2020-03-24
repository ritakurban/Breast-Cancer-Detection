# Capstone Project

### Rita Kurban

### 24 March 2020

Even though breast cancer detection is a heavily researched discipline, thermography-based methods have long received insufficient attention. This can be mainly attributed to experimental design flaws, such as too small sample sizes, especially for positive cases, mixed data acquisition methods with unclear pre-processing steps, and omission of controversial or unclear cases, leading to non-generalizable and non-reproducible results. 
This repository contains notebooks that quantify and compare human evaluation methods for thermography interpretation available in the literature. It also demonstrates that modern feature engineering and data augmentation techniques, combined with a sufficient number of samples, help to overcome the problem of the imbalanced dataset, and achieve industry-level performance.

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

