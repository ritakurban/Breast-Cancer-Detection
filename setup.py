# Import Eva packages
from features_preprocess_pipeline import *
from src import params
from src import pathfiles as pf
from src import imageLoad as imgL
from src.data import preprocess
from src.data import utils

# Import public packages
import os
import time
import math
import cv2
import shap
import random
import pickle
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from scipy import stats
from pathlib import Path
from imblearn import metrics
from datetime import datetime
from scipy.stats import ttest_ind
import skimage.segmentation as seg
from lpproj import LocalityPreservingProjection 
from skimage.feature import local_binary_pattern, hog
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'coolwarm'
plt.style.use('default')
plt.style.use('seaborn-muted')

# Sklearn methods
from sklearn.base import clone
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Anomaly detection 
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
from pyod.models.combination import aom, moa, average, maximization

# Sampling methods
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import *
from imblearn.over_sampling import *

# Set up directories & parameters
temps_path = project_dir / 'data/raw/'
masks_path =  project_dir / 'data/masks/'
interim_path = project_dir / 'data/interim/'

replacement_dict = params.REPLACEMENT_DICT
temps_pos_codes = params.TEMPS_POSITION_CODES
mask_pos_codes = params.MASK_POSITION_CODES
split_pos_codes = params.SPLIT_POSITION_CODES

temps_fnames = sorted(temps_path.glob('**/*.npy'))
masks_fnames = sorted(masks_path.glob('**/*.png'))

position_codes = {k: split_pos_codes[k]['code'] for k in split_pos_codes.keys()}
interim_objects = ['temps', 'box_temps', 
                   'clean_mask', 'templist']

# Get BI-RADS data
birads_file = Path(temps_path / pf.BIRADS_DATA)
birads_data = pd.read_csv(birads_file)
birads_data = birads_data[birads_data.discard!=1]
birads_data = birads_data[['internal_ID', 'BIRADS']]
birads_data = birads_data.drop_duplicates(subset='internal_ID', 
                                          keep='last')

# Load patient explorations
exp = utils.get_preprocess_info(temps_fnames, masks_fnames,
                                [], position_codes,
                                interim_path, interim_objects,
                                discard_existing=False)[1]
exp['internal_ID'] = exp.index.get_level_values(0).values
exp = exp.drop_duplicates(subset='internal_ID', keep='last')

# Merge explorations with birads_data to ensure only valid observations are added
explorations = pd.merge(exp, birads_data, 
                        on='internal_ID', 
                        how='left')
explorations.index = exp.index

# Get raw temperatures and masks
temps_fnames_cols = [f for f in explorations.columns if 'temp' in f]
masks_fnames_cols = [f for f in explorations.columns if 'mask' in f]

temps = []
IDs = []
birads = []

blind_temps = []
blind_IDs = []
blind_birads = []

i = 0
targets = []

# Get processed temperature arrays and corresponding IDs
for exp in explorations.index:
    ID = exp[0]

    # Get dictionary of position-file names per patient
    ID_temps_fnames = explorations.loc[exp, temps_fnames_cols]
    ID_positions = [x.split('_')[0] for x in temps_fnames_cols]
    ID_temps_fnames.index = ID_positions
    ID_temps_fnames = ID_temps_fnames.to_dict()

    ID_masks_fnames = explorations.loc[exp, masks_fnames_cols]
    ID_positions = [x.split('_')[0] for x in masks_fnames_cols]
    ID_masks_fnames.index = ID_positions
    ID_masks_fnames = ID_masks_fnames.to_dict()

    ID_temps = {k: imgL.loadTermo(v, gray=False, resolution=False) for
                k, v in ID_temps_fnames.items()}

    ID_masks = {k: imgL.loadMask(v, resolution=False) for
                k, v in ID_masks_fnames.items()}
    
    # Crop raw temperature arrays
    try:
        ID_splits = preprocess.preprocess(ID_temps,
                                          ID_masks,
                                          split_pos_codes)
    except:
        continue
    if explorations.loc[exp , : ].BIRADS in [1,2,3,4,5,6]:
        IDs.append(ID)
        temps.append([ID_splits['left']['cut_temps'],
                      ID_splits['left-front']['cut_temps'],
                      ID_splits['right']['cut_temps'],
                      ID_splits['right-front']['cut_temps']])
        birads.append(explorations.loc[exp , : ].BIRADS)
        i+=1
    else:
        blind_IDs.append(ID)
        blind_temps.append([ID_splits['left']['cut_temps'],
                      ID_splits['left-front']['cut_temps'],
                      ID_splits['right']['cut_temps'],
                      ID_splits['right-front']['cut_temps']])
        blind_birads.append(explorations.loc[exp , : ].BIRADS)
        i+=1
    
    # Report progress
    update_progress(i/len(explorations))