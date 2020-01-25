import os
os.chdir('..')

 # Import Eva packages
from features_preprocess_pipeline import *
from src import params
from src import pathfiles as pf
from src import imageLoad as imgL
from src.data import preprocess
from src.data import utils

# Import public packages
import itertools
from datetime import datetime
import time
from pathlib import Path
import pandas as pd
import math
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas_profiling 
from sklearn import metrics
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import skimage.segmentation as seg
from skimage.feature import local_binary_pattern, hog
from lpproj import LocalityPreservingProjection 
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")


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
interim_objects = ['temps', 'box_temps', 'clean_mask', 'templist']


# Get BIRADS data
birads_file = Path(temps_path / pf.BIRADS_DATA)
birads_data = pd.read_csv(birads_file)
birads_data = birads_data[birads_data.discard!=1]
birads_data = birads_data[['internal_ID', 'BIRADS']]
birads_data = birads_data.drop_duplicates(subset='internal_ID', keep='last')


# Load patient explorations
exp = utils.get_preprocess_info(temps_fnames, masks_fnames,
                                            [], position_codes,
                                            interim_path, interim_objects,
                                            discard_existing=False)[1]
exp['internal_ID'] = exp.index.get_level_values(0).values
exp = exp.drop_duplicates(subset='internal_ID', keep='last')

# Merge explorations with birads_data to ensure only valid observations are added
explorations = pd.merge(exp, birads_data, on='internal_ID', how='left')
explorations.index = exp.index
explorations = explorations[(explorations.BIRADS!=0) & ~(np.isnan(explorations.BIRADS))]

# Get raw temperatures and masks
temps_fnames_cols = [f for f in explorations.columns if 'temp' in f]
masks_fnames_cols = [f for f in explorations.columns if 'mask' in f]

temps = []
IDs = []
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
    
    # Crop rawtemperature arrays
    ID_splits = preprocess.preprocess(ID_temps,
                                      ID_masks,
                                      split_pos_codes)
    IDs.append(ID)
    temps.append([ID_splits['left']['cut_temps'],
                  ID_splits['left-front']['cut_temps'],
                  ID_splits['right']['cut_temps'],
                  ID_splits['right-front']['cut_temps']])
    i+=1
    
    # Report progress
    update_progress(i/len(explorations))

print('Added {} patients'.format(len(temps))) 