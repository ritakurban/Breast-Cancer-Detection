from setup import *

# Helper functions
def plot_exp(images, ID,
             titles=['left', 'left-front',
                     'right', 'right-front']):
    ''' Plot one exploration. 
    
    Inputs
    ----------
    images: numpy array
         4 temperature arrays
    ID: str
         Patient ID number
    titles: list of strings (optional)
         Titles of respective images
           
    Returns
    -------
    Plot of one exploration with ID
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(12,8))
    fig.subplots_adjust(top=1.35)

    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    fig.suptitle('Patient ID: ' + ID)
    plt.show()


def conf_matrix_plot(orig_data, orig_y, 
                     aug_data, aug_y, 
                     test_data_orig, test_data_aug, 
                     test_y_orig, test_y_aug, 
                     model=LogisticRegression()):
    ''' 
    Plot confusion matrices for original and augmented 
    dataset in absolute and percentage values 
    
    Inputs
    ----------
    images: numpy array
         4 temperature arrays
    ID: str
         Patient ID number
    titles: list of strings (optional)
         Titles of respective images
           
    Returns
    -------
    Dataframe with birads_data and IDs
    List of positive IDs
    '''
    # Use scikit clone to make a deep copy of the model
    model1 = clone(model)
    cmap = sns.light_palette((210, 90, 60), 
                             input="husl", 
                             as_cmap=True)
    model.fit(orig_data, orig_y)
    preds = model.predict(test_data_orig)

    cm = confusion_matrix(test_y_orig, preds)
    # Normalize
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, axs = plt.subplots(2, 2,figsize=(12,10))
    axs[0, 0] = sns.heatmap(cm, annot=True, fmt='.2f', 
                            cmap=cmap, ax=axs[0, 0])
    axs[0, 0].set_title('Original Dataset Absolute')
    axs[0, 1] = sns.heatmap(cmn, annot=True, fmt='.2f', 
                            cmap=cmap, ax=axs[0, 1])
    axs[0, 1].set_title('Original Dataset Percentage')
    model1.fit(aug_data, aug_y)
    preds = model1.predict(test_data_aug)
    cm = confusion_matrix(test_y_aug, preds)
    # Normalize
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    axs[1, 0] = sns.heatmap(cm, annot=True, fmt='.2f', 
                            cmap=cmap, ax=axs[1, 0])
    axs[1, 0].set_title('Augmented Absolute')
    axs[1, 1] = sns.heatmap(cmn, annot=True, fmt='.2f', 
                            cmap=cmap, ax=axs[1, 1])
    axs[1, 1].set_title( 'Augmented Percentage')
    plt.show()
    
    
def plot_tsne(X, y, title='', alpha1=0.3, 
              alpha2=0.4, plot_legend=True):
    '''
    Perform t-sne using the resampled data (can be done
    without augmented data) and blind data.
    
    Inputs
    ----------
    X: pandas dataframe
         Dataframe with all features
    y: list
         Target values
    title: str (optional)
         Plot title
    alpha1, alpha2: float
         Transparency of observations from class 1 and 2
    plot_legend: str (optional)
         Indicator of whether to add legend
        
    Returns
    -------
    t-SNE plots in 2D
    '''
    x = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)

    tsne = TSNE(n_components=2,
                 init='random',
                 learning_rate=5,
                 random_state=1,
                 perplexity=10,
                 early_exaggeration=100,
                 n_iter=550,
                 metric='euclidean',
                 angle=0.2)

    y_tsne = tsne.fit_transform(X_norm)
    
    green = y == 0
    red = y == 1

    plt.scatter(y_tsne[green, 0], y_tsne[green, 1], 
                c="g", alpha = alpha1, label='Healthy')
    plt.scatter(y_tsne[red, 0], y_tsne[red, 1], 
                c="r", alpha = alpha2, label='Cancer')
    plt.axis('off')
    plt.title(title)
    if plot_legend:
        plt.legend()
    plt.show()


def feature_vector_df(feat_vector, IDs, birads_data):
    ''' Create a df of features.
    
    Inputs
    ----------
    feat_vector: pandas dataframe
         Dataframe with all features
    IDs: list
         List of patient IDs 
    birads_data: list
         List of BI-RADS values
        
    Returns
    -------
    Dataframe with birads_data and IDs
    List of positive IDs
    '''
    data = pd.DataFrame(feat_vector)
    data['internal_ID'] = IDs
    data = pd.merge(data, birads_data[['BIRADS', 'internal_ID']], 
                    how='left')
    data = data.dropna()
    data = data[data.BIRADS!=0]
    data.index = data.internal_ID
    data['target']  = np.where(data['BIRADS']>3, 1, 0)
    data = data.drop(columns = (['internal_ID', 'BIRADS']))
    pos_ids = data.index[data.target==1]
    return data, pos_ids


def performance(X, y, X_test, y_test, 
                detailed = False, title='', 
                report=False, model=LogisticRegression()):
    ''' Obtain classification performance.

    Inputs
    ----------
    X: dataframe or numpy array
         Training set with features
    y: np.array
         Training targets    
    X_test: dataframe or numpy array
         Test set with features
    y_test: np.array
         Test targets    

    detailed: bool
         Indicates whether to print a detailed
         Report and ROC curve or not
    title: str
         A title that is given to the ROC
         curve plot and the AUC score

    Returns
    -------
    AUC score for a specific technique
    Plot of the ROC curve
    '''
    clf = model
    clf.fit(X, y)
    try:
        probs = clf.predict_proba(X_test)[:,1]
    except:
        probs = clf.predict(X_test)
    if report:
        preds = clf.predict(X_test)
        target_names = ['Negative', 'Positive']
        print(metrics.classification_report_imbalanced(y_test, preds, 
                                                       target_names=target_names))
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    if detailed: 
        plt.plot(fpr, tpr, label=title + ':'+ str(np.round(auc(fpr,tpr),2)))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend()
    return(auc(fpr,tpr))


def plot_AUC(aucs, title=''):
    '''Plot a set of AUC scores.

    Inputs
    ----------
    aucs: list
         List of AUC scores
    max_comp: int
         Maximum number of components 
    title: str
         A title that is given to the plot 

    Returns
    -------
    Plot of AUC values
    Maximum AUC score
    '''
    plt.plot(np.arange(0, 1.1, 0.1), aucs)
    plt.xlabel('Float Value')
    plt.ylabel('Average AUC Score over 100 Repetitions')
    plt.title(title)
    plt.show()
    print('Max AUC: ', max(aucs))