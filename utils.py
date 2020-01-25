
import numpy as np

# Plot random explorations
def plot_exp(images, ID,
             titles=['left', 'left-front',
                     'right', 'right-front']):
    ''' Plot one exploration. '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(12,8))
    fig.subplots_adjust(top=1.35)

    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    fig.suptitle('Patient ID: ' + ID)
    plt.show()

def feature_vector_df(feat_vector, IDs, birads_data):
    data = pd.DataFrame(feat_vector)
    data['internal_ID'] = IDs
    data = pd.merge(data, birads_data[['BIRADS', 'internal_ID']], how='left')
    data = data.dropna()
    data = data[data.BIRADS!=0]
    data.index = data.internal_ID
    data['target']  = np.where(data['BIRADS']>3,1,0)
    data = data.drop(columns = (['internal_ID', 'BIRADS']))
    pos_ids = data.index[data.target==1]
    return data, pos_ids

def performance(X, y, X_test, y_test, detailed = False, title=''):
    """ Obtain tuned LogReg model classification performance.

    Inputs
    ----------
    X: dataframe or numpy array
         training set with features
    y: np.array
         training targets
    X_test: dataframe or numpy array
         test set with features
    y_test: np.array
         test targets

    detailed: bool
         Indicates whether to print a detailed
         report and ROC curve or not
    title: str
         A title that is given to the ROC
         curve plot and the AUC score


    Returns
    -------
    AUC score for a specific technique
    Optional
    """
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                         C=1.0, fit_intercept=True, intercept_scaling=1,
                         class_weight=None, random_state=None, solver='lbfgs',
                         max_iter=100, multi_class='auto', verbose=0,
                         warm_start=False, n_jobs=None, l1_ratio=None)
    clf.fit(X, y)
    probs = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    if detailed:
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(title)
        plt.show()
        report_performance(y_test, probs)
    return(auc(fpr,tpr))
