import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    #create new col called kfold and fill it with -1
    data['kfold'] = -1

    #randomize the rows
    data = data.sample(frac=1).reset_index(drop=True)

    #calc number of bins using Sturge's Rule
    num_bins = np.floor(1+np.log2(len(data)))

    # bin targets
    data.loc[:,'bins'] = pd.cut(data['target'],bins=num_bins,labels=False)

    #initiate kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f,(t,v) in enumerate(kf.split(X=data,y=data.bins.values)):
        data.loc[v,'kfold'] = f

    data = data.drop('bins',axis=1)

    return data

if __name__ == '__main__':
    #create sample dataset with 15,000 samples
    X,y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)

    df = pd.DataFrame(X,columns=[f'f_{i}' for i in range(X.shape[1])])
    df.loc[:,'target'] = y
    df = create_folds(df)
