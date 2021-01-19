import pandas as pd
from sklearn import model_selection

if __name__ == '__main__' :
    #load training data
    df = pd.read_csv('./train.csv')

    #create a new col called 'kfold' and fill with -1
    df['kfold'] = -1

    #shuffle the rows of dataset
    df = df.sample(frac=1).reset_index(drop=True)

    #initiate kfold class 
    kf = model_selection.KFold(n_splits=5)

    #fill the new kfold col
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_,'kfold'] = fold
    
    df.to_csv('train_folds.csv',index=False)