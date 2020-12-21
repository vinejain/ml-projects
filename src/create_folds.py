# What is k-fold cross validation and why do we need it?
## https://machinelearningmastery.com/k-fold-cross-validation/#:~:text=Cross%2Dvalidation%20is%20a%20resampling,is%20to%20be%20split%20into.
# We will be using Stratified KFold Cross Validation in our case

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    # make new column kfold
    df["kfold"] = -1
    
    # shuffle the data, reset & drop indices
    df = df.sample(frac = 1).reset_index(drop=True)

    # using StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    #save the file
    df.to_csv("input/train_folds.csv", index = False)
    