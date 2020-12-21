import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib # used to run python functions as pipeline jobs

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist() + 
                df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]  
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    
    # using roc_auc since the data is skewed
    print(metrics.roc_auc_score(yvalid, preds))

    # Saving and loading ML models using joblib or pickle: 
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    # 
    # IMP: Take note of the version so that you can re-create the environment if for some reason 
    # you cannot reload your model on another machine or another platform at a later time.
    
    # Tips for Saving Your Model: 
    # 1. Python Version. Take note of the python version. 
    # You almost certainly require the same major (and maybe minor) version of Python used to 
    # serialize the model when you later load it and deserialize it.
    # 2. Library Versions. The version of all major libraries used in your machine learning project 
    # almost certainly need to be the same when deserializing a saved model. 
    # This is not limited to the version of NumPy and the version of scikit-learn.
    
    # The model, label encoders, and  columns for any fold will be saved in models directory
    # Models will be loaded in the predict module

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")