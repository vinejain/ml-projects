**BINARY CLASSIFICATION MODEL FRAMEWORK**

A generic model dispatcher framework for binary classification problems. No need to start from scratch, model can be updated in and chosen from the dispatcher.

- **Instructions**



1) Open Terminal and clone the directory

```
git clone https://github.com/vinejain/random-forest-classifier-framework.git
```

2) Change directory to /random-forest-classifier-framework/src

```
cd /random-forest-classifier-framework/src
```

3) Execute the shell script 'run.sh'

```
sh run.sh
```


- **Description**



**INPUT**

Create a directory INPUT and fetch the categorical data from below link
https://www.kaggle.com/c/cat-in-the-dat/data)


**MODELS**

Create a directory MODELS where TRAIN.PY will save the model using joblib

**DISPATCHER**

SRC/DISPATCHER.py dispatches the required model

**K-FOLD Cross Validation**

SRC/CREATE_FOLDS.py creates Stratified Folds with 5 splits on the training dataset, and saves output file in INPUT/TRAIN_FOLDS.csv

See new column 'Fold'

**TRAINING**

SRC/TRAIN.py will train the data and dump the models in MODELS/.

Following will be fetched using os.environ.get:
TRAINING_DATA
TEST_DATA
FOLD
MODEL

METRICS: Since the data is skewed, using ROC_AUC. On current data it gives score of 0.74 with FOLD=4


**PREDICT**

SRC/PREDICT.py will predict_proba() to predict the probability of 1s, and save it in MODELS/rf_output.csv

**RUN.SH**

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
python -m src.predict

OPEN TERMINAL and RUN below command to run the model:
    
$ sh src/run.sh randomforest
