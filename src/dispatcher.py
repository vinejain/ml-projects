# Dispatcher will dispatch any model directly without messing with the training code

from sklearn import ensemble

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}

# A random forest is a meta estimator that fits a number of decision tree classifiers on various 
# sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# ExtraTreesClassifier class implements a meta estimator that fits a number of randomized decision trees 
# (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive 
# accuracy and control over-fitting.
