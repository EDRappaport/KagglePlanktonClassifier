from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

def KFoldCrossValidate(features, labels, namesClasses):

    '''Do KFolds cross validation and print a classification report
       TO-DO: strip out the random forest classifier and have this take a classifier as an 
              argument
    '''

    kf = KFold(labels, n_folds=5)
    y_pred = np.zeros((len(labels),len(set(labels))))
    for train, test in kf:
        features_train, features_test, labels_train, labels_test = features[train,:], features[test,:], labels[train], labels[test]
        # n_estimators is the number of decision trees
        # max_features also known as m_try is set to the default value of the square root of 
        # the number of features
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(features_train, labels_train)
        y_pred[test] = clf.predict_proba(features_test)

    print classification_report(labels, y_pred, target_names=namesClasses)

    logLoss = multiclass_log_loss(labels, y_pred)

    print("Final log loss: " + logLoss)

    return 