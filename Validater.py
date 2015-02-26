from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

from MetricsEvaluation import multiclass_log_loss
from PlanktonClassifier import loadTrainingDataAndFeaturize

def KFoldCrossValidate(features, labels, namesClasses,n_est=1200):

    '''Do KFolds cross validation and print a classification report
       TO-DO: strip out the random forest classifier and have this take a classifier as an 
              argument
    '''

    kf = KFold(labels, n_folds=2)
    y_pred = np.zeros((len(labels),len(set(labels))))
    for train, test in kf:
        features_train = list()
        labels_train = list()
        features_test = list()
        labels_test = list()
        for ind in train:
            features_train.append(features[ind])
            labels_train.append(labels[ind])
        for ind in test:
            features_test.append(features[ind])
            labels_test.append(labels[ind])
        #features_train, features_test, labels_train, labels_test = features[train], features[test], labels[train],
        # labels[test]
        # n_estimators is the number of decision trees
        # max_features also known as m_try is set to the default value of the square root of 
        # the number of features
        clf = RF(n_estimators=n_est, n_jobs=-1)
        clf.fit(features_train, labels_train)
        y_pred[test] = clf.predict_proba(features_test)

    #print classification_report(labels, y_pred, target_names=namesClasses)

    logLoss = multiclass_log_loss(labels, y_pred)

    print("Final log loss: " + str(logLoss))

    return 

def _main_():    
    features_train,labels,label2ClassName = loadTrainingDataAndFeaturize()
    KFoldCrossValidate(features_train, labels, label2ClassName)


if __name__ == "__main__":
    _main_()