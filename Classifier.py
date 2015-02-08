from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

def Classify(features, labels, namesClasses):

    print "Training"
    # n_estimators is the number of decision trees
    # max_features also known as m_try is set to the default value of the square root of the number of features
    clf = RF(n_estimators=100, n_jobs=3);
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5, n_jobs=1);
    print "Accuracy of all classes"
    print np.mean(scores)


    kf = KFold(labels, n_folds=5)
    y_pred = np.zeros((len(labels),len(set(labels))))
    for train, test in kf:
        features_train, features_test, labels_train, labels_test = features[train,:], features[test,:], labels[train], labels[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(features_train, labels_train)
        y_pred[test] = clf.predict_proba(features_test)

    print classification_report(labels, y_pred, target_names=namesClasses)

    return y_pred