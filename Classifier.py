from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

def Classify(features, labels, label2ClassName,features_test):

    '''

    '''

    # n_estimators is the number of decision trees
    # max_features also known as m_try is set to the default value of the square root of 
    # the number of features
    clf = RF(n_estimators=1200, n_jobs=-1)
    clf.fit(features, labels)
    predictedProbs = clf.predict_proba(features_test)
    
    # build a list of class names ordered the same way as the probabilities in the rows of 
    # predictedProbs
    classNameSet = []
    for label in clf.classes_:
        classNameSet.append(label2ClassName[label])

    return predictedProbs,classNameSet,clf