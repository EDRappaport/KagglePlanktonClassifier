from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

def Classify(features, labels, namesClasses,features_test):

    '''

    '''

    # n_estimators is the number of decision trees
    # max_features also known as m_try is set to the default value of the square root of 
    # the number of features
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(features, labels)
    predictedProbs = clf.predict_proba(features_test)
    
    # build a list of class names ordered the same way as the probabilities in the rows of 
    # predictedProbs

    #first build a dictionary mapping labels to class names
    label2ClassName = {}
    for label,className in zip(labels,namesClasses):
        if not label in label2ClassName:
            label2ClassName[label] = className
    # then use the clf.classes_ attribute which contains the label values as they are ordered
    # in each row of the returned probability matrix to build the classNameSet
    classNameSet = []
    for label in clf.classes_:
        classNameSet.append(label2ClassName[label])

    return predictedProbs,classNameSet