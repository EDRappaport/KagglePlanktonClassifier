import glob
import os
from skimage.io import imread
import numpy as np

from Featurizer import FeaturizeImage
from Classifier import Classify
from MetricsEvaluation import multiclass_log_loss

def loadTrainingData():
    # get the class names from the directory structure
    directory_names = list(set(glob.glob(os.path.join("competition_data", "train", "*"))).difference(
        set(glob.glob(os.path.join("competition_data", "train", "*.*")))))
    directory_names.sort()
    numberOfImages = 0
    for folder in directory_names:
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                 # Only read in the images
                if fileName[-4:] != ".jpg":
                  continue
                numberOfImages += 1

    
    #load the data
    print("Reading Files")
    i = 0
    curLabel = 0
    features = list()
    labels = list()
    label2ClassName = []
    for folder in directory_names:
        currentClass = folder.split(os.sep)[-1]
        label2ClassName.append(currentClass)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
                # Read in the images and create the features
                nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                image = imread(nameFileImage, as_grey=True)
                #files.append(nameFileImage) DELETE THIS

                featureVector = FeaturizeImage(image)
                features.append(featureVector)
                labels.append(curLabel)
                i += 1

                 # report progress for each 5% done
                report = [int((j+1)*numberOfImages/20.) for j in range(20)]
                if i in report: print np.ceil(i *100.0 / numberOfImages), "% done"
        curLabel += 1
    return features,labels,label2ClassName

def loadTestData():
    #count total number of test examples
    numberOfImages = 0
    testFolder = os.path.join("competition_data", "test")
    for fileNameDir in os.walk(testFolder):
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberOfImages += 1

    #build the features_test matrix
    testFileNames = [] 
    print("Reading Test Files")
    i = 0
    features_test = list()
    for fileNameDir in os.walk(testFolder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue
            testFileNames.append(fileName)
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName) 
            image = imread(nameFileImage, as_grey=True)
            
            #featurize the image
            featureVector = FeaturizeImage(image)
            features_test.append(featureVector)
            i += 1

             # report progress for each 5% done
            report = [int((j+1)*numberOfImages/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / numberOfImages), "% done"
    return features_test,testFileNames

def makeSubmission(testFileNames,classNameSet,predictedProbs):
    '''
       classNameSet: a list of class names corresponding to columns of the predictedProbs matrix
       testFileNames: a list of test file names, with entries corresponding to the rows in the
                      predictedProbs matrix
       predictedProbs: a matrix of N test examples by P classes, giving the class probabilities
                       for each test example

    '''

    with open('submission.csv','w') as f:
        header = 'image'+','.join(classNameSet)+'\n'
        f.write(header)
        for testFile,testProbs in zip(testFileNames,predictedProbs):
            probs = [str(p) for p in testProbs]
            line = testFile+','+','.join(probs)+'\n'
            f.write(line)

def _main_():
    
    features,labels,label2ClassName = loadTrainingData()

    features_test,testFileNames = loadTestData()

    predictedProbs,classNameSet,clf = Classify(features, labels, label2ClassName,features_test)

    makeSubmission(testFileNames,classNameSet,predictedProbs)

if __name__ == "__main__":
    _main_()