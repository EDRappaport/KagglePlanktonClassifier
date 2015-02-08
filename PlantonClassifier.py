import glob
import os
from skimage.io import imread
import numpy as np

from Featurizer import FeaturizeImage
from Classifier import Classify
from MetricsEvaluation import multiclass_log_loss

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

# List of string of class names
namesClasses = list()
files = []

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_features = imageSize + 1 # for our ratio
# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
features = np.zeros((numberOfImages, num_features), dtype=float)
# y is the numeric class label
labels = np.zeros((numberOfImages))

print("Reading Files")
i = 0
label = 0
for folder in directory_names:
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)

            featureVector = FeaturizeImage(image)
            features[i,:] = (featureVector)
            labels[i] = label
            i += 1

             # report progress for each 5% done
            report = [int((j+1)*numberOfImages/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / numberOfImages), "% done"
    label += 1


predicted_labels = Classify(features, labels, namesClasses)

logLoss = multiclass_log_loss(labels, predicted_labels)

print("Final log loss: " + logLoss)