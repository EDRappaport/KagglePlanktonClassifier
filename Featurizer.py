from skimage.transform import resize
from skimage import measure
from skimage import morphology
import numpy as np

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_features = imageSize + 1 # for our ratio


# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imageThresh = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imDilated = morphology.dilation(imageThresh, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imDilated)
    label_list = imageThresh*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imageThresh)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

def getRegionPropFeatures(image):
    #image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imageThresh = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imDilated = morphology.dilation(imageThresh, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imDilated)
    label_list = imageThresh*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imageThresh)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

    if not maxregion is None:
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length


def FeaturizeImage(image):
    axisRatio = getMinorMajorRatio(image)
    image = resize(image, (maxPixel, maxPixel))

    # Store the rescaled image pixels and the axis ratio
    X = np.zeros(num_features, dtype=float)
    X[0:imageSize] = np.reshape(image, (1, imageSize))
    X[imageSize] = axisRatio

    return X
