from skimage.transform import resize
from skimage import measure
from skimage import morphology
import numpy as np


# find the largest nonzero region
def getLargestRegion(region_props, labeled_im, thresholded_im):
    regionmaxprop = None
    for regionprop in region_props:
        # check to see if the region is at least 50% nonzero
        if sum(thresholded_im[labeled_im == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    thresholded_im = np.where(image < np.mean(image),1.0,0.)  # ROI is dark relative to background ??? 

    #Dilate the image
    dilated_im = morphology.dilation(thresholded_im, np.ones((4,4)))

    # Create the labeled_image
    labeled_im = measure.label(dilated_im)
    labeled_im = thresholded_im*labeled_im #zeros out labels based on threshhold mask
    labeled_im = labeled_im.astype(int)

    region_props = measure.regionprops(labeled_im)
    maxregion = getLargestRegion(region_props, labeled_im, thresholded_im)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio


def FeaturizeImage(image):
    axisRatio = getMinorMajorRatio(image)

    # We'll rescale the images to be 25x25
    maxPixel = 25
    imageSize = maxPixel * maxPixel
    num_features = imageSize + 1 # for our ratio
    image = resize(image, (maxPixel, maxPixel))

    X = np.zeros(num_features, dtype=float)

    # Store the rescaled image pixels and the axis ratio
    X[0:imageSize] = np.reshape(image, (1,imageSize))
    X[imageSize] = axisRatio

    return X
