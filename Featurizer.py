from skimage.transform import resize
from skimage import measure
from skimage import morphology
import numpy as np

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_regionProp_features = 17
num_features = imageSize + num_regionProp_features 


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
    # Create the thresholded image to eliminate some of the background
    imageThresh = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imDilated = morphology.dilation(imageThresh, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imDilated, background=None)
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
    label_list = measure.label(imDilated, background=None)
    label_list = imageThresh*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list,image)
    maxregion = getLargestRegion(region_list, label_list, imageThresh)

    # guard against cases where the segmentation fails by providing zeros    
    features = []
    if not maxregion is None:
        features.append(maxregion.area/image.size) #normalize area to the size of the image
        features.append(maxregion.eccentricity) #is this the same as minor-major axis ratio?
        features.append(maxregion.equivalent_diameter/maxregion.convex_image.size) #normalize by area of bounding box
        features.append(maxregion.euler_number) #number of objects (= 1) subtracted by number of holes (8-connectivity).
        features.append(maxregion.extent) #Ratio of pixels in the region to pixels in the total bounding box
        features.extend(maxregion.inertia_tensor_eigvals) #The two eigen values of the inertia tensor in decreasing order
        features.append(maxregion.mean_intensity/image.std()) #mean intensity in the region normalized by std of the image
        features.extend(maxregion.moments_hu) #Hu moments (translation, scale and rotation invariant)
        features.append(maxregion.perimeter/maxregion.equivalent_diameter)
        features.append(maxregion.solidity) #Ratio of pixels in the region to pixels of the convex hull image
    else:
        features = [0.0]*num_regionProp_features

    return features


def FeaturizeImage(image):
    features = getRegionPropFeatures(image)
    image = resize(image, (maxPixel, maxPixel))

    # Store the rescaled image pixels and the axis ratio
    X = np.zeros(num_features, dtype=float)
    X[0:imageSize] = np.reshape(image, (1, imageSize))
    X[imageSize:] = features

    #X = np.array(features)
    return X
