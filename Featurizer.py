from skimage.transform import resize
from skimage import measure
from skimage import morphology
import numpy as np

#############################################################################
# find the largest nonzero region
#############################################################################
def getLargestRegion(image):
    # Create the thresholded image to eliminate some of the background
    thresholded_im = np.where(image < np.mean(image),1.0,0.)  # ROI is dark relative to background ??? 
    # Dilate the image
    dilated_im = morphology.dilation(thresholded_im, np.ones((4,4)))
    # Create the labeled_image
    labeled_im = measure.label(dilated_im)
    labeled_im = thresholded_im*labeled_im #zeros out labels based on threshhold mask
    labeled_im = labeled_im.astype(int)

    region_props = measure.regionprops(labeled_im, image)
    
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

#############################################################################
# find a set of features for the given image
#############################################################################
def FeaturizeImage(image):
    maxpixel = 25
    imageSize = maxpixel*maxpixel

    #Find the largest region and use it to compute various features
    maxregion = getLargestRegion(image)
   
    # guard against cases where the segmentation fails by providing zeros    
    features = []
    #normalize area to the size of the image
    features.append(maxregion.area/image.size) if not maxregion is None else features.append(0.0)
    #is this the same as minor-major axis ratio? 
    features.append(maxregion.eccentricity) if not maxregion is None else features.append(0.0)
    #normalize by area of bounding box
    features.append(maxregion.equivalent_diameter/maxregion.convex_image.size) if not maxregion is None else features.append(0.0)
    #number of objects (= 1) subtracted by number of holes (8-connectivity).
    features.append(maxregion.euler_number) if not maxregion is None else features.append(0.0)
    #Ratio of pixels in the region to pixels in the total bounding box
    features.append(maxregion.extent) if not maxregion is None else features.append(0.0)
    #The two eigen values of the inertia tensor in decreasing order
    features.extend(maxregion.inertia_tensor_eigvals) if not maxregion is None else features.extend([0.0]*2)
    #mean intensity in the region normalized by std of the image
    features.append(maxregion.mean_intensity/image.std()) if not maxregion is None else features.append(0.0)
    #Hu moments (translation, scale and rotation invariant)
    features.extend(maxregion.moments_hu) if not maxregion is None else features.extend([0.0]*7)
    features.append(maxregion.perimeter/maxregion.equivalent_diameter) if not maxregion is None else features.append(0.0)
    #Ratio of pixels in the region to pixels of the convex hull image
    features.append(maxregion.solidity) if not maxregion is None else features.append(0.0)
    #Rescaled image
    features.extend(np.reshape( resize(image, (maxPixel, maxPixel) ), (1, imageSize) ) )
    
    return np.array(features)

#############################################################################
# UNUSED CODE
#############################################################################
def getMinorMajorRatio(image):
    # Create the thresholded image to eliminate some of the background
    thresholded_im = np.where(image < np.mean(image),1.0,0.)  # ROI is dark relative to background ??? 

    # Dilate the image
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
