# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# Machine vision model
# --- Description --
# Apply pixel-based conditional random fields to the image through Bilateral and Gaussian filters
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 09/01/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post1_PixelBasedCRF.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
# %% Methods
def PixelBasedCRF(img, pred):
    '''
    Cython-based Python wrapper for Philipp Krähenbühl's Fully-Connected 
    CRFs (DenseCRF) ( https://github.com/lucasb-eyer/pydensecrf )
    '''
    
    # Obtain the region that is the opposite of the predicted mask
    not_pred = cv2.bitwise_not(pred)
    not_pred = np.expand_dims(not_pred, axis=2)
    
    # Data preparation
    pred = np.expand_dims(pred, axis=2)
    img_softmax = np.concatenate([not_pred, pred], axis=2)
    img_softmax = img_softmax / 255.0

    # Two classes (O: GB, 1: not GB)
    n_classes = 2 
    features = img_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
    
    # Unary potential
    unary = unary_from_softmax(features)
    unary = np.ascontiguousarray(unary)
    
    # Apply DenseCRF
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)
    d.setUnaryEnergy(unary)
    
    d.addPairwiseGaussian(sxy=(5, 5), compat=10, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    d.addPairwiseBilateral(sxy=(10, 10), srgb=(100, 100, 100), rgbim=img.copy(),
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    
    tagged_post1 = np.array(res*255, dtype=np.uint8)
    
    # Perform skeletonization (Image should only contain 1s and 0s)
    tagged_post1_skeleton = skeletonize(tagged_post1.astype(bool).astype(np.uint8)).astype(np.uint8)

    # Save them for the next step
    joblib.dump(tagged_post1_skeleton, './Post1.pkl')
    
    return tagged_post1_skeleton
# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":
    # Prepare the images for the post-processing
    # Read in and convert the original image and predicted image from computer vision algorithm
    img = np.asarray(plt.imread('./img.tif'), dtype='uint8')
    pred = np.array(Image.open('./pred.png').convert('L'),dtype='bool').astype(np.uint8)*255
    
    # Apply bilateral and Gaussian filtering
    tagged_post1 = PixelBasedCRF(img, pred)