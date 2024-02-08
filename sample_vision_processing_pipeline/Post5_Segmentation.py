# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# Post4_Pathfinding.py
# --- Description --
# Apply watershed segmentation to segment grains. This procedure is applied 
# to FSGT, pred_before (predicted mask before post-processing), 
# and pred_after (predicted mask after post-processing).
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 06/27/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post5_Segmentation.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
import joblib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import dilation, square
from skimage.measure import regionprops
from PIL import Image
from skimage.morphology import skeletonize

# Check if a display environment exists
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ: # If does not exist
    matplotlib.use('Agg')
    VISUALIZATIONS=False
else: # If exists
    VISUALIZATIONS=True # False if no visualization environment exists
    
import matplotlib.pyplot as plt
import PostProcessingPlots as pplot
# %% Methods
def watershed_segmentation(image):
    # Invert the image, i.e., background becomes 1, foreground becomes 0
    inverted_image = 1 - image

    # Apply dilation operation to remove isolated background pixels
    dilated_image = dilation(inverted_image, square(2))

    # Compute Euclidean distance from every binary pixel
    # to the nearest zero pixel then find peaks
    distance = ndi.distance_transform_edt(dilated_image)
    coords = peak_local_max(distance, min_distance=20, labels=dilated_image)
    
    # Create a mask of the identified peaks
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    
    # Now we need to mark each of these points with a unique id 
    # and create the corresponding markers for watershed.
    markers, _ = ndi.label(mask)
    
    # Perform the watershed operation which should fill the regions enclosed by edges
    labels = watershed(-distance, markers, mask=dilated_image)
    
    # Dilate the regions
    labels_dilated = dilation(labels, square(3))
    
    # Merge regions that are separated by just background pixels
    labels_merged = ndi.label(labels_dilated * inverted_image)[0]
    
    # Reassign sequential labels to ensure all labels are unique
    labels_final, num_labels = ndi.label(labels_merged)
    
    # Use grey_erosion to eliminate single-pixel regions
    footprint = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=bool)
    labels_final = ndi.grey_erosion(labels_final, footprint=footprint)

    return labels_final

# Obtain the grain size based on the area of the labeled region
def grain_size(image, labels):
   # Get properties of each region
   props = regionprops(labels)

   # Area of the region i.e. number of pixels of the region scaled by pixel-area.
   label_sizes = {region.label: region.area for region in props}
   
   return label_sizes

def segment_image(image, image_type):
    
    # Segment images
    labels = watershed_segmentation(image)
    grain_sizes = grain_size(image, labels)

    # Calculate the average grain sizes
    average_size = np.average(list(grain_sizes.values()))

    return {
        'labels': labels,
        'grain_sizes': grain_sizes,
        'average_size': average_size,
    }  

def segmentation(pred_before, pred_after, image_features=None, image_ind = ''):

    # fsgt_seg = segment_image(fsgt,'fsgt')
    pred_before_seg = segment_image(pred_before,'pred_before')
    pred_after_seg = segment_image(pred_after,'pred_after')
    
    if image_features is not None:
        # image_features[image_ind]["fsgt"]["num_grains"] = np.max(np.unique(fsgt_seg['labels']))
        image_features[image_ind]["before"]["num_grains"] = np.max(np.unique(pred_before_seg['labels']))
        image_features[image_ind]["after"]["num_grains"] = np.max(np.unique(pred_after_seg['labels']))
        
        # image_features[image_ind]["fsgt"]["mean_grain_size"] = np.max(np.unique(fsgt_seg['average_size']))
        image_features[image_ind]["before"]["mean_grain_size"] = np.max(np.unique(pred_before_seg['average_size']))
        image_features[image_ind]["after"]["mean_grain_size"] = np.max(np.unique(pred_after_seg['average_size']))

    # Save the arrays for next script
    joblib.dump((pred_before, pred_before_seg, pred_after, pred_after_seg), './Post5.pkl')
    
    return pred_before_seg, pred_after_seg, image_features

# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # Load and skeletonize the image before post-processing
    pred = np.array(Image.open('./pred.png').convert('L'),dtype='bool').astype(np.uint8)*255
    pred_before = skeletonize(pred.astype(bool)).astype(np.uint8) # Before post-processing
    
    # Load the image after post-processing and FSGT
    pred_after = joblib.load('./Post4.pkl')

    field_size = 128
    pad_size = field_size // 2
    
    print('Starting the watershed segmentation procedure...')
    pred_before_seg, pred_after_seg, image_features = segmentation(pred_before, pred_after)
    print('Segmentation is complete.')

    DPI = 500
    # Type (pdf, png, jpg, svg etc.)
    FILETYPE = 'png'
    
    # Plot figures
    pplot.plot_image_with_labels(pred_before, pred_before_seg['labels'], "Before Post-processing")
    plt.savefig('./before_segmentation.' + FILETYPE, dpi=DPI, transparent=False)
    
    pplot.plot_image_with_labels(pred_after, pred_after_seg['labels'], "After Post-processing")
    plt.savefig('./after_segmentation.' + FILETYPE, dpi=DPI, transparent=False)
    if not VISUALIZATIONS:
        plt.close()