# ---------------------------------------------------------------------------
# --- Prerequisite(s) ---
# Post1_PixelBasedCRF.py
# --- Description ---
# Classify each pixel on the image as either triple junction, isolated point, or broken or complete segments
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 09/01/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post2_Classifier.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
from scipy.ndimage import convolve, label
import joblib
from tqdm import tqdm
# Custom Libraries

import PostProcessingLib as pplib

# Check if a display environment exists
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ: # If does not exist
    matplotlib.use('Agg')
    VISUALIZATIONS=False
else: # If exists
    VISUALIZATIONS=True # False if no visualization environment exists

import PostProcessingPlots as pplot
# %% Helper methods
# A helper function to find the neighboring foreground pixel in the same segment for an isolated point:
def find_neighbor_pixel(labeled_segments, y, x, label):
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if (dy != 0 or dx != 0) and 0 <= y+dy < labeled_segments.shape[0] and 0 <= x+dx < labeled_segments.shape[1]:
                if labeled_segments[y+dy, x+dx] == label:
                    return y+dy, x+dx
    return None

# A helper function to extract the broken segment for each isolated point
def extract_broken_segment(labeled_segments, y, x, label):
    segment_pixels = [(y, x)]
    labeled_segments[y, x] = 0  # Mark the starting pixel as visited
    while True: ## TODO: Implement exception with max iteration
        neighbor_pixel = find_neighbor_pixel(labeled_segments, segment_pixels[-1][0], segment_pixels[-1][1], label)
        if neighbor_pixel:
            segment_pixels.append(neighbor_pixel)
            labeled_segments[neighbor_pixel[0], neighbor_pixel[1]] = 0  # Mark the current pixel as visited
        else:
            break
    return segment_pixels

def populate_segment_dicts(labeled_segments, tagged_iso_points):
    """
    Populate a dictionary with segment labels as keys and ordered coordinates as values.
    
    Parameters:
    - labeled_segments: 2D numpy array where each segment is labeled with a unique integer.
    - tagged_iso_points: 2D boolean numpy array indicating the locations of isolated points.
    
    Returns:
    - segment_dict: Dictionary with segment labels as keys and ordered coordinates (from isolated point to triple junction) as values.
    """
    
    # Initialize an empty dictionary to store segment labels and their corresponding coordinates.
    broken_segments = {}
    broken_segment_labels = [] # Keep track of the labels separately
    # Iterate over each unique label in the labeled_segments array.
    for seg_label in np.unique(labeled_segments):
        
        # We ignore the background label (0).
        if seg_label != 0:
            
            # Find the coordinates (y, x) of the isolated point that belongs to the current segment.
            y, x = np.where(tagged_iso_points & (labeled_segments == seg_label))
            
            # Extract the ordered coordinates of the segment starting from the isolated point.
            # The segment is ordered from the isolated point to the triple junction.
            coords = extract_broken_segment(labeled_segments.copy(), y[0], x[0], seg_label)
            
            # Store the segment label and its ordered coordinates in the dictionary.
            broken_segments[seg_label] = coords
            
            broken_segment_labels.append(seg_label)
            
    return broken_segments, broken_segment_labels

# A function to compute the angle using a sliding window approach
def compute_angle_sliding_window(segment_pixels, window_size):
    angles = []
    for i in range(len(segment_pixels) - window_size + 1):
        pixel1 = segment_pixels[i]
        pixel2 = segment_pixels[i + window_size - 1]
        angle = pplib.calcAngleInRads(pixel1, pixel2)
        angles.append(angle)
    return np.mean(angles)
# %% Label each pixel on the image as either triple junction, isolated point, or broken or complete segments and obtain curving angles
def label_features(image, window_size = 5):
    # Define convolution kernel for immediate neighbors
    kernel_immediate_neighbors = np.array([[1, 1, 1],
                                           [1, 0, 1],
                                           [1, 1, 1]])
    
    # Define convolution kernel for second nearest neighbors
    kernel_second_nearest_neighbors = np.array([[1, 0, 1, 0, 1],
                                                [0, 1, 1, 1, 0],
                                                [1, 1, 0, 1, 1],
                                                [0, 1, 1, 1, 0],
                                                [1, 0, 1, 0, 1]])
    
    # Count immediate neighbors and second nearest neighbors for each cell
    immediate_neighbors_count = convolve(image, kernel_immediate_neighbors, mode='constant', cval=0)
    second_nearest_neighbors_count = convolve(image, kernel_second_nearest_neighbors, mode='constant', cval=0)
    
    # Identify and tag (mark as True) triple junctions based on the counts of neighbors
    tagged_triple_junctions = (immediate_neighbors_count >= 3) | (second_nearest_neighbors_count >= 5)
    tagged_triple_junctions = tagged_triple_junctions & (image == 1)
    
    # Identify and tag segments 
    tagged_segments = (image == 1) & ~tagged_triple_junctions & (immediate_neighbors_count == 2)
        
    # Identify iso_points and tag on image (pixels with only one neighbor)
    tagged_iso_points_including_edges = (image == 1) & ~tagged_triple_junctions & (immediate_neighbors_count == 1)

    # Method to remove isolated points at the edges
    def remove_edge_iso_points(tagged_iso_points_including_edges):
        # Create a mask for the edges
        edge_mask = np.zeros_like(tagged_iso_points_including_edges, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
    
        # Remove isolated points at the edges by applying the inverted edge_mask
        tagged_iso_points = tagged_iso_points_including_edges & ~edge_mask
        return tagged_iso_points

    # Remove isolated points at the edges
    tagged_iso_points = remove_edge_iso_points(tagged_iso_points_including_edges)
    
    # For full connectivity, use the following structure
    conn_str = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
    
    # Label the segments (count iso_points as part of the segments too)
    labeled_segments, _ = label(np.logical_or(tagged_segments, tagged_iso_points), structure=conn_str)
    
    # Find the labels that belong to broken segments
    broken_segments_coords = [(y, x) for y, x in zip(*np.where(tagged_iso_points))] # Broken segments should include isolated points
    non_unique_broken_segment_labels = []
    
    # Check all the unique labels in all segments (except 0)
    print("Labeling features...")
    for seg_label in tqdm(np.unique(labeled_segments)[1:]):
        labeled_segments_coords = [(y, x) for y, x in zip(*np.where(labeled_segments==seg_label))]
        for pixel in broken_segments_coords:
            if pixel in labeled_segments_coords:
                non_unique_broken_segment_labels.append(seg_label)

    def remove_unlabeled_elements(array_2d, labels, broken=True): 
        mask = np.isin(array_2d, labels, invert=broken) # 'broken' switch is used for broken_segments, False otherwise
        array_2d[mask] = 0
        return array_2d
    
    # Label the broken and complete segments
    labeled_broken_segments = remove_unlabeled_elements(labeled_segments.copy(), non_unique_broken_segment_labels, broken=True)
    labeled_complete_segments = remove_unlabeled_elements(labeled_segments.copy(), non_unique_broken_segment_labels, broken=False)
    
    # Extract broken segments
    broken_segments, broken_segment_labels = populate_segment_dicts(labeled_broken_segments, tagged_iso_points)
    
    # Find curving angles
    iso_points = {}
    for y, x in zip(*np.where(tagged_iso_points)):
        segment_label = labeled_broken_segments[y, x]
        if segment_label != 0:
            neighbor_pixel = find_neighbor_pixel(labeled_broken_segments, y, x, segment_label)
            if neighbor_pixel:
                angle = pplib.calcAngleInRads((y, x), neighbor_pixel)
                iso_points[(y, x)] = int(angle * 180 / np.pi) # assign these as degrees for readability

    # Find the curving angles for each isolated point in broken_segments. If a value is found change in the original dict.
    for y, x in zip(*np.where(tagged_iso_points)):
        segment_label = labeled_broken_segments[y, x]
        if segment_label != 0:
            broken_segment_pixels = extract_broken_segment(labeled_broken_segments.copy(), y, x, segment_label)
            if len(broken_segment_pixels) >= window_size:
                angle = compute_angle_sliding_window(broken_segment_pixels, window_size)
                iso_points[(y, x)] = int(angle * 180 / np.pi) # assign these as degrees for readability

    return (tagged_triple_junctions, tagged_iso_points, labeled_broken_segments, labeled_complete_segments, iso_points, broken_segments, broken_segment_labels)
# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # HYPERPARAMETERS
    # Set the sliding window size
    window_size = 10
    
    # Load data
    tagged_post1 = joblib.load('./Post1.pkl')

    # Label features
    (tagged_triple_junctions, tagged_iso_points, labeled_broken_segments, labeled_complete_segments, iso_points, broken_segments, broken_segment_labels) = label_features(tagged_post1, window_size)
    print("Features are labeled.")
    
    # Reset the indices of the broken_segments dict
    broken_segments = {i: v for i, v in enumerate(broken_segments.values())}
    
    # Classify all pixels on image for plotting
    classification_params = {
    "triple_junctions": tagged_triple_junctions, 
    "broken_segments": labeled_broken_segments,
    "iso_points": tagged_iso_points
    }
        
    classified_post2 = pplib.classify_pixels(tagged_post1, **classification_params)
    
    if VISUALIZATIONS:
        pplot.post_classified_image(classified_post2, iso_points)
    
    # Save the arrays for the next script
    joblib.dump((tagged_triple_junctions, tagged_iso_points, labeled_broken_segments, labeled_complete_segments, iso_points, broken_segments, broken_segment_labels, classified_post2), './Post2.pkl')