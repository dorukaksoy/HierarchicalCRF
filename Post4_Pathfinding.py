# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# Post3_SegmentBasedCRF.py
# --- Description --
# For each viable pair, traces path between the source and the sink in the viable 
# pair using a cost-based A* pathfinding algorithm. This script continually calls 'TracePaths.py' 
# to obtain the traced paths.
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 06/27/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post4_Pathfinding.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
import joblib
from tqdm import tqdm

# Custom Libraries
import PostProcessingLib as pplib
import TracePaths as tps
# %% Methods
def non_padded_coords_from_dict(coords_dict):
    # Obtain the coordinates of the isolated point for the current pair in the padded image (separately for source and sinks)
    keys = list(coords_dict.keys())  # Convert dict_keys to a list
    first_tuple = keys[0]  # Get the first tuple
    return (first_tuple[0] - field_size // 2 , first_tuple[1] - field_size // 2 )  # Subtract field_size_half from both values of the tuple

def coords_from_dict(coords_dict):
    # Obtain the coordinates of the isolated point for the current pair in the padded image (separately for source and sinks)
    first_tuple = list(coords_dict.keys())[0]  # Get the first tuple
    return (first_tuple[0], first_tuple[1])
    
def pathfinding(viable_pairs, classified_post3, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments, tagged_iso_points, iso_points, title, std_dev=2, low_cost=10, high_cost=20, field_size=128, DEBUGGING=False):
    path_count = 0
    
    img_shape = classified_post3.shape
    # Create traced_paths array and add padding (to handle out-of-bounds fields)
    tagged_traced_paths = np.pad(np.zeros(img_shape, dtype=bool), field_size // 2, mode='constant')
    
    for pair_ind, curr_pair in enumerate(tqdm(viable_pairs if not DEBUGGING else viable_pairs[:1])):
            
        # Trace paths
        traced_path, path_coords = tps.trace_path(curr_pair, std_dev, low_cost, high_cost, field_size)
        tagged_traced_paths[path_coords[0][0]:path_coords[0][1],path_coords[1][0]:path_coords[1][1]] += traced_path
        
        # Tag iso_points as false (normally a part of the traced path)
        
        tagged_traced_paths[coords_from_dict(curr_pair['source'])] = False
        tagged_traced_paths[coords_from_dict(curr_pair['sinks'])] = False
        
        path_count += 1
    
    print(path_count,'paths are created.')
    
    # Remove padding from the traced_paths array
    tagged_traced_paths = pplib.remove_padding(tagged_traced_paths, field_size // 2)
    
    # Reconstruct the image with the 'on' broken segments and corresponding isolated points
    tagged_post4 = np.zeros_like(labeled_complete_segments, dtype=np.uint8)
    tagged_post4 = labeled_complete_segments + tagged_triple_junctions + labeled_broken_segments + tagged_iso_points + tagged_traced_paths
    tagged_post4 = np.clip(tagged_post4, 0, 1)
    
    return tagged_post4
# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # Load data
    viable_pairs, classified_post3, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments, tagged_iso_points, iso_points = joblib.load('./Post3.pkl')
    true_viable_pairs, true_classified_post3, true_labeled_complete_segments, true_tagged_triple_junctions, true_labeled_broken_segments, true_tagged_iso_points, true_iso_points = joblib.load('./Post3_true.pkl')
    
    # HYPERPARAMETERS
    window_size = 10 # Set the sliding window size
    field_size = 128 # The size of the fields around isolated points
    std_dev = 2 # Standard deviation of the exact Gaussian distribution
    low_cost = 10 # Lowest cost to assign when generating the base pattern
    high_cost = 20 # Highest cost to assign when generating the base pattern
    
    # Trace the path between the two points in each viable pair using the pathfinding algorithm
    tagged_post4 = pathfinding(viable_pairs, classified_post3,labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments, tagged_iso_points, iso_points, title=['Optimal with Connected Paths'], std_dev=std_dev, low_cost=low_cost, high_cost=high_cost, field_size=field_size )
    true_tagged_post4 = pathfinding(true_viable_pairs, true_classified_post3,labeled_complete_segments, tagged_triple_junctions, true_labeled_broken_segments, true_tagged_iso_points, true_iso_points, title=['FSGT with Connected Paths'], std_dev=std_dev, low_cost=low_cost, high_cost=high_cost, field_size=field_size)

    # # Save the arrays for next script
    joblib.dump((tagged_post4), './Post4.pkl') 
    joblib.dump((true_tagged_post4), './Post4_true.pkl') 