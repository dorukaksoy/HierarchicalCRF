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
import sys
import numpy as np
import joblib
from tqdm import tqdm

# Custom Libraries
import PostProcessingLib as pplib
import TracePaths as tps

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
    # HYPERPARAMETERS
    window_size = 10 # Set the sliding window size
    field_size = 128 # The size of the fields around isolated points
    std_dev = 2 # Standard deviation of the exact Gaussian distribution
    low_cost = 10 # Lowest cost to assign when generating the base pattern
    high_cost = 20 # Highest cost to assign when generating the base pattern

    # Load data 
    (on_broken_idxs, on_completion_idxs, broken_segments, broken_segment_labels, completion_segments, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments_post3, tagged_l_junctions, l_junctions) = joblib.load('./Post3.pkl')
    
    # Assign HITL variable to use human-in-the-loop approach, which saves two images (all_viable_segments.png and crf_predicted_paths.png) so that a human can decide which indices of segments to add or remove
    # Check if there is at least one command-line argument (excluding the script name)
    # If there is no argument provided, set HITL to True
    if len(sys.argv) < 2:
        HITL = False
    else:
        # If an argument is provided, interpret it for HITL
        # This example simply checks if the first argument is the string 'True'
        HITL = sys.argv[1].lower() == 'true'
    
    if HITL:
        print('Continuing with the HITL approach...')
        # Human-in-the-loop (HITL) approach 
        '''
        After checking the 'all_viable_paths.png' and 'crf_predicted_paths.png' images, which have the indices of individual segments,
        if the user wants to add or remove individual segments four .csv files can be utilized: 
        'broken_to_add.csv', 'broken_to_remove'remove.csv','completion_to_add', and 'completion_to_remove'.
        !!! IMPORTANT !!!: These .csv files should be deleted after, otherwise these indices will be indcluded/discarded in each run, even for a different image.
        '''
        import pandas as pd
        import os
        
        def read_csv_to_list(csv_path):
            """Reads a CSV file and returns a list of values from the first column."""
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, header=None)
                except:
                    return None
                # Check if the DataFrame is empty
                if df.empty:
                    return []
                else:
                    # Convert the first column to a list
                    return list(df.to_numpy()[0])
            else:
                return None
        
        def apply_HITL(items_to_add, items_to_remove, segment_idxs):
            """Applies the Human-in-the-Loop approach to modify segment indices based on add/remove lists."""
            if items_to_add is not None:
                # Add items if they are not already present
                for item in items_to_add:
                    if item not in segment_idxs:
                        segment_idxs.append(item)
            
            if items_to_remove is not None:
                # Remove items if they exist in the list
                for item in items_to_remove:
                    if item in segment_idxs:
                        segment_idxs.remove(item)
            
            return segment_idxs
        
        # Paths to your CSV files
        csv_paths = {
            'broken_to_add': './broken_to_add.csv',
            'broken_to_remove': './broken_to_remove.csv',
            'completion_to_add': './completion_to_add.csv',
            'completion_to_remove': './completion_to_remove.csv'
        }
        
        # Read values from CSVs, or set to None if the file doesn't exist
        broken_to_add = read_csv_to_list(csv_paths['broken_to_add'])
        broken_to_remove = read_csv_to_list(csv_paths['broken_to_remove'])
        completion_to_add = read_csv_to_list(csv_paths['completion_to_add'])
        completion_to_remove = read_csv_to_list(csv_paths['completion_to_remove'])
        
        # Apply HITL approach only if there are items to add/remove
        on_broken_idxs = apply_HITL(broken_to_add, broken_to_remove, on_broken_idxs)
        on_completion_idxs = apply_HITL(completion_to_add, completion_to_remove, on_completion_idxs)
        
        print("HITL segments are added/removed.")
    
    print('Reconstructing the image...')
    # Reconstruct the image with the optimal labels
    classified_post3, iso_points, labeled_broken_segments, tagged_iso_points = pplib.match_labels(broken_segment_labels, on_broken_idxs, l_junctions, tagged_l_junctions, broken_segments, labeled_broken_segments_post3, labeled_complete_segments, tagged_triple_junctions, title='Reconstructed Optimal Labels')
    
    # Identify viable pairs
    print('Identifying viable pairs...')
    viable_pairs = []
    for seg_ind in on_completion_idxs:
        # Skip the current iteration if the segment index does not exist in completion_segments
        if seg_ind not in completion_segments:
            print("Warning: One activated segment index is not part of the completion segment set.")
            continue  # Move on to the next seg_ind without executing more code in the loop
    
        viable_pair = pplib.extract_field(classified_post3, iso_points, completion_segments[seg_ind], field_size=field_size)
        if viable_pair is not None:
            viable_pairs.append(viable_pair)
    print('Viable pairs are identified.')
    
    print('Employing pathfinding procedure...')
    # Trace the path between the two points in each viable pair using the pathfinding algorithm
    tagged_post4 = pathfinding(viable_pairs, classified_post3, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments, tagged_iso_points, iso_points, title=['Optimal with Connected Paths'], std_dev=std_dev, low_cost=low_cost, high_cost=high_cost, field_size=field_size )
    print('Paths between viable pairs are traced.')
    
    # %% Plot and save
    pplot.plot_multiple_images_single_row([tagged_post4],titles=['Traced Paths'])
    
    DPI = 500
    # Type (pdf, png, jpg, svg etc.)
    FILETYPE = 'png'
    plt.savefig('./traced_paths.' + FILETYPE, dpi=DPI, transparent=False)
    
    if not VISUALIZATIONS:
        plt.close()
    # Save the arrays for next script
    joblib.dump((tagged_post4), './Post4.pkl') 