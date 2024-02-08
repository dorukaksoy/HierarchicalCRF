# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# Post2_Classifier.py
# --- Description --
# Apply segment-based conditional random fields to the image
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 09/01/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post3_SegmentBasedCRF.py
# ---------------------------------------------------------------------------
import numpy as np
import sys
import networkx as nx
from scipy.spatial import distance
import joblib
from tqdm import tqdm

# Custom Libraries
import CRFLib as crf
import PostProcessingLib as pplib

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
def SegmentBasedCRF(tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2, HITL=False):
    # Use a graph to find all possible paths within a distance
    # Construct the graph 
    print('Constructing graph of all possible paths...')
    G = nx.Graph()
    for key1 in l_junctions:
        for key2 in l_junctions:
            if key1 != key2:
                cost = crf.calculate_cost(key1, key2, l_junctions, classified_post2, max_distance=max_distance)
                if cost != float('inf'):
                    G.add_edge(key1, key2, weight=cost)

    # All possible paths are the edges of the graph
    all_possible_paths = []
    for edge in G.edges(data=True):
        all_possible_paths.append([edge[0], edge[1]])
    
    print('Graph is constructed.')
    # %% Obtain the completion segments 
    print('Obtaining completion segments...')
    completion_segments = {}
    segment_id = 0

    for path_ind, path in enumerate(all_possible_paths):
        start, end = path
        completion_segments[segment_id] = crf.line(start, end) 
        segment_id += 1
    
    # Find the maximum key in broken_segments
    max_key = max(broken_segments.keys())
    
    # Adjust the keys of completion_segments and merge to obtain all segments
    all_segments = {**broken_segments, **{k + max_key + 1: v for k, v in completion_segments.items()}}    
    
    print('Completion segments are obtained.')

    if HITL:
        print("Saving 'all_viable_paths.png' for HITL approach.")
        # Define the segments and their colors
        segments_info = {
            'line': {
                'segments': broken_segments,
                'color': 'red'
            },
            'completion': {
                'segments': completion_segments,
                'color': 'yellow'
            }
        }
        
        pplot.plot_multiple_segments_on_image(classified_post2, segments_info, font_color='cyan',font_size=1, line_width=1/8, title="All Viable Segments",crop_x=None, crop_y=None)
        
        DPI = 1000 # High DPI is needed for readability
        # Type (pdf, png, jpg, svg etc.)
        FILETYPE = 'png'
        plt.savefig('./all_viable_paths.' + FILETYPE, dpi=DPI, transparent=False)
        print("'all_viable_paths.png' is saved.")
    
    else:
        # Define the segments and their colors
        segments_info = {
            'broken': {
                'segments': broken_segments,
                'color': 'red'
            },
            'completion': {
                'segments': completion_segments,
                'color': 'yellow'
            }
        }

        pplot.plot_multiple_segments_on_image(classified_post2, segments_info, font_color='cyan',font_size=20, line_width=1, title="All Viable Segments",crop_x=None, crop_y=None)
                
    if not VISUALIZATIONS: plt.close()

    # %% Find all cliques
    '''
    Pair Clique, C_P​: We'll check if any completion segment touches a broken segment. If they touch, we'll add them to the pair clique.
    Completion Clique, C_C​: For every completion segment, we'll find its two neighboring broken segments and form a completion clique.
    Broken Clique, C_B​: For each endpoint of a broken segment, we'll find all segments connecting to it and form a broken clique.
    '''
    print('Assigning cliques...')
    # Initialize cliques
    C_P = []
    C_C = []
    C_B = {}
    
    # 1. Pair Clique C_P  
    for b_key, b_segment in broken_segments.items():
        for c_key, c_segment in completion_segments.items():
            if c_segment[0] in b_segment:            
                C_P.append((b_key, c_key))
            elif c_segment[-1] in b_segment:
                C_P.append((b_key, c_key))

    # 2. Completion Clique, C_C
    '''
    if there are two b_keys associated with the same c_key in the C_P list, append to C_C list 
    '''
    c_key_to_b_keys = {}
    for b_key, c_key in C_P:
        if c_key not in c_key_to_b_keys:
            c_key_to_b_keys[c_key] = []
        c_key_to_b_keys[c_key].append(b_key)
        
    # Iterate through the completion_segments by its keys
    for c_key, c_segment in completion_segments.items():
        if c_key in c_key_to_b_keys and len(c_key_to_b_keys[c_key]) == 2:
            b_key_1, b_key_2 = c_key_to_b_keys[c_key]
            C_C.append((c_key, b_key_1, b_key_2))
  
    # 3. Broken Clique, C_B
    '''
    Add any completion segment that coincides with either endpoint of the broken segment 
    Use the pair list to find all neighboring completion segments for a broken segment
    Then, assign to the C_B list as (b_key, c_key1, c_key2, ...)
    '''
    from collections import defaultdict
    # Create a defaultdict
    d = defaultdict(list)
    
    # Populate the defaultdict
    for j, i in C_P:
        d[j].append(i)
    
    # Convert defaultdict to desired format
    C_B = [(k, *v) for k, v in d.items()]
    
    # Remove the entries with fewer than three values (means that it is a pair)
    C_B = [t for t in C_B if len(t) >= 3]
    print("All cliques are assigned.")
 
    # %% Best Weights
    # These are the best weights found by the differential evolution algorithm (to replicate results in the article)
    w_I = np.array([5.00083943, -2.21490751, -0.15143674, -0.23543067,  7.65178626, 9.55137619, 1.39275913])
    w_M = np.array([-7.7278679, 1.36503795])
    omega_u = 8.719406665861571
    omega_I = 0.6798096816392318
    omega_M = -6.369390923932419
    alpha_i = -7.443746501214765 # For definitions of these, please see Ref [53]
    beta_i = 4.158424912526774 # For definitions of these, please see Ref [53]
    # %% Image features
    print('Calculating image features...')
    # Initialize features
    k_p = np.full(len(broken_segments),1) # Unary potential features (the output of a classifier, from Post2)
    k_I = np.zeros((len(completion_segments),7)) # Interface potential features
    k_M = np.zeros((len(completion_segments),2)) # Complexity potential features
    
    # ----- Interface potential features -----
    for clique in tqdm(C_C):
        k_I[clique[0], :] = crf.image_features(clique, broken_segments, completion_segments, l_junctions, classified_post2, alpha_i, beta_i)
                
    # ----- Complexity potential features -----
    for clique in C_C: 
        c_segment = completion_segments[clique[0]]
        b_segment_1 = broken_segments[clique[1]]
        b_segment_2 = broken_segments[clique[2]][::-1]
        
        c_len = distance.euclidean(c_segment[0], c_segment[-1]) # Length of the completion segment
        b1_len = distance.euclidean(b_segment_1[0], b_segment_1[-1]) # Length of the first broken segment
        b2_len = distance.euclidean(b_segment_2[0], b_segment_2[-1]) # Length of the second broken segment
        
        k_M[clique[0], 0] = c_len + b1_len + b2_len # Total length of the completion clique
        
    # Compute the effective length for each completion segment
    k_M[:,0] = np.array([distance.euclidean(segment[0], segment[-1]) for _, segment in completion_segments.items()])
    
    # Compute the angle compatibility
    for ind, (_, segment) in enumerate(completion_segments.items()):
        point_1, point_2 = segment[0], segment[-1]
        k_M[ind,1] = pplib.angle_cost(point_1, point_2, l_junctions)
    
    print("Image features are calculated.")    
    # %% Solve for Optimal Labels
    print('Solving CRF for optimal labels')
    from datetime import datetime
    crfStartTime = datetime.now()
    optimal_labels = crf.MILP(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M, C_P, C_B, all_segments)
    print(f'CRF solved in {datetime.now()-crfStartTime}.')
    # %% Connect near edge broken segments
    print('Connecting near edge broken segments...')
    # Define the near_edge_dist
    near_edge_dist = 10  # HYPERPARAMETER?

    # Get the shape of the image
    image_shape = classified_post2.shape

    # Obtain the near edge broken segments and edge l-junctions to connect them to
    labeled_complete_segments = pplib.connect_near_edge_broken_segments(l_junctions, broken_segments, labeled_complete_segments, image_shape, near_edge_dist=near_edge_dist)
    print('Near edge broken segments are connected.')
    # %% Plot the optimal solution (this is rotated 90 degrees CW in the paper)
    fig, ax = plt.subplots(figsize=(14, 14))
    labeled_elements = np.where(classified_post2 > 1, 1, classified_post2)
    from matplotlib.colors import ListedColormap
    ax.imshow(labeled_elements, cmap=ListedColormap(['black', 'white']), interpolation="lanczos")
    ax.set_title("CRF Predicted Label Configuration")
    plt.axis('off')    
    on_broken_idxs = []
    on_completion_idxs = []
    for ind, label in enumerate(optimal_labels):
        if label == 1:
            if ind < len(broken_segments): # On broken segments
                if HITL: pplot.plot_segment(all_segments[ind], ax, 'red', 'line', index=ind, font_size=1, line_width=1/8, font_color='cyan')
                else: pplot.plot_segment(all_segments[ind], ax, 'red', 'broken', index=ind, font_size=20, line_width=1, font_color='red')
                on_broken_idxs.append(ind)
            else: # On completion segments
                if HITL: pplot.plot_segment(all_segments[ind], ax, 'yellow', 'completion', index=(ind-len(broken_segments)), font_size=1, line_width=1/8, font_color='cyan')
                else: pplot.plot_segment(all_segments[ind], ax, 'yellow', 'completion', index=(ind-len(broken_segments)), font_size=20, line_width=1, font_color='yellow') 
                on_completion_idxs.append(ind-len(broken_segments))

    if HITL:
        print("Saving 'crf_predicted_paths.png' for HITL approach.")
        plt.savefig('./crf_predicted_paths.' + FILETYPE, dpi=DPI, transparent=False) # Save the plot to help with HITL approach   
        print("'crf_predicted_paths.png' is saved.")

    if not VISUALIZATIONS: plt.close()

    # %% Save the arrays for next script
    joblib.dump((on_broken_idxs, on_completion_idxs, broken_segments, broken_segment_labels, completion_segments, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments, tagged_l_junctions, l_junctions), './Post3.pkl')    
            
# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # Variables
    max_distance = 64 # HYPERPARAMETER
    
    # Assign HITL variable to use human-in-the-loop approach, which saves two images (all_viable_segments.png and crf_predicted_paths.png) so that a human can decide which indices of segments to add or remove
    # Check if there is at least one command-line argument (excluding the script name)
    # If there is no argument provided, set HITL to True
    if len(sys.argv) < 2:
        HITL = False
    else:
        # If an argument is provided, interpret it for HITL
        # This example simply checks if the first argument is the string 'True'
        HITL = sys.argv[1].lower() == 'true'

    # Load data
    tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2 = joblib.load('./Post2.pkl')
    
    # Apply segment based CRF
    SegmentBasedCRF(tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2, HITL=HITL)