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
# Execution: python3 Post1_PixelBasedCRF.py
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
from scipy.optimize import differential_evolution
import joblib
from tqdm import tqdm

# Custom Libraries
import CRFLib as crf
import PostProcessingLib as pplib
import PostProcessingPlots as pplot

def SegmentBasedCRF(tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2, LEARN_WEIGHTS = False):
    # Use a graph to find all possible paths within a distance
    # Construct the graph 
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
    
    # %% Obtain the completion segments 
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
    
    # crop_y needs to be inverse (2048,0) due to the convention (this is rotated 90 degrees CW in the paper)
    crop_x = (1749,2016)
    crop_y = (313,6)
    pplot.plot_multiple_segments_on_image(classified_post2, segments_info, line_width=1, title="All Viable Segments",crop_x=crop_x, crop_y=crop_y)
    
    # %% Fragmented Segmentation Ground Truth (FSGT)

    # Read FSGT active (on) segments from csvs created by human operator for performance evaluation
    true_on_broken_idxs = np.unique(np.loadtxt("./FSGT_broken.csv",delimiter=",", dtype=int))
    true_on_completion_idxs = np.unique(np.loadtxt("./FSGT_completion.csv",delimiter=",", dtype=int))

    # Initialize segments as off first
    true_broken_segment_labels = np.full(len(broken_segments),0)
    true_completion_segment_labels = np.full(len(completion_segments),0)

    # Turn on selected segments
    true_broken_segment_labels[true_on_broken_idxs] = 1
    true_completion_segment_labels[true_on_completion_idxs] = 1
    
    # Adjust the keys of completion_segments and merge to obtain all segments
    true_labels = np.concatenate([true_broken_segment_labels, true_completion_segment_labels])
    
    # Write data for figure
    joblib.dump((true_broken_segment_labels, true_completion_segment_labels), './Fig3b.pkl')
        
    # Plot the FSGT (this is rotated 90 degrees CW in the paper)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(14, 14))
    labeled_elements = np.where(classified_post2 > 1, 1, classified_post2)
    ax.imshow(labeled_elements, cmap=ListedColormap(['black', 'white']), interpolation="lanczos")
    ax.set_title("Fragmented Segmentation Ground Truth (FSGT)")
    for segment_ind, segment_label in enumerate(true_broken_segment_labels):
          if segment_label == 1:
            pplot.plot_segment(broken_segments[segment_ind], ax, color='red', segment_type='broken')    
    for segment_ind, segment_label in enumerate(true_completion_segment_labels):
        if segment_label == 1:
            pplot.plot_segment(completion_segments[segment_ind], ax, color='yellow')
    
    crop_x = (1749,2016)
    crop_y = (313,6)
    if crop_x is not None: ax.set_xlim(crop_x)
    if crop_y is not None: ax.set_ylim(crop_y)
    plt.axis('off')
    # %% Find all cliques
    '''
    Pair Clique, C_P​: We'll check if any completion segment touches a broken segment. If they touch, we'll add them to the pair clique.
    Completion Clique, C_C​: For every completion segment, we'll find its two neighboring broken segments and form a completion clique.
    Broken Clique, C_B​: For each endpoint of a broken segment, we'll find all segments connecting to it and form a broken clique.
    '''

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
    
    print("\nImage features are calculated.")    
    # %% Solve for Optimal Labels
    from datetime import datetime
    crfStartTime = datetime.now()
    optimal_labels = crf.MILP(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M, C_P, C_B, all_segments)
    print(f'CRF solved in {datetime.now()-crfStartTime}')

    # Plot the optimal solution (this is rotated 90 degrees CW in the paper)
    fig, ax = plt.subplots(figsize=(14, 14))
    labeled_elements = np.where(classified_post2 > 1, 1, classified_post2)
    ax.imshow(labeled_elements, cmap=ListedColormap(['black', 'white']), interpolation="lanczos")
    ax.set_title("CRF Predicted Label Configuration")
    plt.axis('off')    
    on_broken_idxs = []
    on_completion_idxs = []
    for ind, label in enumerate(optimal_labels):
        if label == 1:
            if ind < len(broken_segments): # On broken segments
                pplot.plot_segment(all_segments[ind], ax, 'red', 'broken')
                on_broken_idxs.append(ind)
            else: # On completion segments
                pplot.plot_segment(all_segments[ind], ax, 'yellow')  
                on_completion_idxs.append(ind-len(broken_segments))

    crop_x = (1749,2016)
    crop_y = (313,6)
    if crop_x is not None: ax.set_xlim(crop_x)
    if crop_y is not None: ax.set_ylim(crop_y)
    plt.axis('off')
    
    # %% Prediction Performance
    correctly_predicted = np.size(true_labels) - int(np.sum(np.abs(true_labels - optimal_labels)))
    segment_pred_accuracy = (np.size(true_labels) - int(np.sum(np.abs(true_labels - optimal_labels))))/np.size(true_labels) * 100
    
    print('Correctly predicted: {}/{} - Prediction accuracy: {:.2f}%'.format(correctly_predicted,np.size(true_labels),segment_pred_accuracy))

    # %% Connect near edge broken segments
    # Define the near_edge_dist
    near_edge_dist = 10  # HYPERPARAMETER?

    # Get the shape of the image
    image_shape = classified_post2.shape

    # Obtain the near edge broken segments and edge l-junctions to connect them to
    labeled_complete_segments = pplib.connect_near_edge_broken_segments(l_junctions, broken_segments, labeled_complete_segments, image_shape, near_edge_dist=near_edge_dist)
    # %% Reconstruct the image with the optimal labels
    classified_post3, l_junctions_post3, labeled_broken_segments_post3, tagged_l_junctions_post3 = pplib.match_labels(broken_segment_labels, on_broken_idxs, l_junctions, tagged_l_junctions, broken_segments, labeled_broken_segments, labeled_complete_segments, tagged_triple_junctions, title='Reconstructed Optimal Labels')
    true_classified_post3, true_l_junctions_post3, true_labeled_broken_segments_post3, true_tagged_l_junctions_post3 = pplib.match_labels(broken_segment_labels, true_on_broken_idxs, l_junctions, tagged_l_junctions, broken_segments, labeled_broken_segments, labeled_complete_segments, tagged_triple_junctions, title='Reconstructed Ground Truth')
        
    # %% Identify viable pairs
    viable_pairs = []
    
    for seg_ind in on_completion_idxs:
        viable_pair = pplib.extract_field(classified_post3, l_junctions_post3, completion_segments[seg_ind], field_size=max_distance*2)
        viable_pairs.append(viable_pair)
    # Save the arrays for next script
    joblib.dump((viable_pairs, classified_post3, labeled_complete_segments, tagged_triple_junctions, labeled_broken_segments_post3, tagged_l_junctions_post3, l_junctions_post3), './Post3.pkl')
    
    true_viable_pairs = []
    for seg_ind in true_on_completion_idxs:
        viable_pair = pplib.extract_field(true_classified_post3, true_l_junctions_post3, completion_segments[seg_ind], field_size=max_distance*2)
        true_viable_pairs.append(viable_pair)

    # Save the arrays for next script
    joblib.dump((true_viable_pairs, true_classified_post3, labeled_complete_segments, tagged_triple_junctions, true_labeled_broken_segments_post3, true_tagged_l_junctions_post3, true_l_junctions_post3), './Post3_true.pkl')
    # %% Learn actual weights (This is required once for each new problem)    
    if LEARN_WEIGHTS:
        max_iter = 1
        all_weights = []
        all_results = []
        i = 0
        diff_label = 1E6
        print("Start learning the optimal weights...")
        while (i < max_iter) and (diff_label != 0):
            # Define the bounds for the weights (7 for w_I, 2 for w_M and 1 each for omega_u, omega_I, omega_M, alpha_i, and beta_i)
            bounds = [(-10, 10)] * 7 + [(-10, 10)] * 2 + [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
        
            iterTime = datetime.now()
            # Use differential evolution to optimize the weights (workers: 1 for single core, -1 for all cores)
            result = differential_evolution(crf.fitness, bounds, args=(broken_segments, completion_segments, k_p, k_M, C_P, C_C, C_B, all_segments, true_labels, l_junctions, classified_post2), updating='deferred',disp=True, workers=-1)

            # Save the optimized weights
            all_weights.append(result.x)
            all_results.append(int(result.fun))
            i += 1
            print(i,"/",max_iter,":",diff_label,f' in {datetime.now()-iterTime}')
    
        # Extract the optimized weights
        best_weights = all_weights[np.argmin(all_results)]
        
        w_I_best = best_weights[:7]
        w_M_best = best_weights[7:9]
        omega_u_best = best_weights[9]
        omega_I_best = best_weights[10]
        omega_M_best = best_weights[11]
        alpha_i_best = best_weights[12]
        beta_i_best = best_weights[13]
        
        print("Best w_I_l:", w_I_best)
        print("Best w_M_l:", w_M_best)
        print("Best omega_u:", omega_u_best)
        print("Best omega_I:", omega_I_best)
        print("Best omega_M:", omega_M_best)
        print("Best alpha_i:", alpha_i_best)
        print("Best beta_i:", beta_i_best)        

        k_I_best = np.zeros((len(completion_segments),7)) # Interface potential features
        for clique in C_C: 
            k_I_best[clique[0], :] = crf.image_features(clique, broken_segments, completion_segments, l_junctions, classified_post2, alpha_i_best, beta_i_best)
            
        optimal_labels = crf.MILP(broken_segments, completion_segments, k_p, k_I_best, k_M, w_I_best, w_M_best, omega_u_best, omega_I_best, omega_M_best, C_P, C_B, all_segments)
        
        # Plot Labels with learned weights
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(classified_post2, cmap=plt.cm.gray)
        ax.set_title("Solution with Best Weights")
        
        for ind, label in enumerate(optimal_labels):
            if label == 1:
                if ind < len(broken_segments):
                    pplot.plot_segment(all_segments[ind], ax, 'red', 'broken') # broken segment in red
                else: 
                    pplot.plot_segment(all_segments[ind], ax, 'yellow')  # Completion segments in yellow    
    
# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # Variables
    max_distance = 64 # HYPERPARAMETER

    # Load data
    tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2 = joblib.load('./Post2.pkl')
    
    # Apply segment based CRF
    SegmentBasedCRF(tagged_triple_junctions, tagged_l_junctions, labeled_broken_segments, labeled_complete_segments, l_junctions, broken_segments, broken_segment_labels, classified_post2, LEARN_WEIGHTS=False)
