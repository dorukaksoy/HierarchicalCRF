# ---------------------------------------------------------------------------
# --- Prerequisite(s) --
# Post3_SegmentBasedCRF.py
# --- Description --
# Accepts one field, generates an exact 2D Gaussian distribution as the base 
# pattern, and then orients the base pattern according to source and 
# sink angles. From these two patterns, cost maps are created, which are 
# provided to the astar function to obtain the lowest-cost shortest path. 
# Then, the traced path, along with its absolute coordinates in the predicted mask, 
# are returned to the "Post4_Pathfinding.py".
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 06/27/2023
# ------------------
# Version: Python 3.8
# Execution: python3 TracePaths.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
from math import dist
import joblib
from scipy.ndimage import rotate, zoom

# Custom Libraries
import AstarLib as astarlib
import PostProcessingLib as pplib
# %% Task 1 - Obtain Base Patterns
# Generate base pattern
def generate_2d_gaussian(size, std_dev, low_cost, high_cost, field_size):
    sample_size = field_size*2
    # Generate x and y values over large range
    x = np.linspace(-3*std_dev, 3*std_dev, sample_size)
    y = np.linspace(-3*std_dev, 3*std_dev, sample_size)
    
    # Generate 2D grid of x and y values
    x, y = np.meshgrid(x, y)

    # Generate Gaussian function values over the grid
    d = np.sqrt(x*x+y*y)
    g = np.exp(-(d**2 / (2.0*std_dev**2)))
    
    # Flip the Gaussian distribution
    g = np.max(g) - g
    
    # Normalize the distribution to the range [low_cost, high_cost]
    g = g / np.max(g)
    g = g * (high_cost - low_cost) + low_cost
    
    # Downsample to the desired size using zoom function
    g_downsampled = zoom(g, (size/sample_size, size/sample_size))

    # Discretize the cost values
    num_bins = size // 2 + 1
    bins = np.linspace(low_cost, high_cost, num_bins)
    g_downsampled = np.digitize(g_downsampled, bins)

    return g_downsampled

def generate_base_pattern(source_coord, sink_coord, field_size, std_dev, low_cost, high_cost):
    # Variables to generate the base pattern
    # Find the Eucledian distance between the source and the sink
    eucl_dist = dist(source_coord,sink_coord)
    
    pad_size = int(field_size // 2)
    # Define a buffer
    buffer_size = pad_size # This is necessary because when we rotate the base pattern later, there needs to be extra so that we can trim 
    
    # Round up to the next odd number and add buffers to obtain the pattern size
    pattern_size = int(np.ceil(eucl_dist) // 2 * 2 + 1) + buffer_size * 2  

    base_pattern = (generate_2d_gaussian(pattern_size, std_dev, low_cost, high_cost, field_size)*10)
    
    # Increase the cost of LHS (to obtain an preferred direction that is on the +x axis)
    LHS_cost = base_pattern.copy()/4
    LHS_cost[:,pattern_size//2:] = 0
    base_pattern = base_pattern + LHS_cost
    
    # In addition, decrease the RHS cost
    RHS_cost = base_pattern.copy()/4
    RHS_cost[:,:pattern_size//2+1] = 0
    base_pattern = base_pattern - RHS_cost
    
    return base_pattern, buffer_size

# %% Task 2 - Orient Base Patterns
def orient_pattern(base_pattern, rotation_angle, pattern_size, buffer_size):
    
    # Use a rotation based scheme to handle the angle
    # Rotate it according to the source angle
    rotated_pattern = rotate(base_pattern, rotation_angle, reshape=False)
    
    # Trim the buffer
    rotated_pattern = rotated_pattern[buffer_size:(pattern_size - buffer_size),buffer_size:(pattern_size - buffer_size)]

    return rotated_pattern

# %% Task 3 - Generate the cost map
def generate_cost_map(field, base_pattern, source_rel, source_angle, sink_rel, sink_angle, pattern_size, buffer_size, low_cost, high_cost):
    # Orient the base pattern the match the source and sink angles
    source_pattern = orient_pattern(base_pattern, source_angle, pattern_size, buffer_size)
    sink_pattern = orient_pattern(base_pattern, sink_angle, pattern_size, buffer_size)
    
    # Initialize cost map to match your field shape
    cost_map = np.ones(field.shape,dtype=np.int32)
    
    # When applying the sink and source patterns, we need to remove the buffers
    buffer_lo = (pattern_size - 2 * buffer_size)//2
    buffer_hi = (pattern_size - 2 * buffer_size)//2 + 1
    
    # To handle out-of-bounds exceptions
    pad2_size = int(np.ceil(pattern_size/2))
    
    # Now for the cost map generation with padding to handle out-of-bound indices.
    pad_cost_map = np.pad(cost_map, pad2_size)
    source_rel_padded = tuple(np.array(source_rel) + [pad2_size,pad2_size])
    sink_rel_padded =  tuple(np.array(sink_rel) + [pad2_size,pad2_size])
    
    # Define the boundaries of source and sink patterns
    source_x_range = (source_rel_padded[1]-buffer_lo, source_rel_padded[1]+buffer_hi)
    source_y_range = (source_rel_padded[0]-buffer_lo, source_rel_padded[0]+buffer_hi)
    
    sink_x_range = (sink_rel_padded[1]-buffer_lo, sink_rel_padded[1]+buffer_hi)
    sink_y_range = (sink_rel_padded[0]-buffer_lo, sink_rel_padded[0]+buffer_hi)
    
    # Calculate overlapping range in x and y directions
    overlap_x_range = (max(source_x_range[0], sink_x_range[0]), min(source_x_range[1], sink_x_range[1]))
    overlap_y_range = (max(source_y_range[0], sink_y_range[0]), min(source_y_range[1], sink_y_range[1]))
    
    # Check if there is an overlap and assign low_cost to the overlapping region
    if overlap_x_range[0] < overlap_x_range[1] and overlap_y_range[0] < overlap_y_range[1]:
   
        # Initially assign the patterns to the cost map separately
        source_cost_map = pad_cost_map.copy()
        sink_cost_map = pad_cost_map.copy()
    
        source_cost_map[source_rel_padded[0]-buffer_lo:source_rel_padded[0]+buffer_hi, source_rel_padded[1]-buffer_lo:source_rel_padded[1]+buffer_hi] = source_pattern
        sink_cost_map[sink_rel_padded[0]-buffer_lo:sink_rel_padded[0]+buffer_hi, sink_rel_padded[1]-buffer_lo:sink_rel_padded[1]+buffer_hi] = sink_pattern
        
        # Now create a mask where both source and sink have non-default values
        overlap_mask = np.logical_and(source_cost_map != 1, sink_cost_map != 1)
    
        # Assign the sink and source patterns as is
        pad_cost_map[source_rel_padded[0]-buffer_lo:source_rel_padded[0]+buffer_hi, source_rel_padded[1]-buffer_lo:source_rel_padded[1]+buffer_hi] = source_pattern
        pad_cost_map[sink_rel_padded[0]-buffer_lo:sink_rel_padded[0]+buffer_hi, sink_rel_padded[1]-buffer_lo:sink_rel_padded[1]+buffer_hi] = sink_pattern
    
        # Then assign low_cost to the overlapping region
        pad_cost_map[overlap_mask] = low_cost
    
    else:
        # Use the padded cost map and patterns
        pad_cost_map[source_rel_padded[0]-buffer_lo:source_rel_padded[0]+buffer_hi, source_rel_padded[1]-buffer_lo:source_rel_padded[1]+buffer_hi] = source_pattern
        pad_cost_map[sink_rel_padded[0]-buffer_lo:sink_rel_padded[0]+buffer_hi, sink_rel_padded[1]-buffer_lo:sink_rel_padded[1]+buffer_hi] = sink_pattern
    
    # Now remove the padding to return to the original size.
    cost_map = pad_cost_map[pad2_size:-pad2_size, pad2_size:-pad2_size]
        
    # Fill the remaining pixels in the cost map with maximum cost in both patterns, or the high_cost variable if the former can't be assigned 
    try:
        cost_map[cost_map == 1] = max(np.partition(source_pattern.flatten(), -2)[-2],np.partition(sink_pattern.flatten(), -2)[-2])
    except:
        cost_map[cost_map == 1] = high_cost
    
    # Make the wall cost 20% higher than the highest cost for visualization purposes
    wall_cost = np.max(cost_map) * 1.2
    
    # Put back the foreground pixels (walls) as 
    cost_map[np.where(field==1)] = wall_cost
    
    # Assign wall cost to source and sink points
    for coord in [source_rel, sink_rel]:
        cost_map[coord] = wall_cost
        
    # plot_multiple_images_single_row([field,cost_map],titles=['Field','Cost Map'],cmap='RdYlGn_r')
    return cost_map

# %% Task 4 - Cost-based Pathfinding
def astar(field, cost_map, source_coord, sink_coord):
    # Create a square weighted grid that has the same shape as a field
    diagram = astarlib.GridWithWeights(field.shape[0],field.shape[1])
    
    # Assing walls (non-passable grain boundary pixels)
    diagram.walls = astarlib.get_foreground_coords(field)
    
    # Assign weights according to the cost map calculated above
    diagram.weights = astarlib.array_to_dict(cost_map)
    
    # Start from the source and move to the goal
    start, goal = source_coord, sink_coord
    
    # Keep track of the path and the associated cost using the A* algorithm
    came_from, cost_so_far = astarlib.a_star_search(diagram, start, goal)
    
    # Reconstruct the path by backtracking the lowest-cost shortest path
    path = astarlib.reconstruct_path(came_from, start=start, goal=goal)

    # Mark the traced path
    traced_path = np.zeros(field.shape, dtype=bool)
    for point in path:
        traced_path[point] = True
        
    return traced_path

# Obtain the absolute coordinates of the path based on the source absolute points
def get_path_abs_coords(source_abs, field_size):
    pad_size = int(field_size // 2)
    
    path_y_min, path_y_max = source_abs[0]-pad_size, source_abs[0] + pad_size
    path_x_min, path_x_max = source_abs[1]-pad_size, source_abs[1] + pad_size
    
    path_coords = [(path_y_min, path_y_max),(path_x_min, path_x_max)]
    
    return path_coords
    
def trace_path(curr_pair, std_dev, low_cost, high_cost, field_size=64):
    
    # Field properties
    field = curr_pair['field']
    source_abs = list(curr_pair['source'].keys())[0] # padded value
    sinks_abs = list(curr_pair['sinks'].keys())[0] # padded value
    source_rel = (field_size // 2 , field_size // 2)
    sink_rel = pplib.sink_abs_to_rel(sinks_abs, source_abs, source_rel)
    source_angle = list(curr_pair['source'].values())[0]
    sink_angle = list(curr_pair['sinks'].values())[0]
    
    base_pattern, buffer_size = generate_base_pattern(source_rel, sink_rel, field_size, std_dev, low_cost, high_cost)
    pattern_size = int(np.shape(base_pattern)[0])
    
    cost_map = generate_cost_map(field, base_pattern, source_rel, source_angle, sink_rel, sink_angle, pattern_size, buffer_size, low_cost, high_cost)
    traced_path = astar(field, cost_map, source_rel, sink_rel)
    path_coords = get_path_abs_coords(source_abs, field_size)

    return traced_path, path_coords

# %% The following operations are not executed when this script is imported as a module
if __name__ == "__main__":   
    # Load data
    viable_pairs, _, _, _, _, _, _ = joblib.load('./Post3.pkl')
    
    # select one pair for analysis
    curr_pair = viable_pairs[0]
    
    # Define variables 
    field_size = 64
    std_dev = 2 # Standard deviation of the exact Gaussian distribution
    low_cost = 10 # Lowest cost to assign when generating the base pattern
    high_cost = 20 # Highest cost to assign when generating the base pattern

    traced_path, path_coords = trace_path(curr_pair, std_dev, low_cost, high_cost, field_size)