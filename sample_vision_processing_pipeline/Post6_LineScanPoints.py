# ---------------------------------------------------------------------------
# --- Prerequisite(s) ---
# Post4_PixelBasedCRF.py
# --- Description ---
# Obtain the labels of the completion segments. Order points for each segment. 
# Then, obtain the midpoints (or points at certain fractions) of each completion segment, 
# find a number of points on a perpendicular line separated by certain number of pixels
# Outputs scan_points.csv for the subsequent analysis
# User can choose the fractions, number of scan points perpendicular to the points 
# chosen at the fractions, and the spacing between the points
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 09/01/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Post6_LineScanPoints.py
# ---------------------------------------------------------------------------
# %% Imports
import numpy as np
import joblib
import pandas as pd  # For saving the scan points to a CSV file

# Custom Libraries
import Post2_Classifier as post2

# Check if a display environment exists
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ: # If does not exist
    matplotlib.use('Agg')
    VISUALIZATIONS=False
else: # If exists
    VISUALIZATIONS=True # False if no visualization environment exists

import PostProcessingPlots as pplot
import matplotlib.pyplot as plt

# %% Methods
# Function to remove edges from the labeled segments to avoid edge effects in analysis
def modify_edges(labeled_segments):
    # Set the border pixels to 0
    labeled_segments[0, :], labeled_segments[:, 0], labeled_segments[-1, :], labeled_segments[:, -1] = 0, 0, 0, 0
    return labeled_segments

# Function to find endpoints of each labeled segment
def find_endpoints(labeled_segments, label):
    endpoints = [] # Store endpoints here
    # Iterate through each pixel to find endpoints
    for y in range(labeled_segments.shape[0]):
        for x in range(labeled_segments.shape[1]):
            if labeled_segments[y, x] == label:
                # Count neighbors with the same label
                neighbor_count = sum([labeled_segments[y + dy, x + dx] == label for dy in range(-1, 2) for dx in range(-1, 2) if not (dx == 0 and dy == 0) and 0 <= y + dy < labeled_segments.shape[0] and 0 <= x + dx < labeled_segments.shape[1]])
                if neighbor_count == 1:
                    endpoints.append((y, x))
    return endpoints

# Function to order the points in a segment from start to end
def order_segment_points(labeled_segments, start, label):
    ordered_points = [start]  # Start with the first endpoint
    current_point = start
    # Loop until no more connected points
    while True:
        found_next = False
        # Check neighbors for the next point in the segment
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue  # Skip the current point
                next_point = (current_point[0] + dy, current_point[1] + dx)
                if next_point in ordered_points:
                    continue  # Skip if already in the list
                # Check if the neighbor belongs to the segment
                if 0 <= next_point[0] < labeled_segments.shape[0] and 0 <= next_point[1] < labeled_segments.shape[1] and labeled_segments[next_point[0], next_point[1]] == label:
                    ordered_points.append(next_point)
                    current_point = next_point
                    found_next = True
                    break
            if found_next:
                break
        if not found_next:  # No more points to add
            break  # Exit if no next point found
    return ordered_points

# Function to get points at specified fractions along the ordered segment
def get_fraction_points(ordered_points, fractions):
    fraction_points = []
    for fraction in fractions:
        index = int(len(ordered_points) * fraction)
        # Adjust for indexing and ensure it doesn't exceed the list length
        fraction_points.append(ordered_points[min(index, len(ordered_points)-1)])  # Adjust for indexing
    return fraction_points

# Function to calculate points perpendicular to the midpoint or specified fraction points
def perpendicular_points(start, end, num_points, separation):
    # Calculate direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0: return []  # Avoid division by zero
    dx, dy = dx / norm, dy / norm
    # Calculate perpendicular direction
    perp_dx, perp_dy = -dy, dx
    mid = np.array([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2])
    points = []
    # Generate points along the perpendicular line
    for i in range(-num_points // 2, num_points // 2 + 1):
        if i == 0: continue  # Skip the middle point to ensure symmetry
        point = mid + i * separation * np.array([perp_dx, perp_dy])
        points.append(point.astype(int))
    return points

# Main function to process segments and calculate scan points
def process_segments(labeled_segments, fractions, num_scan_points=2, pixel_separation=2):
    img_shape = np.shape(labeled_segments)[0]
    modified_segments = np.copy(labeled_segments)  # Copy to avoid modifying the original
    all_scan_points = []  # Store all scan points
    for label in np.unique(labeled_segments)[1:]:  # Skip background
        endpoints = find_endpoints(labeled_segments, label)
        if not endpoints:
            continue  # Skip if no endpoints (shouldn't happen in well-formed segments)
        ordered_points = order_segment_points(labeled_segments, endpoints[0], label)
        fraction_points = get_fraction_points(ordered_points, fractions)
        
        for point in fraction_points:
            # Mark fraction points for visualization and add to scan points list
            modified_segments[point[0], point[1]] = -1  # Special value for visualization
            all_scan_points.append([point[1], point[0]])  # Append as [x, y] for CSV
            neighbor = post2.find_neighbor_pixel(labeled_segments, point[0], point[1], label)
            if neighbor:
                # Calculate and mark perpendicular points
                perp_points = perpendicular_points(point, neighbor, num_scan_points, pixel_separation)
                for perp_point in perp_points:
                    if 0 <= perp_point[0] < img_shape and 0 <= perp_point[1] < img_shape:
                        modified_segments[perp_point[0], perp_point[1]] = -1 # Special value for visualization
                        all_scan_points.append([perp_point[1], perp_point[0]])  # Append as [x, y] for CSV
            
    return modified_segments, all_scan_points
# %% Main execution block
if __name__ == "__main__":   
    # User inputs
    window_size = 10 # Size of the sliding window for processing
    fractions = [0.5] # Specify fractions to place scan points (e.g, [0.5] or [0.25, 0.75])
    num_scan_points = 2 # Number of scan points along the perpendicular line (excluding the fraction point)
    pixel_separation = 2  # Separation between scan points
    
    # Load data from a previously saved file
    pred_after = joblib.load('./Post4.pkl')

    # Label features in the loaded data
    print("Labeling features...")
    (_, _, _, labeled_complete_segments, _, _, _) = post2.label_features(pred_after, window_size) # TODO: This can be obtained from previous scripts
    print("Features are labeled.")

    # %% Find Line scan points
    print("Finding line scan points...")
    
    # Process segments to identify line scan points
    modified_segments, scan_points = process_segments(modify_edges(labeled_complete_segments), fractions)
    print("Line scan points are identified.")
    
    # %% Plot the results
    # Prepare an image for visualization
    classified_post6 = np.zeros_like(modified_segments, dtype=np.uint8)
    classified_post6[modified_segments > 0] = 1 # Complete segments
    classified_post6[modified_segments == -1] = 2 # Line scan points  
    
    # Define a custom colormap
    from matplotlib.colors import ListedColormap
    color_list = ['black', 'gray', 'cyan']
    cmap = ListedColormap(color_list)    
    # Plot the result with the custom colormap
    pplot.plot_multiple_images_single_row([classified_post6], fig_size=(10,10), cmap=cmap, titles=["Line Scan Points at {} fractions ({} points separated by {} pixels)".format(fractions, num_scan_points, pixel_separation)])

    DPI = 500
    # Type (pdf, png, jpg, svg etc.)
    FILETYPE = 'png'
    
    # Plot figures
    plt.savefig('./scan_points.' + FILETYPE, dpi=DPI, transparent=False)
    if not VISUALIZATIONS:
        plt.close()
    # %% Save scan points to CSV
    scan_points_df = pd.DataFrame(scan_points, columns=['X', 'Y']) # Prepare DataFrame
    scan_points_df.to_csv('scan_points.csv', index=False)  # Save to CSV
    
    print("Processing complete. Line scan points are saved to 'scan_points.csv'.")