# ---------------------------------------------------------------------------
# --- Description --
# Library containing methods required for the MILP solution of the combinatorial 
# optimization problem described in the paper. It also contains methods for calculating
# image features.
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 06/27/2023
# ------------------
# Version: Python 3.8
# ---------------------------------------------------------------------------
import numpy as np
from scipy.spatial import distance
import math
from scipy.optimize import linprog
# %% General Helper functions
# Bresenham's Line Algorithm
def line(start, end):
    """Bresenham's Line Algorithm
    This function generates a list of grid coordinates that approximates a 
    line between two points.
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # initialize list of points
    points = [] # Empty list to store points
 
    # decide whether to move in the positive or negative x direction
    is_steep = abs(dy) > abs(dx) # Check if the line is steep
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # swapped decides whether to swap the start and end points
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    dx = x2 - x1 # Recalculate delta x
    dy = y2 - y1 # Recalculate delta y
 
    # calculate error
    error = int(dx / 2.0)
    y = y1
    y_step = None
 
    # decide the direction of y
    if y1 < y2:
        y_step = 1 # Move y in positive direction
    else:
        y_step = -1 # Move y in negative direction
 
    # iterate over the x range and generate points
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y) # Determine point coordinates
        points.append(coord)
        error -= abs(dy) # Decrease error
        if error < 0: # If error falls below zero
            y += y_step # Adjust y
            error += dx # Adjust error
 
    # reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points
    
def calculate_cost(point1, point2, l_junctions, labeled_elements, max_distance = 64):
    # Calculate the distance between two points
    euclidean_distance = distance.euclidean(point1, point2)
    
    if (euclidean_distance > max_distance):
        return float('inf')
    else:
        # Find the coordinates of the path in the discrete grid using Bresenham's Line Algorithm
        path_coords = line(point1, point2)
        
        # Remove the coords of isolated points from the found coordinates
        path_coords = [item for item in path_coords if item not in l_junctions]
    
        # Find how many segments need to be crossed between the two points (for occlusion)
        segment_cross_count = sum([1 for pt in path_coords if labeled_elements[pt] != 0])
        
        if (segment_cross_count > 0):
            return float('inf')
        else:
            return 0
# %% Helper functions to calculate image features
def line_parameters(p, q):
    """
    Compute the slope and y-intercept for the line defined by points p and q.

    Args:
    - p, q: Tuple[int, int] - endpoints of the line segment.

    Returns:
    - Tuple[float, float] - (slope, y-intercept) of the line.
    """
    
    # Calculate the slope (m) of the line segment. 
    # If the denominator is 0 (i.e., vertical line), set slope to infinity.
    if q[1] - p[1] == 0: 
        m = float('inf')
        b = p[1]
    else:
        m = (q[0] - p[0]) / (q[1] - p[1])
        b = p[0] - m * p[1]  # Calculate the y-intercept (b) using the formula: b = y - mx
    return m, b

def find_intersection(extended_line_1, extended_line_2):
    """
    Find the intersection point of two line segments, if it exists.

    Args:
    - extended_line_1, extended_line_2: Tuple[Tuple[int, int], Tuple[int, int]]
      - Each line is defined by its start and end points.

    Returns:
    - Tuple[int, int] if the intersection exists and lies on both line segments.
    - None otherwise.
    """
    
    # Extract the start and end points of both lines.
    p1, q1 = extended_line_1
    p2, q2 = extended_line_2

    # Calculate the parameters (slope and y-intercept) for both lines.
    m1, b1 = line_parameters(p1, q1)
    m2, b2 = line_parameters(p2, q2)

    # If slopes are equal, the lines are parallel and might not intersect (unless collinear).
    if m1 == m2:
        return None

    # Calculate the x-coordinate of the intersection point using the formula: x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    
    # Calculate the y-coordinate using the line equation of the first line: y = m1 * x + b1
    y = m1 * x + b1

    # Check if the calculated intersection point lies on both the line segments.
    # It should be within the x and y bounds of both segments.
    if (x >= min(p1[1], q1[1]) and x <= max(p1[1], q1[1]) and
            y >= min(p1[0], q1[0]) and y <= max(p1[0], q1[0]) and
            x >= min(p2[1], q2[1]) and x <= max(p2[1], q2[1]) and
            y >= min(p2[0], q2[0]) and y <= max(p2[0], q2[0])):
        
        # If it does, return the intersection point.
        return y,x
    
    # If not, return None.
    return None

# Helper function to extend a line from a point in a given direction
def extend_line_from_point(point, angle, length=64):
    # Convert angle to radians
    rad = np.radians(angle)
    
    # Calculate the end point
    end_y = int(np.round(point[0] - length * np.sin(rad)))
    end_x = int(np.round(point[1] + length * np.cos(rad)))
    
    return point, (end_y, end_x)

def triangle_centroid(p1, p2, p3):
    """Compute the centroid of a triangle defined by points p1, p2, and p3."""
    centroid_y = (p1[0] + p2[0] + p3[0]) / 3
    centroid_x = (p1[1] + p2[1] + p3[1]) / 3
    return (centroid_y, centroid_x)

def circle_from_three_points(p1, p2, p3):
    """Compute the center and radius of the circle passing through points p1, p2, and p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the determinants
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    if D == 0:
        print("The three points ({},{},{}) are collinear.".format(p1, p2, p3))
        return None, None
        # raise ValueError("The three points are collinear.")
    
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    center = (Ux, Uy)
    
    # Radius is the distance from the center to any of the three points
    radius = np.sqrt((Ux - x1)**2 + (Uy - y1)**2)
    
    return center, radius

def arc_length(radius, angle_degrees):
    """Compute the arc length given the circle's radius and subtended angle (in degrees)."""
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees
    return radius * np.radians(angle_degrees)

def angle_between_lines(line_1, line_2):
    """Compute the angle between lines"""
    y1, x1 = line_1[0]
    y3, x3 = line_1[1]
    y2, x2 = line_2[0]
    y4, x4 = line_2[1]
    
    # Calculate direction vectors
    A = (y3-y1, x3-x1)
    B = (y4-y2, x4-x2)
    
    # Dot product
    dot_product = A[0] * B[0] + A[1] * B[1]
    
    # Magnitudes
    mag_A = math.sqrt(A[0]**2 + A[1]**2)
    mag_B = math.sqrt(B[0]**2 + B[1]**2)
    
    # Angle in radians
    if mag_A == 0 or mag_B == 0:
        theta=np.inf
        print("!!! Division by zero error in angle_between_lines")
    else:
        theta = math.acos(dot_product / (mag_A * mag_B))
    
    # Convert to degrees
    angle_in_degrees = np.round(math.degrees(theta),0) # Round to the closest value (to avoid 0.1 to 0.3 discrepancies)

    # Return the acute angle
    return min(180 - angle_in_degrees, angle_in_degrees)

# %% Compute the interface potential image features
def image_features(clique, broken_segments, completion_segments, iso_points, image, alpha_i, beta_i):
    """ Compute the interface potential features, see Ming et al 2012 Connected Contours for descriptions"""
    from scipy.spatial import distance

    c_segment = completion_segments[clique[0]]
    b_segment_1 = broken_segments[clique[1]]
    b_segment_2 = broken_segments[clique[2]][::-1]

    k_I_curr_clique = np.zeros(7)
    
    # Last four are distances (new ones)
    k_I_curr_clique[4] = distance.euclidean(c_segment[0], c_segment[-1]) # Length of the completion segment
    k_I_curr_clique[5] = distance.euclidean(b_segment_1[0], b_segment_1[-1]) # Length of the first broken segment
    k_I_curr_clique[6] = distance.euclidean(b_segment_2[0], b_segment_2[-1]) # Length of the second broken segment
    
    # Extend lines from the isolated points in the direction of the curving angles
    extended_line_1 = extend_line_from_point(c_segment[0], iso_points[c_segment[0]], image.shape[0])
    extended_line_2 = extend_line_from_point(c_segment[-1], iso_points[c_segment[-1]], image.shape[0])
        
    # Check if the lines extended from the endpoints of the broken segments cross
    intersection_point = find_intersection(extended_line_1, extended_line_2)
    
    if intersection_point:
        edge_ij = (c_segment[0], intersection_point)
        edge_ik = (intersection_point, c_segment[-1])

        # Calculate effective corner distance
        dist1 = distance.euclidean(edge_ij[0], edge_ij[1])
        dist2 = distance.euclidean(edge_ik[0], edge_ik[1])
        
        k_I_curr_clique[0] = dist1 + dist2

        # Calculate the effective smooth distance
        centroid = triangle_centroid(c_segment[0], c_segment[-1], intersection_point)
        center, radius = circle_from_three_points(centroid, c_segment[0], c_segment[-1])
        theta = angle_between_lines((c_segment[0], centroid), (centroid, c_segment[-1]))
        if radius is not None: smooth_distance = arc_length(radius, 2 * theta)
        else: smooth_distance = dist1 + dist2
        
        k_I_curr_clique[1] = smooth_distance

        if (dist1 == 0) or (dist2 == 0):
            theta_ij = iso_points[c_segment[0]]
            theta_ik = iso_points[c_segment[-1]]        
        else:
            # Calculate theta_i,j and theta_i,k
            theta_ij = int(angle_between_lines(c_segment, edge_ij))
            theta_ik = int(angle_between_lines(c_segment, edge_ik))
        
        k_I_curr_clique[2] = alpha_i * (theta_ij + theta_ik) ** 2
        k_I_curr_clique[3] = beta_i * (theta_ij - theta_ik) ** 2
        
        return k_I_curr_clique

    else:
        # Calculate y and x differences between either end of the completion segment
        y_diff = np.abs(c_segment[0][0] - c_segment[-1][0])
        x_diff = np.abs(c_segment[0][1] - c_segment[-1][1])
        
        if y_diff != 0 and x_diff != 0:
            edge_ij = (c_segment[0]),(c_segment[0][0], c_segment[-1][1])
            edge_ik = (c_segment[0][0], c_segment[-1][1]),(c_segment[-1])

            # Calculate effective corner distance
            k_I_curr_clique[0] = y_diff + x_diff

            # Calculate the effective smooth distance
            centroid = triangle_centroid(c_segment[0], c_segment[-1], (c_segment[0][0], c_segment[-1][1]))
            center, radius = circle_from_three_points(centroid, c_segment[0], c_segment[-1])
            theta = angle_between_lines((c_segment[0], centroid), (centroid, c_segment[-1]))
            if radius is not None: smooth_distance = arc_length(radius, 2 * theta)
            else: smooth_distance = y_diff + x_diff
            
            k_I_curr_clique[1] = smooth_distance

            # Calculate theta_i,j and theta_i,k
            theta_ij = int(np.round(np.degrees(math.atan2(y_diff,x_diff)),0))
            theta_ik = int(np.round(np.degrees(math.atan2(x_diff,y_diff)),0))
              
            k_I_curr_clique[2] = alpha_i * (theta_ij + theta_ik) ** 2
            k_I_curr_clique[3] = beta_i * (theta_ij - theta_ik) ** 2 

            return k_I_curr_clique
        else:                
            # Effective corner and smooth distances are both equal to the length of the completion segment
            dist = distance.euclidean(c_segment[0], c_segment[-1])
            
            k_I_curr_clique[0] = dist
            k_I_curr_clique[1] = dist
            
            # Both theta_i,j and theta_i,k are 0
            theta_ij = iso_points[c_segment[0]]
            theta_ik = iso_points[c_segment[-1]]
             
            k_I_curr_clique[2] = alpha_i * (theta_ij + theta_ik) ** 2
            k_I_curr_clique[3] = beta_i * (theta_ij - theta_ik) ** 2
            
            return k_I_curr_clique
# %% Methods to solve the mixed-integer linear programming
def objective_function(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M):
    obj_fun = []
    # Unary potential
    for key in broken_segments.keys():
        obj_fun.append(omega_u * k_p[key])
    
    # Interface potential and Complexity potential
    for key in completion_segments.keys():
        obj_fun.append(omega_I * sum([w_I[i] * k_I[key,i] for i in range(len(w_I))]) + omega_M * sum([w_M[i] * k_M[key,i] for i in range(len(w_M))]))          

    return np.array(obj_fun, dtype=object) # To avoid making it a "ragged" array

# Constraints
def constraint_function(broken_segments, completion_segments, C_P, C_B):
    A = np.zeros(((len(C_P) + len(C_B)),(len(broken_segments) + len(completion_segments))))
    b = np.zeros((len(C_P) + len(C_B)))

    # Completion Constraint
    for ind, (j, i) in enumerate(C_P): # j : broken, i : completion
        A[ind,(i + len(broken_segments))] = 1
        A[ind,j] = -1

    # Extension Constraint
    for ind, segment in enumerate(C_B):
        A[(ind + len(C_P)),segment[0]] = 1 # j: broken
        for i in segment[1:]: # i = completion
            A[(ind + len(C_P)),(i + len(broken_segments))] = -1

    return A, b

def MILP(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M, C_P, C_B, all_segments):
    
    c = objective_function(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M)

    A, b = constraint_function(broken_segments, completion_segments, C_P, C_B)

    # Bounds for each variable (0 <= y <= 1 for linear relaxation)
    original_bounds = [(0, 1) for _ in all_segments.keys()]
    
    # Solve the linear relaxation of the ILP
    res = linprog(c, A_ub=A, b_ub=b, bounds=original_bounds)
    # Check if the optimization was successful (handle cases where the optimization is infeasible)
    if not res.success:
        print("Optimization failed:", res.message)
        return None  # or some default value
    
    # Extract the selected paths
    selected_edges = [key for i, key in enumerate(all_segments.keys()) if res.x[i] >= 0.5]
    optimal_labels = np.zeros(len(all_segments))
    for segment_ind in selected_edges:
        optimal_labels[segment_ind] = 1        

    return optimal_labels

# Define the fitness function
def fitness(weights, broken_segments, completion_segments, k_p, k_M, C_P, C_C, C_B, all_segments, ground_truth_labels, iso_points, labeled_elements):
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, axis=0)
    scores = []
    k_I = np.zeros((len(completion_segments),7)) # Interface potential features
    for weight in weights:
        w_I = weight[:7].reshape(7, 1)
        w_M = weight[7:9].reshape(2, 1)
        omega_u = weight[9]
        omega_I = weight[10]
        omega_M = weight[11]
        alpha_i = weight[12]
        beta_i = weight[13]
        for clique in C_C: 
            k_I[clique[0], :] = image_features(clique, broken_segments, completion_segments, iso_points, labeled_elements, alpha_i, beta_i)
            
        optimal_labels = MILP(broken_segments, completion_segments, k_p, k_I, k_M, w_I, w_M, omega_u, omega_I, omega_M, C_P, C_B, all_segments)
        
        # Check if MILP returned a solution
        if optimal_labels is None:
            # Raise an exception
            raise Exception("No feasible solution with the given constraints can be found.")
        
        scores.append(np.sum(np.abs(ground_truth_labels - optimal_labels)))
    return np.array(scores)