# From https://www.redblobgames.com/pathfinding/a-star/implementation.html
# typing library removed and comments are added by ChatGPT

import numpy as np
import heapq
'''
`Graph` is a protocol class that declares a method `neighbors` that should be 
implemented by any class that wishes to behave as a graph. This method is 
supposed to return a list of neighbors for a given node.
'''
class Graph:
    def neighbors(self, id):
        pass

'''
`SquareGrid` is a class that represents a grid of squares. It has a method 
`in_bounds` to check if a given location is within the bounds of the grid, 
`passable` to check if a given location is not a wall (is passable), and 
`neighbors` to return the neighbors of a given location.
'''
class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    # Check if the id is within the grid's boundaries
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    # Check if the id is passable (not a wall)
    def passable(self, id):
        return id not in self.walls
    
    # Check if the movement is diagonal
    def is_diagonal_movement(self, id1, id2):
        x1, y1 = id1
        x2, y2 = id2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return dx == dy == 1  # true if movement is diagonal

    # Return the id's neighboring grid locations
    def neighbors(self, id):
        (x, y) = id
        # 8-way movement - E W N S NE NW SE SW
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        if (x + y) % 2 == 0: neighbors.reverse() # reverse to prioritize straight paths

        results = []
        for next in neighbors:
            if not self.in_bounds(next): continue
            if not self.passable(next): continue
        
            '''
            While the agent can now move diagonally, it can also move through 
            the diagonal between two adjacent obstacles
            '''
            # Restrict movement through diagonal walls 
            if self.is_diagonal_movement(id, next):
                dx, dy = next[0] - id[0], next[1] - id[1]
                # Check if the move is "through" a wall by checking the two adjacent cells
                if not self.passable((id[0] + dx, id[1])) or not self.passable((id[0], id[1] + dy)):
                    continue
            results.append(next)
        return results    

'''
`GridWithWeights` is a class that extends `SquareGrid`. It adds weights to the 
squares in the grid and it declares a method `cost` that returns the weight of 
moving to a given square.
'''
class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}


    # Note, we satisfy the non-linearity by selecting different source and sink angles
    # Here it is related to the A* algorithm
    # Return the cost of moving to the node and prefer straighter paths
    def cost(self, from_node, to_node):
        # Calculate the init_cost (could result in a 4-way tie)
        init_cost = self.weights.get(to_node, 1)
        
        # 4-way tie-breaking hack: Add a tiny movement penalty (1.001 if diagonal, 1 otherwise)
        nudge = 0
        (x1, y1) = from_node
        (x2, y2) = to_node
        if (x1 + y1) % 2 == 0 and x2 != x1: nudge = 1
        if (x1 + y1) % 2 == 1 and y2 != y1: nudge = 1
        
        # Implement higher cost for diagonal movements 1.001 (Original Source) or sqrt(2) = 1.41421356 (ChatGPT)
        return init_cost + 0.001 * nudge

'''
`PriorityQueue` is a class that wraps a list of elements into a priority queue 
where elements are dequeued based on their priority.
'''
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    # Check if the queue is empty
    def empty(self):
        return not self.elements
    
    # Add an item with priority to the queue
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    # Pop and return the item with the lowest priority
    def get(self):
        return heapq.heappop(self.elements)[1]

'''
`reconstruct_path` is a function that takes a dictionary of locations and their 
preceding locations, a start location, and a goal location. It returns the path 
from the start to the goal.
'''
# Function to reconstruct the path from start to goal using came_from dict
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

'''
`heuristic` is a function that computes the Manhattan distance (sum of absolute 
differences in x and y coordinates) between two locations. This is used in A* 
to estimate the remaining cost to the goal.
'''
# Function to compute the Manhattan distance between a and b
def heuristic(a, b):
    
    def manhattan_disance(a,b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)
    
    def euclidean_distance(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def chebyshev_distance(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return max(abs(x1 - x2), abs(y1 - y2))
    
    def zero_heuristic(a, b):
        return 0
    
    def octile_distance(a, b):
        (x1, y1) = a
        (x2, y2) = b
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return dx + dy + (2**0.5 - 2) * min(dx, dy)
    
    return euclidean_distance(a,b)
    # return chebyshev_distance(a,b)


'''
`a_star_search` is the function that implements the A* search algorithm. It 
uses a priority queue to keep track of the frontier (the set of locations to 
be explored next), and dictionaries to keep track of the location the path came 
from and the cost to reach each location. It returns these two dictionaries.
'''
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

'''
`array_to_dict` is a function that takes a 2D array (a cost map) and converts 
it to a dictionary that maps locations to costs.
'''
# Function to convert a 2D cost map to a dictionary
def array_to_dict(cost_map):
    result_dict = {}
    for i in range(cost_map.shape[0]):
        for j in range(cost_map.shape[1]):
            result_dict[(i, j)] = cost_map[i, j]
    return result_dict

'''
Obtain foreground pixels from the field to use as walls.
'''
def get_foreground_coords(field):
    '''
    np.where() returns a tuple of arrays, one for each dimension of 'field', 
    containing the indices where 'field' is 1
    '''
    foreground_coords = np.where((field == 1))

    '''
    zip the arrays together to get tuples of coordinates
    '''
    coords = list(zip(foreground_coords[0], foreground_coords[1]))
    
    return coords