import numpy as np
import heapq
from collections import deque
from .graph import Cell
from .utils import trace_path

"""
General graph search instructions:

First, define the correct data type to keep track of your visited cells
and add the start cell to it. If you need to initialize any properties
of the start cell, do that too.

Next, implement the graph search function. When you find a path, use the
trace_path() function to return a path given the goal cell and the graph. You
must have kept track of the parent of each node correctly and have implemented
the graph.get_parent() function for this to work. If you do not find a path,
return an empty list.

To visualize which cells are visited in the navigation webapp, save each
visited cell in the list in the graph class as follows:
     graph.visited_cells.append(Cell(cell_i, cell_j))
where cell_i and cell_j are the cell indices of the visited cell you want to
visualize.
"""


def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm."""
    graph.init_graph()
    stack = [start]
    visited = {(start.i, start.j)}
    graph.parent[(start.i, start.j)] = None
    
    while stack:
        current = stack.pop()
        graph.visited_cells.append(Cell(current.i, current.j))
        
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        for neighbor in graph.find_neighbors(current.i, current.j):
            key = (neighbor.i, neighbor.j)
            if key not in visited:
                visited.add(key)
                graph.parent[key] = Cell(current.i, current.j)
                stack.append(neighbor)
    
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm."""
    graph.init_graph()
    queue = deque([start])
    visited = {(start.i, start.j)}
    graph.parent[(start.i, start.j)] = None
    
    while queue:
        current = queue.popleft()
        graph.visited_cells.append(Cell(current.i, current.j))
        
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        for neighbor in graph.find_neighbors(current.i, current.j):
            key = (neighbor.i, neighbor.j)
            if key not in visited:
                visited.add(key)
                graph.parent[key] = Cell(current.i, current.j)
                queue.append(neighbor)
    
    return []


def a_star_search(graph, start, goal):
    """A* Search algorithm."""
    graph.init_graph()
    
    def h(cell):
        return np.sqrt((cell.i - goal.i)**2 + (cell.j - goal.j)**2)
    
    counter = 0
    open_set = [(h(start), 0, counter, start)]
    visited = set()
    graph.distance[(start.i, start.j)] = 0
    graph.parent[(start.i, start.j)] = None
    
    while open_set:
        f, g, _, current = heapq.heappop(open_set)
        key = (current.i, current.j)
        
        if key in visited:
            continue
            
        visited.add(key)
        graph.visited_cells.append(Cell(current.i, current.j))
        
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        for neighbor in graph.find_neighbors(current.i, current.j):
            nkey = (neighbor.i, neighbor.j)
            new_g = g + 1
            
            if nkey not in visited and (nkey not in graph.distance or new_g < graph.distance[nkey]):
                graph.distance[nkey] = new_g
                graph.parent[nkey] = Cell(current.i, current.j)
                counter += 1
                heapq.heappush(open_set, (new_g + h(neighbor), new_g, counter, neighbor))
    
    return []
