import copy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import heapq


def construct_knn_graph(points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    graph = {}  # Initialize an empty graph
    edge_distances = {}
    for i, neighbors in enumerate(indices):
        graph[i] = set(neighbors[1:])
        for j, dist in zip(neighbors[1:], distances[i][1:]):
            edge_distances[(i, j)] = dist
    return graph, edge_distances

def assign_node_weights(graph, V):
    node_weights = {}
    for node, neighbors in graph.items():
        weight = np.max([np.abs(V[node]), np.max(np.abs(V[list(neighbors)]))])
        weight = np.round(weight, 5)
        node_weights[node] = weight
    return node_weights

def assign_edge_weights(graph, edge_distances, V):
    edge_weights = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            weight = np.max([np.abs(V[node]), np.abs(V[neighbor])])
            weight = np.round(weight, 5)
            edge_weights[(node, neighbor)] = {'weight': weight*edge_distances[(node, neighbor)], 'potential': weight, 'distance': edge_distances[(node, neighbor)]}
    return edge_weights

def get_knn_graph(surface, min_components, max_K=None, verbose=False):
    if max_K is None:
        max_K = np.sqrt(len(surface)).astype(int)
    for k in range(2, max_K):
        knn_graph, edge_distances = construct_knn_graph(surface, k=k)


        G = nx.DiGraph()

        for u, nghbrs in knn_graph.items():
            for v in nghbrs:
                G.add_edge(u, v)
        sccs = list(nx.strongly_connected_components(G))
        if verbose:
            print(f"K={k} -> {len(sccs)}")
        if len(sccs) <= min_components:
            break
        if verbose:
            print(f"Nodes remaining: {sorted(sccs, key=lambda x: len(x))[0]}")
    return knn_graph, edge_distances

def a_star_search(G, start, goal):
    # Initialize the priority queue with the start node and its f value (f = g + h)
    open_queue = [(G.nodes[start].get('potential', 0), start, 0)]  # (f_value, node, g_value)
    # Dictionary to store the best g_value for each visited node
    g_values = {start: 0}
    # Dictionary to store the parent of each node (used for reconstructing the path)
    came_from = {start: None}
    
    while open_queue:
        # Pop the node with the smallest f_value
        _, current, current_g = heapq.heappop(open_queue)
        
        # Goal reached, reconstruct and return path
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        # Explore neighbors
        for neighbor, edge_data in G[current].items():
            tentative_g = current_g + edge_data.get('weight', 1)
            
            # If a better path to this neighbor is found, update its g_value and f_value
            if tentative_g < g_values.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_values[neighbor] = tentative_g
                f_value = tentative_g + G.nodes[neighbor].get('potential', 0)
                heapq.heappush(open_queue, (f_value, neighbor, tentative_g))
                
    # If the loop completes, then there is no path from start to goal
    return None