import numpy as np
from sklearn.metrics import silhouette_score

def shortest_path_distance(graph, source, target):
    visited = {node: False for node in graph.nodes()}
    distance = {node: float('inf') for node in graph.nodes()}
    queue = [source]
    visited[source] = True
    distance[source] = 0
    
    while queue:
        current_node = queue.pop(0)
        
        for neighbor in graph.neighbors(current_node):
            if not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[current_node] + 1  # Increase distance by 1
                queue.append(neighbor)
                
                # Stop early if we reach the target node
                if neighbor == target:
                    return distance[neighbor]
                    
    return float('inf') # No path between the nodes

def calc_graph_distance_matrix(graph):
  # For undirected graph
  N = len(graph.nodes())
  dist_matrix = np.zeros((N, N))
  for i, n1 in enumerate(graph.nodes()):
    for j, n2 in enumerate(graph.nodes()):
      if (i == j): continue
      if (dist_matrix[i, j] != 0): continue
      dist = shortest_path_distance(graph, n1, n2)
      dist_matrix[i, j] = dist
      dist_matrix[j, i] = dist
  return dist_matrix

def calc_clusters_distance_matrix(graph, clusters, kind="mean"):
  # For undirected graph
  N = len(clusters)
  dist_matrix = np.zeros((N, N))
  node_labels = list(graph.nodes())
  for i, cluster_i in enumerate(clusters):
    for j, cluster_j in enumerate(clusters):
      if (i == j): continue
      if (dist_matrix[i, j] != 0): continue

      if (kind == "min"):
        # Min distance
        min_dist = float('inf')
        for ni in cluster_i:
          for nj in cluster_j:
            dist = shortest_path_distance(graph, node_labels[ni], node_labels[nj])
            if (dist < min_dist):
              dist_matrix[i, j] = dist
              dist_matrix[j, i] = dist
              min_dist = dist

      if (kind == "mean"):
        # Average distance
        s_dist = 0
        n_pairs = 0
        for ni in cluster_i:
          for nj in cluster_j:
            dist = shortest_path_distance(graph, node_labels[ni], node_labels[nj])
            s_dist += dist
            n_pairs += 1
        mean_dist = s_dist / n_pairs
        dist_matrix[i, j] = mean_dist
        dist_matrix[j, i] = mean_dist
  return dist_matrix

def single_linkage_distance(cluster1, cluster2, distance_matrix):
    """Single linkage distance between two clusters"""
    min_dist = float('inf')
    for i in cluster1:
        for j in cluster2:
            min_dist = min(min_dist, distance_matrix[i][j])
    return min_dist

def hierarchical_clustering(graph, n_clusters, kind="mean"):
  """Heirarchical clustering"""
  dist_matrix = calc_graph_distance_matrix(graph)
  N = len(dist_matrix)
  clusters = [[i] for i in range(N)]
  current_dist_matrix = np.array(dist_matrix)

  while len(clusters) > n_clusters:
    min_dist = float('inf')
    to_merge = None
    for i in range(len(clusters)):
      for j in range(i + 1, len(clusters)):
        if (len(clusters) == N):
          dist = single_linkage_distance(clusters[i], clusters[j], current_dist_matrix)
        else:
          dist = current_dist_matrix[i][j]
        if dist < min_dist:
          min_dist = dist
          to_merge = (i, j)

    # Merge the closest clusters
    i, j = to_merge
    clusters[i].extend(clusters[j])
    del clusters[j] 

    current_dist_matrix = calc_clusters_distance_matrix(graph, clusters, kind)
    if (len(clusters) != len(current_dist_matrix)): print("wrong size")

  return clusters

import copy

def hierarchical_clustering_unc(graph, kind="mean"):
  """Heirarchical clustering with unknown number of clusters"""
  dist_matrix = calc_graph_distance_matrix(graph)
  N = len(dist_matrix)
  clusters = [[i] for i in range(N)]
  current_dist_matrix = np.array(dist_matrix)

  best_score = -1 
  best_clusters = copy.deepcopy(clusters)
  n_no_improvements = 0
  silhouette_scores = []

  while len(clusters) > 2:
    min_dist = float('inf')
    to_merge = None
    for i in range(len(clusters)):
      for j in range(i + 1, len(clusters)):
        if (len(clusters) == N):
          dist = single_linkage_distance(clusters[i], clusters[j], current_dist_matrix)
        else:
          dist = current_dist_matrix[i][j]
        if dist < min_dist:
          min_dist = dist
          to_merge = (i, j)

    # Merge the closest clusters
    i, j = to_merge
    clusters[i].extend(clusters[j])
    del clusters[j]

    current_dist_matrix = calc_clusters_distance_matrix(graph, clusters, kind)
    if (len(clusters) != len(current_dist_matrix)): print("wrong size")

    cluster_labels = np.zeros(N)
    for idx, cluster in enumerate(clusters):
      for node in cluster:
        cluster_labels[node] = idx

    try:
      score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    except ValueError:
      score = -1
    
    silhouette_scores.append(score)
    
    if (score > best_score):
      best_score = score
      best_clusters = copy.deepcopy(clusters)
      n_no_improvements = 0

    if (score <= best_score): n_no_improvements +=1

    if (n_no_improvements > 10 and best_score > 0):
      break

  # print(silhouette_scores)
  return best_clusters