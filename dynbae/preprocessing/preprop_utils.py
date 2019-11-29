# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:02:43 2019

@author: orteg
"""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

# take first element for sort
def takeFirst(elem):
    return elem[0]
  
def desc_graph(graph):
  '''Generate Descritive Statistics of a Graph

  Args:
    graph (nx network):

  Raises:
    Exception: Format Problem

  Returns:
    No Return: Print Descriptive Statistics
  '''
  if not isinstance(graph, nx.Graph):
    raise Exception('first argument must be a nx Graph. The type of argument is {}'.format(type(graph)))
    
  print('Number of edges: ' + str(len(graph.edges())))
  print('Number of nodes: ' + str(len(graph.nodes())))
  print('Average degree: ' + str(np.mean(np.array(list(graph.degree()))[:,1])))
  print('Average clustering coef.: ' + str(nx.average_clustering(graph)))
  print('Number of connected components: ' + str(nx.number_connected_components(graph)))
  print('Degree assortativity coefficient: ' + str(nx.degree_assortativity_coefficient(graph)))
  print('Density: ' + str(nx.density(graph)))
    
def splitgraph(df, sep, dynamic = True, unseen_nodes_test = False, 
               unseen_nodes_train = True):
  '''Splitting the pandas df into training and test

  Args:
    df (pandas dataframe): The dataframe in which node1, node2, and separator must be included
    sep (tuple): Pair of variable name and snapshot period
    dynamic (boolean): Generate a list of graphs for the training
    unseen_nodes_test (boolean): Generate a test graph with only known nodes from training (i.e. a subgraph)
    unseen_nodes_train (boolean): Generate all training graphs with the complete list of known nodes
      
  Raises:
    Exception: Format Problem

  Returns:
    grafs_tr_ (list): Training Graph Snapshots
    grafs_ts_ (list): Test Graph Snapshots
  '''
  if not isinstance(df, pd.DataFrame):
    raise Exception('first argument must be a pandas DataFrame. The type of argument is {}'.format(type(df)))
  if not isinstance(sep, tuple):
    raise Exception('first argument must be a tuple. The type of argument is {}'.format(type(sep)))
   
  var_ = sep[0]  # Name of the separator
  val_ = sep[1]  # Size of the Training Set in Snapshots
   
  df.loc[:,var_] = df.loc[:,var_].astype(int)  # Ensure separator is integer
  min_val = df.loc[:,var_].min()  # the timestamps need to be monotonically increasing
  tuple_aux = list(df[['node1','node2',var_]].itertuples(index=False, name=None))
  G_tr_ = nx.Graph([(u,v) for u,v,e in tuple_aux if e <= val_])
  
# Length of total time steps the graph will dynamically change
# training Set in which last period is test set
  
  length_ = 1 + np.ravel(np.where(df.loc[:,var_].unique() == val_))[0]
  
  if dynamic:
    
    print('Dynamic Approach!')
    print("-----------------")
    grafs_tr_ = [0]*length_
  
    # Train Set - For loop that adds nodes to Graphs
    for j in range(length_):
      i = j + min_val  # Set the training periods from the min to the end of training length
      
      print("Time Period: " + str(i))
    
      # Adding only edges that correspond to the period i of training set
      grafs_tr_[j] = nx.Graph([(u,v) for u,v,e in tuple_aux if e == i])
            
      print("Length of nodes training snapshot: " + str(len(grafs_tr_[j].nodes())))      
      print("Length of edges training snapshot: " + str(len(grafs_tr_[j].edges())))
      print("*---------------------------------*")
      
      # Create Subgraph to control for the unseen nodes in training sets
      if unseen_nodes_train == True:
        grafs_tr_[j].add_nodes_from(G_tr_.nodes(data=True))  
    # Test Set
    grafs_ts_ = nx.Graph()
    G_ts_ = nx.Graph([(u,v) for u,v,e in tuple_aux if e == (val_ + 1) ])
    grafs_ts_ = G_ts_
    
    # Create Subgraph to control for the unseen nodes
    if unseen_nodes_test == True:
      grafs_ts_ = grafs_ts_.subgraph(G_tr_.nodes())
    
    
    print("Time Period: " + str(val_ + 1))
    print("Length of nodes test snapshot: " + str(len(grafs_ts_.nodes())))      
    print("Length of edges test snapshot: " + str(len(grafs_ts_.edges())))
    print("*---------------------------------*")
    return(grafs_tr_, grafs_ts_)
    
  else:
    
    print('Static Approach!')
    print("-----------------")
    grafs_tr_ = nx.Graph()
    
    # Adding only edges that correspond to the period i of training set
    grafs_tr_ = G_tr_
    
    # Description of Snapshots
    print("Time Period: " + str(val_))
    print("Length of nodes training snapshot: " + str(len(grafs_tr_.nodes())))      
    print("Length of edges training snapshot: " + str(len(grafs_tr_.edges())))
    print("*---------------------------------*")
      
    # Test Set
    grafs_ts_ = nx.Graph()
    
    G_ts_ = nx.Graph([(u,v) for u,v,e in tuple_aux if e == (val_ + 1) ])
    grafs_ts_ = G_ts_
    
    # Create Subgraph to control for the unseen nodes
    if unseen_nodes_test == True:
      grafs_ts_ = grafs_ts_.subgraph(grafs_tr_.nodes())
    
    # Description of Snapshots
    print("Time Period: " + str(val_ + 1))
    print("Length of nodes test snapshot: " + str(len(grafs_ts_.nodes())))      
    print("Length of edges test snapshot: " + str(len(grafs_ts_.edges())))
    print("*---------------------------------*")
    return(grafs_tr_, grafs_ts_)

# Splits id with format (node1_node2)  
def nodeid_split(word,i):
  aux_list = word['id'].split('_')
  return(aux_list[i])
  
def getEdgeListFromAdjMtx(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if(j == i):
                    continue
                if(is_undirected and i >= j):
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result
  
def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=10):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key = lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors

def MAP(predicted_edge_list, true_digraph, max_k=10):
    node_num = true_digraph.number_of_nodes()
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    node_AP = [0.0] * node_num
    count = 0
# Bug from original authors: it needs to match to the number id of the nodes 
    for i in range(1,node_num+1):
        if true_digraph.out_degree(i) == 0:
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(node_edges[i], true_digraph, max_k)
        precision_rectified = [p * d for p,d in zip(precision_scores,delta_factors)]
        if(sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count
  
def dwnsmpl_edges(graph, seed = np.random.RandomState(123)):
  '''Downsample non-edges

  Args:
    graph (nx network):
    seed (int): 

  Raises:
    Exception: Format Problem

  Returns:
    df_dwn (pandas df): edge list
  '''
  
  # Create list of sorted edges and non-edges
  sorted_edges = sorted(graph.edges(), key = lambda x: x[0])
  sorted_nonedges = sorted(list(nx.non_edges(graph)), key = lambda x: x[0] )

  # Build labels and ids
  labels_list = list(np.ones(len(sorted_edges),dtype = int)) + list(np.zeros(len(sorted_nonedges), dtype = int))
  nodes_tupple_l = []
  for u, v in sorted_edges + sorted_nonedges:
    nodes_tupple_l.append(str(u)+'_'+str(v))
    
  # Build the dataset
  df = pd.DataFrame(columns = ['id', 'label'])
  df['id'] = nodes_tupple_l
  df['label'] = labels_list

  # Create Class
  i_class0 = np.where(df['label'] == 0)[0]
  i_class1 = np.where(df['label'] == 1)[0]

  # For every observation of class 1, randomly sample from class 0 without replacement
  i_class0_downsampled = seed.choice(i_class0, size = len(i_class1), replace=False)

  # Join together class 0's target vector with the downsampled class 1's target vector
  df_dwn = pd.concat([df.iloc[i_class1], df.iloc[i_class0_downsampled]]) 

  # Return Edge List  
  return df_dwn

def randomsplit_graph(graph, partition, seed = np.random.RandomState(123)):
  '''Split Randomly Graph into Training and Test

  Args:
    graph (nx network): Graph to be split
    partition (float): Size (%) of Test Set in Edges
    seed (int): Seed for Reproducibility

  Raises:
    Exception: Format Problem

  Returns:
    df_dwn (pandas df): edge list
  '''
  if not isinstance(graph, nx.Graph):
    raise Exception('first argument must be a nx Graph. The type of argument is {}'.format(type(graph)))
  
  # Create list of sorted edges and non-edgnes
  sorted_edges = sorted(graph.edges(), key = lambda x: x[0])
  
  # Split Training and Test by using edge list
  train, test = train_test_split(sorted_edges, test_size = partition, random_state=seed)
  
  graph_train = nx.Graph(train)
  graph_test = nx.Graph(test)
    
  return (graph_train, graph_test)