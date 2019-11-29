# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:30:11 2019

@author: orteg
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:59:24 2019

@author: orteg
"""
import click
import networkx as nx
from preprop_utils import randomsplit_graph, dwnsmpl_edges, nodeid_split

  
###################################################################

@click.command()
@click.argument('dynamic_graph', 
                type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_directory', 
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--generate_dynamic', is_flag=True, required = True)
@click.option('--static_graph', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True))

def main(dynamic_graph, output_directory, generate_dynamic, static_graph):
  
  if generate_dynamic:
  ########################
  ### DYNAMIC NETWORKS ###
  ########################
  
    Gsub_dy_test = nx.read_gpickle(dynamic_graph)
    
    ## Intrapolation Setting
    
    # Random split on test graph
    Gsub_dy_intratrain, Gsub_dy_intratest = randomsplit_graph(Gsub_dy_test, 0.3)
  
    # Downsampling the non-edges
    pd_dwn_sub_dy_intratrain = dwnsmpl_edges(Gsub_dy_intratrain)
    pd_dwn_sub_dy_intratest = dwnsmpl_edges(Gsub_dy_intratest)
    
    # Splitting id into Node1 and Node2 on training nodes of test subgraph
    pd_dwn_sub_dy_intratrain['node1'] = pd_dwn_sub_dy_intratrain.apply(lambda row: nodeid_split(row,0), axis = 1)
    pd_dwn_sub_dy_intratrain['node2'] = pd_dwn_sub_dy_intratrain.apply(lambda row: nodeid_split(row,1), axis = 1)
    
    # Splitting id into node1 and node2 on test nodes of test subgraph
    pd_dwn_sub_dy_intratest['node1'] = pd_dwn_sub_dy_intratest.apply(lambda row: nodeid_split(row,0), axis = 1)
    pd_dwn_sub_dy_intratest['node2'] = pd_dwn_sub_dy_intratest.apply(lambda row: nodeid_split(row,1), axis = 1)
    
    # Export Files: Edge list into CSV Files
    pd_dwn_sub_dy_intratrain.to_csv(output_directory+'pd_dwn_sub_dy_intratrain.csv', index = False)
    pd_dwn_sub_dy_intratest.to_csv(output_directory+'pd_dwn_sub_dy_intratest.csv', index = False)
    
    ## Extrapolation Setting
    
    # Downsampling into Pandas Edge List
    pd_dwn_sub_dy_extratest = dwnsmpl_edges(Gsub_dy_test)
    
    # Splitting id into Node1 and Node2 on
    pd_dwn_sub_dy_extratest['node1'] = pd_dwn_sub_dy_extratest.apply(lambda row: nodeid_split(row,0), axis = 1)
    pd_dwn_sub_dy_extratest['node2'] = pd_dwn_sub_dy_extratest.apply(lambda row: nodeid_split(row,1), axis = 1)
    
    # Export Files: Edge list into CSV Files
    pd_dwn_sub_dy_extratest.to_csv(output_directory+'pd_dwn_sub_dy_extratest.csv', index = False)
    
  else:
  #######################
  ### STATIC NETWORKS ###
  #######################
    Gsub_st_test = nx.read_gpickle("Gsub_st_test")
  
    ### STATIC NETWORKS ###
    
    ## Intrapolation Setting
    
    # Random split on test graph
    Gsub_st_intratrain, Gsub_st_intratest = randomsplit_graph(Gsub_st_test, 0.3)
    
    # Downsampling the non-edges
    pd_dwn_sub_st_intratrain = dwnsmpl_edges(Gsub_st_intratrain)
    pd_dwn_sub_st_intratest = dwnsmpl_edges(Gsub_st_intratest)
    
    # Splitting id into Node1 and Node2 on training nodes of test subgraph
    pd_dwn_sub_st_intratrain['node1'] = pd_dwn_sub_st_intratrain.apply(lambda row: nodeid_split(row,0), axis = 1)
    pd_dwn_sub_st_intratrain['node2'] = pd_dwn_sub_st_intratrain.apply(lambda row: nodeid_split(row,1), axis = 1)
    
    # Splitting id into node1 and node2 on test nodes of test subgraph
    pd_dwn_sub_st_intratest['node1'] = pd_dwn_sub_st_intratest.apply(lambda row: nodeid_split(row,0), axis = 1)
    pd_dwn_sub_st_intratest['node2'] = pd_dwn_sub_st_intratest.apply(lambda row: nodeid_split(row,1), axis = 1)

    # Export Files: Edge list into CSV Files
    pd_dwn_sub_st_intratrain.to_csv(output_directory+'pd_dwn_sub_st_intratrain.csv', index = False)
    pd_dwn_sub_st_intratest.to_csv(output_directory+'pd_dwn_sub_st_intratest.csv', index = False)

if __name__ == '__main__':
  main()