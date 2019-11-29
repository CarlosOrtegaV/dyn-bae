# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:59:24 2019

@author: orteg
"""
import click
import pandas as pd
import networkx as nx
from preprop_utils import splitgraph
from datetime import datetime

   
###################################################################

@click.command()
@click.argument('input_file', 
                type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_directory', 
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--generate_dynamic', is_flag=True, required = True)
@click.option('--training_size', default=9, required = True)
@click.option('--interm_directory', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True))

def main(input_file, output_directory, interm_directory, generate_dynamic, training_size):

  print('Preprocessing Facebook Graph Data')
  
  Gdf_fb = pd.read_csv(input_file, header = None)
  Gdf_fb.columns = ['node1','node2','timestamp']
  
  ## Replace node ids for a simpler sequence
  G_fb = nx.from_pandas_edgelist(Gdf_fb, 'node1', 'node2')
  dict_nodes_fb = dict(zip(G_fb.nodes(),list(range(len(G_fb.nodes()))))) 
  G_fb = nx.relabel_nodes(G_fb, dict_nodes_fb)
  
  ## Replace nodes for a pandas index
  Gdf_fb['node1'] = Gdf_fb['node1'].map(dict_nodes_fb)
  Gdf_fb['node2'] = Gdf_fb['node2'].map(dict_nodes_fb)

  # Create a time variables  
  f_utc = '%Y-%m-%d %H:%M:%S'
  Gdf_aux = [datetime.utcfromtimestamp(Gdf_fb.timestamp[i]).strftime(f_utc) \
             for i in range(0,len(Gdf_fb))]
  
  df_time = pd.DataFrame({
                        'year': [Gdf_aux[i][0:4] for i in range(0,len(Gdf_aux))],
                        'month_day': [Gdf_aux[i][5:10] for i in range(0,len(Gdf_aux))],
                        'month': [Gdf_aux[i][5:7] for i in range(0,len(Gdf_aux))],
                        'iso_week': [datetime.isocalendar(datetime.utcfromtimestamp(Gdf_fb.timestamp[i]))[1] for i in range(0,len(Gdf_aux))],
                        'day': [Gdf_aux[i][8:10] for i in range(0,len(Gdf_aux))]
                        })
  Gdf_fb = pd.concat([df_time, Gdf_fb], axis = 1)
  
  #### CREATION OF DYNAMIC GRAPHS TIME SERIES ####
  
  ### STATIC NETWORKS ###
  train_limit = training_size

  if generate_dynamic == True:

  ### DYNAMIC NETWORKS ###

    # Create For Dynamic Setting  
    G_dy_train_fb, Gsub_dy_test_fb = splitgraph(Gdf_fb, 
                                                ('month', train_limit), 
                                                dynamic = True,
                                                unseen_nodes_test = True)
    
    ## Export Dynamic Networks ##
  
    ## Export Graphs as Pickle
    nx.write_gpickle(G_dy_train_fb,output_directory+'G_dy_train_fb')
    nx.write_gpickle(Gsub_dy_test_fb,output_directory+'Gsub_dy_test_fb')
    
  else:
    # Create for Static Setting
    G_st_train_fb, Gsub_st_test_fb = splitgraph(Gdf_fb, 
                                                ('month', train_limit),
                                                dynamic = False,
                                                unseen_nodes_test = True)
    ## Export Static Networks ##
    # Static Graphs
    nx.write_gpickle(G_st_train_fb,output_directory+'G_st_train_fb')
    nx.write_gpickle(Gsub_st_test_fb,output_directory+'Gsub_st_test_fb')
  
  ### EXPORTING CSV Files ###
  
  ## Export CSV
  fb_train = Gdf_fb[Gdf_fb['month']<=train_limit]
  fb_test = Gdf_fb[Gdf_fb['month']>train_limit]
  
  fb_train.to_csv(interm_directory+'train.csv', index = False)
  fb_test.to_csv(interm_directory+'test.csv', index = False) 
    


if __name__ == '__main__':
  main()