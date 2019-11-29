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
import numpy as np
   
###################################################################

@click.command()
@click.argument('input_file', 
                type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_directory', 
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--generate_dynamic', is_flag=True)
@click.option('--training_size', default=26, required = True)
@click.option('--interm_directory', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True))

def main(input_file, output_directory, generate_dynamic, training_size, interm_directory):

  print('Preprocessing Reality Call Graph Data')
  
  Gdf_cal = pd.read_csv(input_file, header = None)
  Gdf_cal.columns = ['node1','node2','timestamp','var1']
  
  ## Replace node ids for a simpler sequence
  G_cal = nx.from_pandas_edgelist(Gdf_cal, 'node1', 'node2')
  dict_nodes_cal = dict(zip(G_cal.nodes(),list(range(len(G_cal.nodes()))))) 
  G_cal = nx.relabel_nodes(G_cal, dict_nodes_cal)
  
  ## Replace nodes for a pandas index
  Gdf_cal['node1'] = Gdf_cal['node1'].map(dict_nodes_cal)
  Gdf_cal['node2'] = Gdf_cal['node2'].map(dict_nodes_cal)

  # Create a time variables  
  f_utc = '%Y-%m-%d %H:%M:%S'
  Gdf_aux = [datetime.utcfromtimestamp(Gdf_cal.timestamp[i]).strftime(f_utc) \
             for i in range(0,len(Gdf_cal))]
  
  df_time = pd.DataFrame({
                        'year': [Gdf_aux[i][0:4] for i in range(0,len(Gdf_aux))],
                        'month_day': [Gdf_aux[i][5:10] for i in range(0,len(Gdf_aux))],
                        'month': [Gdf_aux[i][5:7] for i in range(0,len(Gdf_aux))],
                        'iso_week': [datetime.isocalendar(datetime.utcfromtimestamp(Gdf_cal.timestamp[i]))[1] for i in range(0,len(Gdf_aux))],
                        'day': [Gdf_aux[i][8:10] for i in range(0,len(Gdf_aux))]
                        })
  Gdf_cal = pd.concat([df_time, Gdf_cal], axis = 1)
  
  # Change iso_week == 1 to 54 because calls happen in next year
  Gdf_cal['iso_biweek'] = Gdf_cal['iso_week'].replace(1,54)
  Gdf_cal['iso_biweek'] = np.ceil(Gdf_cal['iso_biweek']/2)
  
  #### CREATION OF DYNAMIC GRAPHS TIME SERIES ####
  
  ### STATIC NETWORKS ###
  train_limit = training_size

  if generate_dynamic == True:

  ### DYNAMIC NETWORKS ###

  # Create For Dynamic Setting  
    G_dy_train_cal, Gsub_dy_test_cal = splitgraph(Gdf_cal, 
                                                ('iso_biweek', train_limit), 
                                                dynamic = True,
                                                unseen_nodes_test = True)
    
    ## Export Dynamic Networks ##
  
    ## Export Graphs as Pickle
    nx.write_gpickle(G_dy_train_cal,output_directory+'G_dy_train_cal')
    nx.write_gpickle(Gsub_dy_test_cal,output_directory+'Gsub_dy_test_cal')
    
  else:
    # Create for Static Setting
    G_st_train_cal, Gsub_st_test_cal = splitgraph(Gdf_cal, 
                                                ('iso_biweek', train_limit),
                                                dynamic = False,
                                                unseen_nodes_test = True)
    ## Export Static Networks ##
    # Static Graphs
    nx.write_gpickle(G_st_train_cal,output_directory+'G_st_train_cal')
    nx.write_gpickle(Gsub_st_test_cal,output_directory+'Gsub_st_test_cal')
  
  ### EXPORTING CSV Files ###
  
  ## Export CSV
  cal_train = Gdf_cal[Gdf_cal['iso_biweek']<=train_limit]
  cal_test = Gdf_cal[Gdf_cal['iso_biweek']>train_limit]
  
  cal_train.to_csv(interm_directory+'train.csv', index = False)
  cal_test.to_csv(interm_directory+'test.csv', index = False) 
    
if __name__ == '__main__':
  main()