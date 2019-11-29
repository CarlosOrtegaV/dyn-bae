# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:43:07 2019

@author: orteg
"""
import numpy as np
import pandas as pd


## Third Party Libraries 
import networkx as nx
import os
os.path.dirname(os.path.realpath(__file__))
import pickle

## Import own functions
os.chdir("..")
from hyperparameter import Hyperconfig

#############
#### ETL ####
#############

# Import Data
os.chdir("./fb_forum")

## Read CSVs as pandas

## Dynamic Graphs
G_dy_train_fb = nx.read_gpickle("G_dy_train_fb")

###########################
#### NODE2VEC TRAINING ####
###########################

hyper_node2vec = {
        'p': [0.25,0.5,0.75,1],
        'q': [0.1, 0.5, 1, 2, 5, 10, 100]
        }

h_node2vec = Hyperconfig('grid_search', hyper_node2vec)
h_node2vec.get_grid()

h_node2vec.fit_loop_node2vec(G_dy_train_fb)

# Export
with open('embs_node2vec.pkl', 'wb') as f:
  pickle.dump({'embs': h_node2vec._superembs,'configs': h_node2vec._grid}, f)

    
#@click.command()
#@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
#@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
#@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
#
#def main(input_file, output_file, excel):
#  
#  print('Running Anomaly Detection Techniques')
#
#  df = read_processed_data(input_file)
#  m1 = LabelCleaner()
#  df_neg = m1.anomaly_neg(df)
#  
#  q = np.quantile(df_neg['anomaly_score'],0.20)
#  m1.labclean(df_neg,q,0.20)
  
#if __name__ == '__main__':
#  main()