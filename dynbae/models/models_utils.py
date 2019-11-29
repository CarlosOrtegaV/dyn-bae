# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:15 2019

@author: orteg
"""
from sklearn.model_selection import ParameterGrid, ParameterSampler
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynSDNE        import DynSDNE  
import node2vec
import numpy as np

from time import time

class Hyperconfig:
  def __init__(self, type_search, params):
    self._type_search = type_search
    self._grid = []
    self._params = params
    self._superembs = []
    
    if (self._type_search == 'grid_search') or (self._type_search == 'random_search') or (self._type_search == 'bayesian_search'):
      pass
    else:
      raise ValueError('Choose an appropriate option!')
      
    if not isinstance(self._params, dict):
      raise Exception('first argument must be a dictionary. The type of argument is {}'.format(type(self._params))) 
  
    
  
  def get_grid(self, number_iter = 10, random_seed = 123):
    if self._type_search == 'grid_search':
      self._grid = list(ParameterGrid(self._params))
    if self._type_search == 'random_search':
      self._grid = list(ParameterSampler(self._params, number_iter, random_seed))
  
  
  def fit_loop_node2vec(self, graphs_train, dim = 128):
    length_train = len(graphs_train)
    
    for config in self._grid:
      print(config)
      t = time()
      embs = []

      for t in range(length_train):
        model = node2vec.Node2Vec(graphs_train[t], dimensions = dim, walk_length=30, 
                                num_walks = 10, p=config['p'], q=config['q'], 
                                workers=1)
        fitted = model.fit(window = 10, min_count = 1)
        aux_embs = np.zeros((len(graphs_train[t]), dim), dtype=np.float64)
        set_nodes = graphs_train[t].nodes()
        
        i = 0
        for j in set_nodes:
          j = str(j) 
          aux_embs[i, :] = fitted.wv.get_vector(j)
          i += 1
        embs.append(aux_embs)
      self._superembs.append(embs)
    
    
  def fit_loop_ae(self, graphs_train, dim = 128):
    length_train = len(graphs_train)
    for config in self._grid:
      print(config)
      t = time()
      embs = []
      model = AE(d=dim, alpha = config['alpha'], beta=config['beta'], nu1=config['nu1'], 
                 nu2=config['nu2'], K=3, n_units=[500, 300, ], n_iter=200,
                 xeta=1e-4, n_batch=100)
      
      for t in range(length_train):
        emb, _ = model.learn_embeddings(graphs_train[t])
        embs.append(emb)
        print(model._method_name + ':\n\tTraining time: %f' % (time() - t))
      self._superembs.append(embs)
      
  def fit_loop_dynae(self, graphs_train, dim = 128):
    length_train = len(graphs_train)
    for config in self._grid:
      print(config)
      t = time()
      embs = []
      model = DynAE(d=dim, beta=config['beta'], nu1=config['nu1'], nu2=config['nu2'], 
                    K=3, rho = 0.3, n_prev_graphs = config['lookback'], 
                    n_units=[500, 300, ], n_iter=250, xeta=1e-4, n_batch=100)
      
      
      for temp_var in range(config['lookback'] + 1, length_train + 1):
        emb, _ = model.learn_embeddings(graphs_train[:temp_var])
        embs.append(emb)
        print(model._method_name + ':\n\tTraining time: %f' % (time() - t))
      self._superembs.append(embs)
      
  def fit_loop_dyngem(self, graphs_train, dim = 128):
      length_train = len(graphs_train)
      for config in self._grid:
        print(config)
        t = time()
        embs = []
        model = DynSDNE(d=dim, alpha = config['alpha'], beta=config['beta'], 
                        nu1=config['nu1'], nu2=config['nu2'], K=3, rho = 0.3, 
                      n_units=[500, 300, ], n_iter=250, xeta=1e-4, n_batch=100)
        
        
        for t in range(length_train):
          emb, _ = model.learn_embedding(graphs_train[t])
          embs.append(emb)
          print(model._method_name + ':\n\tTraining time: %f' % (time() - t))
        self._superembs.append(embs)
