# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:43:07 2019

@author: orteg
"""
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from models_utils import rus, read_processed_data

class LabelCleaner():
  def __init__(self, ad_technique = 'iso_for', labcleaner = 'rus', **kwargs):
    self._ad_technique = ad_technique
    self._labcleaner = labcleaner
    
    if (self._labcleaner not in ['relabel','rus','naive','nothing']):
      raise ValueError('Choose an appropriate option!')
    
    #
    config = {
        'num_estimators': 100,
        'rand_state': np.random.RandomState(123),
    }
    config.update(kwargs)
    for key in config.keys():
        self.__setattr__('_%s' % key, config[key])


  def anomaly_neg(self, df):

    if not isinstance(df, pd.DataFrame):
      raise TypeError('first argument must be a pandas Dataframe. The type of argument is {}'.format(type(self._params))) 
    
    
    df_neg = df[df['noisy_label'] == 0].copy()

    
    dfx_neg = df_neg.drop(columns = ['CHURN', 'noisy_label'])
    
    if self._ad_technique == 'iso_for':
      model = IsolationForest(n_estimators=self._num_estimators,
                              random_state=self._rand_state)
      
      # fit the model
      model.fit(dfx_neg)
      anomaly_score = list(model.decision_function(dfx_neg))
      
      self._ad_model = model
      df_neg['anomaly_score'] = anomaly_score
      return df_neg
    
    
  def labclean(self, df, thresh, ratio):
    if self._labcleaner == 'rus':
      df_rus = rus(df, thresh, self._rand_state, ratio)
      return df_rus
    elif self._labcleaner == 'relabel':
      pass
    elif self._labcleaner == 'naive':
      pass
    else:
      pass

    
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