# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:15 2019

@author: orteg
"""
import pandas as pd

def rus(df, thresh, rand_state, ratio):
  outlier_df = df[df['anomaly_score'] <= thresh]
  inlier_df = df[df['anomaly_score'] >= thresh]
  
  outlier_df = outlier_df.sample(frac = ratio, random_state = rand_state)
  outlier_df = outlier_df.reset_index(drop=True)
  
  dfs = [outlier_df.reset_index(drop=True), inlier_df.reset_index(drop=True)]
  df = pd.concat(dfs)
  return df

def read_processed_data(fname):
  dframe = pd.read_pickle(fname)
  return dframe