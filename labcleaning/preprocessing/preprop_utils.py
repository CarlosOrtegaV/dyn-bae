# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:02:43 2019

@author: orteg
"""
import pandas as pd

def read_raw_data(fname, desc_sorted_by = None):
  df = pd.read_csv(fname, sep = ';')
  if desc_sorted_by is not None:
    df = df.sort_values(by=[desc_sorted_by])
  return df


def read_processed_data(fname):
  dframe = pd.read_pickle(fname)
  return dframe