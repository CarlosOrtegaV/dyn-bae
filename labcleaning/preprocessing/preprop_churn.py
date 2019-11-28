# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:59:24 2019

@author: orteg
"""
import click
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from preprop_utils import read_processed_data

def create_pipeline(df):
  # Pipeline: Scaling all continous variables
  steps = [('scaler', preprocessing.StandardScaler())]
  pipeline = Pipeline(steps)
  pipeline.fit(df)
  return pipeline

def preprocess(df):
  # TODO: Create a general case for recognizing numerical cont vars
  cols_cont = [col for col in df if len(df[col].dropna().unique()) >= 6]
  # TODO: Create a general case for categorical variables into one-hot or other
  
  # Eliminate redundant variables
  df_cat = df.drop(columns = cols_cont)
  
  # Standardize continuous variables
  df_cont = df[cols_cont]
  pipe = create_pipeline(df_cont)
  df_cont_std = pipe.transform(df_cont)
  df_cont_std = pd.DataFrame(df_cont_std, columns=df_cont.columns)
  dfs_trans = [df_cont_std.reset_index(drop=True), df_cat.reset_index(drop=True)]
  df_trans = pd.concat(dfs_trans, axis = 1)
  
  return df_trans
    

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, excel):
  
  print('Preprocessing Churn data')
  
  df = read_processed_data(input_file)
  df_trans = preprocess(df)
  
  # Send files as Pickles
  df_trans.to_pickle(output_file)
  
  # Send files as Excel Files if Excel == True
  if excel:
    df_trans.to_excel(excel, index = False)  

if __name__ == '__main__':
  main()