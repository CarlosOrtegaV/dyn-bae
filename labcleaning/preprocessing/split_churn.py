# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:59:24 2019

@author: orteg
"""
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprop_utils import read_raw_data

def split_data(df):
  # Shuffle data to avoid order in churn label
  df = df.sample(frac=1,random_state=np.random.RandomState(123)).reset_index(drop=True)
  df = df.drop(columns = ['ID', 'START_DATE', 'END_DATE'])
  dfy = df['CHURN']
  dfx = df.drop(columns = ['CHURN'])
  dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(dfx, dfy, test_size=0.30, random_state=123)
  df_train = pd.concat([dfx_train.reset_index(drop=True), dfy_train.reset_index(drop=True)], axis=1)
  df_test = pd.concat([dfx_test.reset_index(drop=True), dfy_test.reset_index(drop=True)], axis=1)
  return df_train, df_test

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--excel', is_flag=True)

def main(input_file, output_directory, excel):
  print('Splitting Churn data into Training and Test Set')
  
  df = read_raw_data(input_file, desc_sorted_by = 'ID')
  df_train, df_test = split_data(df)
  output_train = output_directory+'train.pkl'
  output_test = output_directory+'test.pkl'
  
  # Send files as Pickles
  df_train.to_pickle(output_train)
  df_test.to_pickle(output_test)
  
  # Send files as Excel Files if Excel == True
  if excel:
    df_train.to_excel(output_directory+'train.xlsx', index = False)
    df_test.to_excel(output_directory+'test.xlsx', index = False)  

if __name__ == '__main__':
  main()