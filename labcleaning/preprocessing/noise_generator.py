# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:59:24 2019

@author: orteg
"""
import click
import pandas as pd
import numpy as np
from preprop_utils import read_raw_data

def noise_generator(df, noise_level):
  df_pos = df[df['CHURN'] == 1].copy()
  df_neg = df[df['CHURN'] == 0].copy()
  
  # Create Noise
  np.random.seed(123)
  noise = np.random.binomial(1, noise_level, len(df_pos))
  
  # TODO: Include explicitly asymmetric noise
  df_pos['noisy_label'] = df_pos['CHURN'] - noise
  df_neg['noisy_label'] = df_neg['CHURN']
  
  dfs = [df_pos.reset_index(drop = True), df_neg.reset_index(drop = True)]
  df_noisy = pd.concat(dfs)
  df_noisy = df_noisy.sample(frac=1).reset_index(drop=True)
  
  return df_noisy

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--noise_level', type=click.FloatRange(min = 0, max = 1), default=0.2)
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))

def main(input_file, output_file, noise_level, excel):
  print('Generating Noise in Churn data')
  
  df = read_raw_data(input_file)
  df_noisy = noise_generator(df, noise_level)
  
   # Send files as CSV
  df_noisy.to_csv(output_file, sep = ';', index = False)
  
  # Send files as Excel Files if Excel == True
  if excel:
    df_noisy.to_excel(excel, index = False)
    
if __name__ == '__main__':
  main()
