# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:14:08 2019

@author: orteg
"""
### Removable imports and objects for the on-spot experimentations
fname_tr = 'C:\\Users\\orteg\\Dropbox\\1Almacen\\Computer_Science\\ML_AI\\5_PhD_Fraud_Analytics\\Noise_Label\\data\\processed\\trans_noisy_train.pkl'
fname_ts = 'C:\\Users\\orteg\\Dropbox\\1Almacen\\Computer_Science\\ML_AI\\5_PhD_Fraud_Analytics\\Noise_Label\\data\\processed\\trans_noisy_test.pkl'

####
import sys
sys.path.append('C:\\Users\\orteg\\Dropbox\\1Almacen\\Computer_Science\\ML_AI\\5_PhD_Fraud_Analytics\\Noise_Label\\src\\data')
sys.path.append('C:\\Users\\orteg\\Dropbox\\1Almacen\\Computer_Science\\ML_AI\\5_PhD_Fraud_Analytics\\Noise_Label\\src\\models')

from sklearn.ensemble import IsolationForest
import utils_data
import numpy as np
import utils_models

###



@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, excel):
  
  print('Running Anomaly Detection Techniques')
  
  ### DATA
  df_tr = utils_data.read_processed_data(input_file)
  df_ts = utils_data.read_processed_data(fname_ts)
  
  df_tr_noisy_neg = df_tr[df_tr['noisy_label'] == 0].copy()
  df_tr_noisy_neg_outliers = df_tr_noisy_neg[df_tr_noisy_neg['CHURN'] == 1].copy()
  df_tr_noisy_neg_inliers = df_tr_noisy_neg[df_tr_noisy_neg['CHURN'] == 0].copy()
  
  dfx_tr_noisy_neg = df_tr_noisy_neg.drop(columns = ['CHURN', 'noisy_label'])
  dfx_tr_noisy_neg_outliers = df_tr_noisy_neg_outliers.drop(columns = ['CHURN', 'noisy_label'])
  dfx_tr_noisy_neg_inliers = df_tr_noisy_neg_inliers.drop(columns = ['CHURN', 'noisy_label'])
  ### ANOMALY DETECTION MODELS
  
  ad_model1 = IsolationForest(n_estimators=100, random_state=np.random.RandomState(123))
  
  # fit the model
  ad_model1.fit(dfx_tr_noisy_neg)
  
  # Predict
  pred_train_label = list(ad_model1.predict(dfx_tr_noisy_neg))
  pred_train_score = list(ad_model1.decision_function(dfx_tr_noisy_neg))
  pred_train_sc_outliers = list(ad_model1.decision_function(dfx_tr_noisy_neg_outliers))
  pred_train_sc_inliers = list(ad_model1.decision_function(dfx_tr_noisy_neg_inliers))
  
  np.mean(pred_train_sc_outliers)
  np.mean(pred_train_sc_inliers)
  
  np.max(pred_train_sc_inliers)
  
  ##
  df_tr_noisy_neg['anomaly_score'] = pred_train_score
  
  thresh = np.quantile(df_tr_noisy_neg['anomaly_score'],0.15)
  rus_df_tr_noisy_neg = utils_models.semi_undersampling(df_tr_noisy_neg, thresh, 0.50 )


if __name__ == '__main__':
  main()