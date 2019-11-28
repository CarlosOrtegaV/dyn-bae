  
  ### DATA
  df_tr = read_processed_data(input_file)
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