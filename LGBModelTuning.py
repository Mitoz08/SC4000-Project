import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split 
import numpy as np
import pandas as pd

#AmexMetric
def amex_metric_mod(y_true, y_pred):
    '''
    Calculates the AmexMetric which uses the Gini Coefficient, G and the default rate captured at 4%,D, 
    which is the percentage of the positive labels (defaults) captured within the highest-ranked 4% of the predictions, and represents a Sensitivity/Recall statistic
    '''
    #Calculate D
    labels     = np.transpose(np.array([y_true, y_pred]))                   #Merge y_true and y_pred into two columns
    labels     = labels[labels[:, 1].argsort()[::-1]]                       #Sort the matrix in descending order with reference to y_pred
    weights    = np.where(labels[:,0]==0, 20, 1)                            #Weight defaults(1) to 20 and no-default(0) to 1
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]  #Find cutoff value which is 4% of the total sum of the weights
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])                #Get D

    #Calculate Gini Coefficient
    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def Metric(labels,preds):
    return amex_metric_mod(labels,preds)



#Read train data into a dataframe
file_path = "./train_data/train.parquet"
df = pd.read_parquet(file_path)

print(f"Shape before including target column {df.shape}")

#Features in LightGBM
features = [col for col in df.columns if col not in ["customer_ID", "target"]]
print(f"Number of features is {len(features)}")
X = df[features]
print(f"Shape of X: {X.shape}")

#Target labels for training 
y = df["target"] 
print(f"Shape of y: {y.shape}")

#Hyper-parameters to tune
NLs = [64,128,256]
bins = [64, 128, 255]

results = [[0 for _ in range(3)] for _ in range(3)]
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state =42)

for i,bin in enumerate(bins):
    cv = []
    for j,NL in enumerate(NLs):
        print(f"Training on {bin} bin and {NL} number of leaves")
        params = {
            "objective" : "binary",
            "metric" : "binary_logloss",
            "boosting" : "dart",
            "num_iterations" : 6000,
            "learning_rate" : 0.02,
            "num_leaves" : NL,                  
            "device_type" : "gpu",
            "seed" : 42,
            "max_depth" : 70,
            "min_data_in_leaf" : 128,
            "lambda_l1" : 0.1,
            "lambda_l2" : 30,
            "bagging_fraction" : 0.7,
            "bagging_freq" : 5,
            "feature_fraction" : 0.3,
            "max_bin" : bin,
            "min_data_in_bin" : 128,    
        }
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"Training Fold {fold+1} with bin: {bin} and number of leaves: {NL} ")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            lgb_train_data = lgb.Dataset(X_train,y_train)
            lgb_val_data = lgb.Dataset(X_val,y_val)
            model = lgb.train(params = params,train_set =lgb_train_data,valid_sets = [lgb_train_data, lgb_val_data], valid_names = ["train","valid"], callbacks=[lgb.log_evaluation(50)])

            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            AmexMetric = Metric(y_val, val_pred)
            print(f"The AmexMetric for the {fold+1}-th fold is {AmexMetric}")
            cv.append(AmexMetric)

        print(f"The average AmexMetric for the 5-cross validation is {np.mean(cv)}")
        results[i][j] = np.mean(cv)

print(results)

#Use 255 bins and 64 number of leaves used in final model
