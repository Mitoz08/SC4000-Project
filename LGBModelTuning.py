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
file_path_1 = "./train_data/train_data.csv"
df = pd.read_csv(file_path_1)

print(f"Shape before including target column {df.shape}")

#Read train labels into a dataframe
file_path_2 = "./Data/train_labels.csv"
df_2 = pd.read_csv(file_path_2)

df = pd.merge(df, df_2, on=["customer_ID"], how="left")

print(f"Shape after including target colum {df.shape} ")

std_cols = [c for c in df.columns if 'std' in c]
df[std_cols] = df[std_cols].fillna(0)

column_dict = {
    "one_hot_col":[],
    "cat_col": [],
    "S_2_count_col": [],
    "num_col": []
}

for col in df.columns:
    if col == "customer_ID":
        continue
    if "oneHot" in col:
        column_dict["one_hot_col"].append(col)
        continue
    if "nunique" in col:
        column_dict["cat_col"].append(col)
        continue
    if "S_2_count" in col:
        column_dict["S_2_count_col"].append(col)
        continue
    if "target" in col:
        continue
    column_dict["num_col"].append(col)


#Features in LightGBM
features = [col for col in df.columns if col not in ["customer_ID", "target"]]
print(f"Number of features is {len(features)}")
X = df[features]
print(f"Shape of X: {X.shape}")
# print(f"The first 5 entries of X is {X.head(5)}")

#Target labels for training 
y = df["target"] 
print(f"Shape of y: {y.shape}")
# print(f"The first 5 entires of y is {y.head(5)}")

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
            "num_iterations" : 1000,
            "learning_rate" : 0.05,
            "num_leaves" : NL,                  
            "device_type" : "gpu",
            "seed" : 42,
            "max_depth" : 70,
            "min_data_in_leaf" : 128,
            "lambda_l1" : 0.1,
            "lambda_l2" : 30,
            "bagging_fraction" : 0.5,
            "bagging_freq" : 5,
            "feature_fraction" : 0.5,
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

#Use 64 number of leaves and 255 bins for final model
