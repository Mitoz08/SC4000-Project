import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split 
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
file_path_1 = "./train_data/train.parquet"
df = pd.read_parquet(file_path_1)

print(f"Shape before including target column {df.shape}")
print(df.head(5))


#Features in LightGBM
features = [col for col in df.columns if col not in ["customer_ID", "target"]]
print(f"Number of features is {len(features)}")
X = df[features]
print(f"Shape of X: {X.shape}")


# #Target labels for training 
y = df["target"] 
print(f"Shape of y: {y.shape}")
# print(f"The first 5 entires of y is {y.head(5)}")


print(f"Training final model on {255} bin and {64} number of leaves")
params = {
    "objective" : "binary",
    "metric" : "binary_logloss",
    "boosting" : "dart",
    "num_iterations" : 6000,
    "learning_rate" : 0.025,
    "num_leaves" : 64,                  
    "device_type" : "gpu",
    "seed" : 62,
    "max_depth" : 70,
    "min_data_in_leaf" : 128,
    "lambda_l1" : 0.1,
    "lambda_l2" : 30,
    "bagging_fraction" : 0.70,
    "bagging_freq" : 5,
    "feature_fraction" : 0.3,
    "max_bin" : 255,
    "min_data_in_bin" : 128,    
}
lgb_train_data = lgb.Dataset(data=X, label=y)
eval_results = lgb.cv(params = params, train_set = lgb_train_data, nfold =5, stratified=True, shuffle=True,seed=62, callbacks=[lgb.log_evaluation(50)], return_cvbooster=True)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, stratify=y)
cvboosters = eval_results["cvbooster"]
for i,booster in enumerate(cvboosters.boosters):
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    AmexMetric = Metric(y_val, val_pred)
    print(f"The AmexMetric for the {i+1}-th fold is {AmexMetric}")
    print(f"Saving model now...")
    booster.save_model(f"./modelsNew62_3_0.3/fold_{i}.txt")


