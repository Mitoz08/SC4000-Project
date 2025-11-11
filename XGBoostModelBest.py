import xgboost as xgb
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

X = X.replace([np.inf, -np.inf], np.nan)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True, stratify=y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'booster': 'gbtree',
    'device': 'cuda',
    'learning_rate': 0.005,
    'max_depth': 10,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 30,
    'subsample': 0.5,
    'colsample_bytree': 0.2,
    'sampling_method' : 'gradient_based',
    'tree_method': 'hist',
    'max_bin': 128,
    'grow_policy': 'lossguide',
    'seed' : 67,
}

xgb_model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=8000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50)
preds = xgb_model.predict(dval)
print(amex_metric_mod(y_val,preds))

file_path_1 = "./train_data/test_fe_plus_plus.parquet"
df1 = pd.read_parquet(file_path_1)
df1 = df1.replace([np.inf, -np.inf], np.nan)
print(df1.shape)

features = [col for col in df1.columns if col not in ["customer_ID"]]

df = df1[features]
new_data_dmatrix = xgb.DMatrix(df)
print(f"Shape before including target column {df.shape}")
val_pred = xgb_model.predict(new_data_dmatrix)

df_pred = df1[["customer_ID"]].copy()
df_pred["prediction"] = val_pred
df_final = df_pred[["customer_ID","prediction"]]

df_final.to_csv("submission_final_xgb_67.csv.zip", index= False, compression="zip")
print(df_final.head(100))


