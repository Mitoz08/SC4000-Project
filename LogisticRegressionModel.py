import numpy as np, pandas as pd
from datetime import datetime

from numpy.ma.core import shape
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import DataProcessing as dp


# Combine all chunks into one DataFrame
merged_df = pd.read_csv("SplitData/train_data_0.csv")

# print(train_df.shape)
# print(label_df.shape)

# merged_df = pd.merge(train_df, label_df, on="customer_ID")
# merged_df = dp.replace_nulls_with_mean(merged_df)
# print(merged_df.shape)

merged_df['S_2'] = pd.to_datetime(merged_df['S_2'])
merged_df = merged_df.drop(columns=['customer_ID'])
df = merged_df

train_val_mask = (df['S_2'] >= "2017-03-01") & (df['S_2'] <= "2017-11-30")
test_mask = (df['S_2'] >= "2017-12-01") & (df['S_2'] <= "2018-03-31")

num_cols = [c for c in df.columns if c not in ['S_2','target']]
X = df[num_cols].copy()
y = df['target'].copy()

X_trainval, y_trainval = X[train_val_mask], y[train_val_mask]
X_test, y_test = X[test_mask], y[test_mask]


# Preprocessing pipeline
preprocess = Pipeline(steps=[
    ("impute0", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("denoise", dp.DenoiseFloor100()),
    # ("feat_add", FeatureAdder(lags=(1,), roll_windows=(3,))),  # tweak as needed
    ("corr_drop", dp.CorrelationFilter(threshold=0.95)),
    ("standardize", StandardScaler(with_mean=True, with_std=True)),
])


# 5-fold time-series CV on train+val window
tscv = TimeSeriesSplit(n_splits=5)
folds = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_trainval), start=1):
    X_tr_raw, X_val_raw = X_trainval.iloc[tr_idx], X_trainval.iloc[val_idx]
    y_tr, y_val = y_trainval.iloc[tr_idx], y_trainval.iloc[val_idx]

    # fit pipeline on training split only; transform both train & val
    pipe = preprocess.fit(X_tr_raw, y_tr)
    X_tr  = pipe.transform(X_tr_raw)
    X_val = pipe.transform(X_val_raw)

    folds.append({
        "fold": fold,
        "X_train": X_tr, "y_train": y_tr.to_numpy(),
        "X_val":   X_val, "y_val":  y_val.to_numpy(),
        "fitted_pipe": pipe,  # keep if you want per-fold models
    })
    print(f"Fold {fold}: train={X_tr.shape}, val={X_val.shape}")

# Final fit on full train+val and transform test
final_pipe = preprocess.fit(X_trainval, y_trainval)
X_trainval_proc = final_pipe.transform(X_trainval)
X_test_proc = final_pipe.transform(X_test)
# print("Train+Val processed:", X_trainval_proc.shape)
# print("Test processed:", X_test_proc.shape)

model = LinearRegression()
model.fit(X_trainval_proc, y_trainval)

# Predict on both train and test sets
y_train_pred = model.predict(X_trainval_proc)
y_test_pred  = model.predict(X_test_proc)

# Evaluate performance
train_mse = mean_squared_error(y_trainval, y_train_pred)
test_mse  = mean_squared_error(y_test, y_test_pred)
train_r2  = r2_score(y_trainval, y_train_pred)
test_r2   = r2_score(y_test, y_test_pred)

print("\nModel performance:")
print(f"Train MSE: {train_mse:.4f},  R²: {train_r2:.4f}")
print(f"Test  MSE: {test_mse:.4f},  R²: {test_r2:.4f}")
