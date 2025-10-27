import numpy as np, pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit


# replacing null values with their column average
def replace_nulls_with_mean(df):
    df = df.copy()
    df.replace("<null>", np.nan, inplace=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_val)
    return df


# Custom transformers

# makes everything round to 2 d.p
class DenoiseFloor100(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return np.floor(X.to_numpy(dtype=float)*100.0)/100.0\

# removes one of the parameters if their correlation is more than the threshold (0.95)
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.keep_cols_ = None
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        corr = Xdf.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        self.keep_cols_ = [c for c in Xdf.columns if c not in drop_cols]
        return self
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy()
        return Xdf[self.keep_cols_].to_numpy()


# class FeatureAdder(BaseEstimator, TransformerMixin):
#     """
#     Simple example: add lag-1 and 3-step rolling mean for each feature
#     (works on the provided order; for true calendar windows, pre-aggregate by date).
#     """
#     def __init__(self, lags=(1,), roll_windows=(3,)):
#         self.lags = lags
#         self.roll_windows = roll_windows
#         self.colnames_ = None
#     def fit(self, X, y=None):
#         self.colnames_ = list(pd.DataFrame(X).columns)
#         return self
#     def transform(self, X):
#         Xdf = pd.DataFrame(X).reset_index(drop=True)
#         out = Xdf.copy()
#         for lag in self.lags:
#             out[[f"{c}_lag{lag}" for c in self.colnames_]] = Xdf.shift(lag)
#         for w in self.roll_windows:
#             out[[f"{c}_rmean{w}" for c in self.colnames_]] = Xdf.rolling(w).mean()
#         # fill initial NaNs from lags/rolling with 0 (consistent with your “add 0 for imputation” rule)
#         return out.fillna(0.0).to_numpy()







