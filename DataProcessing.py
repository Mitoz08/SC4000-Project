import numpy as np, pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import os
import gc

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
        print(X[0])
        X = np.array(X, dtype=float)
        return np.floor(X * 100.0) / 100.0

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



import pandas as pd
import numpy as np

def denoise(df):
    df['D_63'] = df['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)
    for col in df.columns:
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100)
    return df

def one_hot_encoding(df,cols,is_drop=True):
    for col in cols:
        print('one hot encoding:',col)
        dummies = pd.get_dummies(pd.Series(df[col]),prefix='oneHot_%s'%col)
        df = pd.concat([df,dummies],axis=1)
    if is_drop:
        df.drop(cols,axis=1,inplace=True)
    return df

def cat_feature(df):
    one_hot_features = [col for col in df.columns if 'oneHot' in col]
    num_agg_df = df.groupby("customer_ID",sort=False)[one_hot_features].agg(['mean', 'std', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    cat_agg_df = df.groupby("customer_ID",sort=False)[cat_features].agg(['nunique'])
    cat_agg_df.columns = ['_'.join(x) for x in cat_agg_df.columns]

    count_agg_df = df.groupby("customer_ID",sort=False)[['S_2']].agg(['count'])
    count_agg_df.columns = ['_'.join(x) for x in count_agg_df.columns]
    df = pd.concat([num_agg_df, cat_agg_df,count_agg_df], axis=1).reset_index()
    print('cat feature shape after engineering', df.shape )

    return df

def num_feature(df):
    num_agg_df = df.groupby("customer_ID",sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
    for col in num_agg_df.columns:
        num_agg_df[col] = num_agg_df[col] // 0.01
    df = num_agg_df.reset_index()
    print('num feature shape after engineering', df.shape )

    return df



for i in range(5):
    print(f"Converting file {i}")
    df = pd.read_csv(f"SplitData/train_data_{i}.csv")

    # print(df.shape)
    columns = df.columns

    # print(columns)

    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    num_features = [col for col in columns if (col not in cat_features and col not in ["customer_ID", "S_2"])]

    # print(num_features)

    df = replace_nulls_with_mean(df)
    df = denoise(df)
    df = one_hot_encoding(df,cat_features,False)
    cat_df = cat_feature(df)
    num_df = num_feature(df)
    # print(cat_df.shape)
    # print(num_df.shape)
    #
    # print(cat_df.columns)
    # print(num_df.columns)

    merged_df = pd.merge(cat_df, num_df, on=["customer_ID"], how="left")

    merged_df.to_csv(f"SplitData/train_data_{i}.csv", index=False)

    del merged_df
    gc.collect()

    # print(merged_df.shape)



