from DataProcessing import *

label_df = pd.read_csv("Data/train_labels.csv")
train_df = pd.read_csv("Data/train_data.csv")

merged_df = pd.merge(train_df, label_df, on="customer_ID")
merged_df = replace_nulls_with_mean(merged_df)
# print(merged_df.shape)

merged_df['S_2'] = pd.to_datetime(merged_df['S_2'])
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
    ("denoise", DenoiseFloor100()),
    # ("feat_add", FeatureAdder(lags=(1,), roll_windows=(3,))),  # tweak as needed
    ("corr_drop", CorrelationFilter(threshold=0.95)),
    ("standardize", StandardScaler(with_mean=True, with_std=True)),
])



