import pandas as pd
from torch.utils.data import Dataset

from MLP import *


# Load in the train data
df = pd.read_csv('ProcessedData/train_data_full.csv')
print(df)
print(df.columns)

std_cols = [c for c in df.columns if 'std' in c]
df[std_cols] = df[std_cols].fillna(0)

has_any_nan = df.isna().any()
cols_with_nan = has_any_nan[has_any_nan].index.tolist() # Adds 0 to the std that is NaN as a result of having only 1 transaction
# print(cols_with_nan)

class NewDataset(Dataset):
    def __init__(self, x_one_hot, x_s_2_count, x_cat, x_num, y):
        self.x_one_hot = x_one_hot
        self.x_s_2_count = x_s_2_count
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_one_hot[idx], self.x_s_2_count[idx], self.x_cat[idx], self.x_num[idx], self.y[idx]