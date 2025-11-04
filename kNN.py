import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split



# Combine all chunks into one DataFrame


# train_df = pd.read_csv("Data/train_data.csv")
# label_df = pd.read_csv("Data/train_labels.csv")
# merged_df = pd.merge(train_df, label_df, on="customer_ID")


# merged_data.csv is the train data merged with train labels on customer_ID
# merged_df = pd.read_csv("Data/merged_data.csv")
merged_df = pd.read_csv("SplitData1/train_data_2.csv")

merged_df = merged_df.drop(columns=['customer_ID'])
df = merged_df


num_cols = [c for c in df.columns if c not in ['S_2','target']]
X = df[num_cols].copy()
y = df['target'].copy()


# Split into trainval and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_trainval = X_trainval.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_trainval = y_trainval.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Imputing the data
imputer = SimpleImputer(missing_values = np.nan, 
                        strategy ='constant', 
                        fill_value = 0.0)

imputer = imputer.fit(X_trainval)
X_trainval = imputer.transform(X_trainval)

imputer = imputer.fit(X_test)
X_test = imputer.transform(X_test)


# Scaling for kNN
sc = StandardScaler()
X_trainval = sc.fit_transform(X_trainval)
X_test = sc.fit_transform(X_test)


# Defining Amex Metric
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
        gini[i]= np.sum((lorentz - weight_random) * weight)

    # print(top_four)
    # print(gini)

    return 0.5 * (gini[1]/gini[0] + top_four), top_four, gini

def Metric(labels,preds):
    return amex_metric_mod(labels,preds)



# kNN with PCA and StratifiedKFold

# The two hyperparams variance percentage (in pca_vals) and no. of neighbors
pca_vals = [0.25, 0.5, 0.7, 0.9, 0.95]
n_neighbors = [20, 50, 70, 100, 120]
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X_trainval, y_trainval)):
    
    print(f"Fold {i}:")
    X_train = X_trainval[train_index].copy()
    y_train = y_trainval[train_index].copy()
    X_val = X_trainval[test_index].copy()
    y_val = y_trainval[test_index].copy()

    pca = PCA(n_components = pca_vals[i])
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    for j in range(len(n_neighbors)):
        classifier = KNeighborsClassifier(n_neighbors = n_neighbors[j])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        amex_metric, top_four, gini = amex_metric_mod(y_val, y_pred)
        print(f"Fold {i}, n_neighbors={n_neighbors[j]}: Accuracy = {accuracy}, AmexMetric = {amex_metric}, top_four = {top_four}, gini = {gini}")

   

    
# Final evaluation on test set



