
# %%
import pandas as pd
import xgboost as xgb
import sklearn
import dask.distributed
#import dask_ml.model_selection as dcv
import time
import pyodbc


#%%
from sklearn.model_selection import train_test_split
#from dask_ml.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from dask.distributed import Client, LocalCluster
from dask import array as da
from joblib import parallel_backend
from dask.diagnostics import ProgressBar
from dask.distributed import progress
ProgressBar().register()
#%%
import warnings 
warnings.filterwarnings('ignore')
warnings.simplefilter(action = 'ignore', category=FutureWarning)

#%%
test_sql = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=127.0.0.1, 1433;"
                      "Database=DB1;"
                      "UID=user;"
                      "PWD=123;")

# %%
query = "SELECT [col1], [col2], [col3], [col4], [col5], [col6], [f1], [col8], [Frequency] FROM Sheet1$"

# %%
spindle = pd.read_sql_query(query, test_sql)

#%%
spindle_main = spindle[spindle["col1"] == "main"]
#%%
spindle_main
# %%
spindle_main.head()
# %%
data_1_updated = spindle_main.dropna(axis = 0, how = 'any', inplace = False)
data_1_updated

#%%
data_1_updated["overlap 34"] = data_1_updated["col4"] + data_1_updated["col5"]
data_1_updated
#%%
df = data_1_updated.groupby(["col2", "col3", "f1", "overlap 34", "Frequency"]).sum().reset_index()
#%%
df.head()

#%%
df['Frequency'].plot(kind = 'hist')
#%%
df['Frequency'].describe()

#%%
q = pd.qcut(df['Frequency'], q=4)

#%%
df['quantile_ex_1'] = pd.qcut(df['Frequency'], q=4)
df['quantile_ex_2'] = pd.qcut(df['Frequency'], q=10, precision=0)
#%%
df

#%%
df['quantile_ex_1'].value_counts()
#%%
df['quantile_ex_2'].value_counts()
#%%
bin_labels_7 = ['a', 'b', 'c', 'd', 'e']
df['quantile_ex_3'] = pd.qcut(df['Frequency'],
                              q=[0, .2, .4, .6,  .8, 1],
                              labels=bin_labels_7)
df.head()
#%%
df['quantile_ex_3'].value_counts()
# %%
df['quantile_ex_3']
# %%
a = df[df['quantile_ex_3'] == "a"]
a
# %%
data_1_a = a[['col2', 'col3', 'f1', 'overlap 34', 'col6', 'Frequency' ]]
data_1_a
# %%
X_a, y_a = data_1_a.iloc[:, :-1], data_1_a.iloc[:, -1]
X_a
# %%
y_a
# %%
X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size = 0.3, shuffle = False)
# %%
X_a_train
# %%
X_a_test
# %%
y_a_train
#%%
y_a_test
# %%
# %%
b = df[df['quantile_ex_3'] == "b"]
b
# %%
data_1_b = b[['col2', 'col3', 'f1', 'overlap 34', 'col6', 'Frequency' ]]
data_1_b
# %%
X_b, y_b = data_1_b.iloc[:, :-1], data_1_b.iloc[:, -1]
X_b
# %%
y_b
# %%
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size = 0.3, shuffle = False)
# %%
X_b_train
# %%
X_b_test
# %%
y_b_train
#%%
y_b_test

# %%
c = df[df['quantile_ex_3'] == "c"]
c
# %%
data_1_c = c[['col2', 'col3', 'f1', 'overlap 34', 'col6', 'Frequency' ]]
data_1_c
# %%
X_c, y_c = data_1_c.iloc[:, :-1], data_1_c.iloc[:, -1]
X_c
# %%
y_c
# %%
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size = 0.3, shuffle = False)
# %%
X_c_train
# %%
X_c_test
# %%
y_c_train
#%%
y_c_test

# %%
d = df[df['quantile_ex_3'] == "d"]
d
# %%
data_1_d = d[['col2', 'col3', 'f1', 'overlap 34', 'col6', 'Frequency' ]]
data_1_d
# %%
X_d, y_d = data_1_d.iloc[:, :-1], data_1_d.iloc[:, -1]
X_d
# %%
y_d
# %%
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size = 0.3, shuffle = False)
# %%
X_d_train
# %%
X_d_test
# %%
y_d_train
#%%
y_d_test
# %%
e = df[df['quantile_ex_3'] == "e"]
e
# %%
data_1_e = e[['col2', 'col3', 'f1', 'overlap 34', 'col6', 'Frequency' ]]
data_1_e
# %%
X_e, y_e = data_1_e.iloc[:, :-1], data_1_e.iloc[:, -1]
X_e
# %%
y_e
# %%
X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(X_e, y_e, test_size = 0.3, shuffle = False)
# %%
X_e_train
# %%
X_e_test
# %%
y_e_train
#%%
y_e_test
# %%
X_train = pd.concat([X_a_train, X_b_train, X_c_train, X_d_train, X_e_train], ignore_index= True)
X_train
# %%
X_test = pd.concat([X_a_test, X_b_test, X_c_test, X_d_test, X_e_test], ignore_index=True)
X_test
# %%
y_train = pd.concat([y_a_train, y_b_train, y_c_train, y_d_train, y_e_train], ignore_index=True)
y_train
# %%
y_test = pd.concat([y_a_test, y_b_test, y_c_test, y_d_test, y_e_test], ignore_index=True)
y_test
# %%
X_test.to_csv("X_test_bin.csv")
# %%
X_train.to_csv("X_train_bin.csv")
# %%
y_train.to_csv("y_train_bin.csv")
# %%
y_test.to_csv("y_test_bin.csv")
# %%
data_1_all = pd.concat([data_1_a, data_1_b, data_1_c, data_1_d, data_1_e], ignore_index=True)

data_1_all.to_csv("data_1_bin.csv")
# %%
data_1_all
# %%
