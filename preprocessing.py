
#%%

import pandas as pd
import pyodbc
import configparser
import re

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

path = './'


def preprocessing(config_name):


    config = configparser.ConfigParser()
    config_file = path + config_name
    config.read(config_file)

    # name = config['category']['column_value of category']
    # thresh = int(config['thresh']['thresh'])
    
    DB = config['Database']['name of the database']
    f1 = config['features']['f1']
    f2 = config['features']['f2']
    f3 = config['features']['f3']
    p = config['pred_feature']['p']
    

    test_sql = pyodbc.connect(driver="{SQL Server Native Client 11.0}",
                            server="m1, 1433",
                            database=DB,
                            uid="ycm",
                            pwd="neurelli")

    query = "SELECT * FROM Sheet1$"
    Data = pd.read_sql_query(query, test_sql)
    Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    if DB == 'ycm_spindle':
        category = config['category']['column_name of category']
        category_name = config['category']['column_value of category']

        Type = config['type']['column_name of type']
        Type_name = config['type']['column_value of type']

        spec = config['spec']['column_name of spec']
        spec_name = config['spec']['column_value of spec']

        Data_cat = Data[Data[category] == category_name]
        Data_type = Data_cat[Data_cat[Type] == Type_name]
        Data_spec = Data_type[Data_type[spec] == spec_name]
        Data = Data_spec
    else:
        category = config['category']['column_value of category']
        Data = Data.dropna(axis = 0, subset = [category])
        Data = Data.drop(Data[Data.phosphorus > 0.099].index)
    if len(list(config.items('features'))) == 3:
        Data_f =Data[[f1, f2, f3, p]]
    elif len(list(config.items('features'))) == 4:
        f4 = config['features']['f4']
        Data_f = Data[[f1, f2, f3, f4, p]]
    elif len(list(config.items('features'))) == 5:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        Data_f = Data[[f1, f2, f3, f4, f5, p]]   
    elif len(list(config.items('features'))) == 6:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        f6 = config['features']['f6']
        Data_f = Data[[f1, f2, f3, f4, f5, f6, p]]     
    elif len(list(config.items('features'))) == 7:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        f6 = config['features']['f6'] 
        f7 = config['features']['f7']
        Data_f = Data[[f1, f2, f3, f4, f5, f6, f7, p]]
    elif len(list(config.items('features'))) == 8:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        f6 = config['features']['f6'] 
        f7 = config['features']['f7']
        f8 = config['features']['f8']
        Data_f = Data[[f1, f2, f3, f4, f5, f6, f7, f8, p]]
    elif len(list(config.items('features'))) == 9:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        f6 = config['features']['f6'] 
        f7 = config['features']['f7']
        f8 = config['features']['f8']
        f9 = config['features']['f9']
        Data_f = Data[[f1, f2, f3, f4, f5, f6, f7, f8, f9, p]]
    elif len(list(config.items('features'))) == 10:
        f4 = config['features']['f4']
        f5 = config['features']['f5']
        f6 = config['features']['f6'] 
        f7 = config['features']['f7']
        f8 = config['features']['f8']
        f9 = config['features']['f9']
        f10 = config['features']['f10']
        Data_f = Data[[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, p]]            
    if bool(config.items('overlap')) == True:
        o1 = config['overlap']['o1']
        o2 = config['overlap']['o2']
        Data_f['overlap'] = Data[o1] + Data[o2]
    else:
        Q1 = Data_f[p].quantile(0.25)
        Q3 = Data_f[p].quantile(0.75)
        IQR = Q3 - Q1 
        ind = Data_f[(Data_f[p] < Q1-1.5*IQR ) | (Data_f[p] > Q3+1.5*IQR)][p]
        l = ind.values.tolist()
        Data_f = Data_f[~Data_f[p].isin(l)] 
            
    col = Data_f.columns.to_list()
    df = Data_f.groupby(col).sum().reset_index()
    testing_size = float(config['testing']['train_test_split'])
    
    bin_labels = ['a', 'b', 'c', 'd', 'e']
    df['quantile'] = pd.qcut(df[p],
                                    q=[0, .2, .4, .6,  .8, 1],
                                    labels=bin_labels)

    a = df[df['quantile'] == "a"]
    df_a = a[col]
    y_a = df_a.pop(p)
    X_a = df_a
    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size = testing_size, shuffle = True, random_state = 11)
    b = df[df['quantile'] == "b"]
    df_b = b[col]
    y_b = df_b.pop(p)
    X_b = df_b
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size = testing_size, shuffle = True, random_state = 11)
    c = df[df['quantile'] == "c"]
    df_c = c[col]
    y_c = df_c.pop(p)
    X_c = df_c
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size = testing_size, shuffle = True, random_state = 11)


    d = df[df['quantile'] == "d"]
    df_d = d[col]
    y_d = df_d.pop(p)
    X_d = df_d
    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size = testing_size, shuffle = True, random_state = 11)


    e = df[df['quantile'] == "e"]
    df_e = e[col]
    y_e = df_e.pop(p)
    X_e = df_e
    X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(X_e, y_e, test_size = testing_size, shuffle = True, random_state = 11)


    X_train = pd.concat([X_a_train, X_b_train, X_c_train, X_d_train, X_e_train], ignore_index= True)


    X_test = pd.concat([X_a_test, X_b_test, X_c_test, X_d_test, X_e_test], ignore_index=True)


    y_train = pd.concat([y_a_train, y_b_train, y_c_train, y_d_train, y_e_train], ignore_index=True)


    y_test = pd.concat([y_a_test, y_b_test, y_c_test, y_d_test, y_e_test], ignore_index=True)

    if (DB == 'ycm_spindle'):
        X_test.to_excel(path + str(category_name) + "_X_test.xlsx", index = False)

        X_train.to_excel(path + str(category_name) + "_X_train.xlsx", index = False)

        y_train.to_excel(path + str(category_name) + "_y_train.xlsx", index = False)

        y_test.to_excel(path + str(category_name) + "_y_test.xlsx", index = False)
    else:
        X_test.to_excel(path + str(p) + "_X_test.xlsx", index = False)

        X_train.to_excel(path + str(p) + "_X_train.xlsx", index = False)

        y_train.to_excel(path + str(p) + "_y_train.xlsx", index = False)

        y_test.to_excel(path + str(p) + "_y_test.xlsx", index = False)
    return X_train, X_test, y_train, y_test



#%%

# %%
