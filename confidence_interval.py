


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from alive_progress import alive_bar
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from preprocessing import preprocessing


import warnings

warnings.filterwarnings("ignore")

path = './'


name = str('test')
thresh = int(1000)

#%%

def preprocessing_one_two():
    X_train_one, X_test_one, y_train_one, y_test_one = preprocessing('config_1.ini')
    X_train_two, X_test_two, y_train_two, y_test_two = preprocessing('config_2.ini')

    #X_test_one['class'] = 0
    #X_train_one['class'] = 0
    #X_test_two['class'] = 1
    #X_train_two['class'] = 1

    one_two_X_train = pd.concat([X_train_one, X_train_two], ignore_index=True)
    one_two_X_test = pd.concat([X_test_one, X_test_two], ignore_index=True)
    one_two_y_train = pd.concat([y_train_one, y_train_two], ignore_index=True)
    one_two_y_test = pd.concat([y_test_one, y_test_two], ignore_index=True)
    one_two_X_training = one_two_X_train
    one_two_X_testing = one_two_X_test
    #two_X_test = X_test_two.iloc[:, :-1]
    #two_y_test = y_test_two.iloc[:, :-1]
    return one_two_X_training, one_two_X_testing, one_two_y_train, one_two_y_test, X_train_two, y_train_two


X_train, X_test_two, y_train, y_test_two, X_test, y_test = preprocessing_one_two()


#X_test = X_testing.iloc[:, :-1]
#y_test = y_testing.iloc[:, :-1]

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label = y_test)
#%%
param_grid1 = {'learning_rate': np.arange(0.0035, 0.0037, 0.0001), 
                'max_depth': np.arange(5, 6, 1),
                'min_data_in_leaf' : np.arange(5, 6, 1),
                'bagging_fraction': np.arange(0.55, 0.59, 0.01)
                }                 
            
grid1 = ParameterGrid(param_grid1)
#%%
param_grid1 = {'learning_rate': np.arange(0.0035, 0.0037, 0.0001), 
                'max_depth': np.arange(5, 6, 1),
                'min_data_in_leaf' : np.arange(5, 6, 1),
                'bagging_fraction': np.arange(0.55, 0.59, 0.01)
                }                 
            
grid = ParameterGrid(param_grid1)
#%%
len(grid)
#%%


def lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf):
                # Extend the model creation section
    params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'bagging_fraction': bagging_fraction,
                    'min_data_in_leaf': min_data_in_leaf,
                    'num_leaves': pow(2, max_depth),
                    'bagging_freq':5,
                    'objective' : 'quantile',
                    'alpha':0.95,
                    'num_threads':50,
                    'metric': 'mean_absolute_error',
                    'verbose': -1,
                    'metric' : 'mae',
                    'first_metric_only': True}
    
                # Extend the return part
    cv = lgb.cv(params = params, train_set=train_data, num_boost_round = 10000,
                                                    nfold=5, stratified=False, early_stopping_rounds=10)
    return cv    


#%%            
results_list = []
learning_rate_list = param_grid1['learning_rate']
max_depth_list = param_grid1['max_depth']
bagging_fraction_list = param_grid1['bagging_fraction']
min_data_in_leaf_list = param_grid1['min_data_in_leaf']

#%%
for learning_rate in learning_rate_list:
    for max_depth in max_depth_list:
        for bagging_fraction in bagging_fraction_list:
            for min_data_in_leaf in min_data_in_leaf_list:
                cv_df = pd.DataFrame(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf))
                cv_df_list =[cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1]]
                results_list.append([learning_rate, max_depth, bagging_fraction, min_data_in_leaf, cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1] ])
#%%
cv_df                
#%%
results_df = pd.DataFrame(results_list)
results_df
#%%
def lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf):
                # Extend the model creation section
    params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'bagging_fraction': bagging_fraction,
                    'feature_fraction': feature_fraction,
                    'min_data_in_leaf': min_data_in_leaf,
                    'num_leaves': pow(2, max_depth),
                    'objective' : 'quantile',
                    'alpha':0.95,
                    'num_threads':50,
                    'bagging_seed': 11,
                    'verbose': -1,
                    'metric' : 'l1',
                    'first_metric_only': True}
    
                # Extend the return part
    cv = lgb.cv(params = params, train_set=train_data, num_boost_round = 10000,
                                                    nfold=5, stratified=False, early_stopping_rounds=10)
    return cv    

#%%


learning_rate_list = param_grid1['learning_rate']
max_depth_list = param_grid1['max_depth']
bagging_fraction_list = param_grid1['bagging_fraction']
min_data_in_leaf_list = param_grid1['min_data_in_leaf']
bagging_freq_list = param_grid1['bagging_freq']
feature_fraction_list = param_grid1['feature_fraction']



#%%
results_list = []
for learning_rate in learning_rate_list:
    for max_depth in max_depth_list:
        for bagging_fraction in bagging_fraction_list:
            for min_data_in_leaf in min_data_in_leaf_list:
                for bagging_freq in bagging_freq_list:
                    for feature_fraction in feature_fraction_list:
                        print(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf))
                        cv_df = pd.DataFrame(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf))
                        cv_df_list =[cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1]]
                        results_list.append([learning_rate, max_depth, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf, cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1] ])

#%%
cv_df
#%%
results_list
#%%
results_df = pd.DataFrame(results_list)
results_df            
#%%
results_df.columns = ['learning_rate', 'max_depth', 'bagging_fraction','bagging_freq', 'feature_fraction', 'min_data_in_leaf', 'n_estimators', 'mae']
results_df

#%%
mae_lower = results_df[results_df['mae'] == results_df['mae'].min()]


#%%

def confidence_interval_test(PI):

    for k in PI:
        lower = (100 - k)/200
        upper = (100 + k)/200

        if (k == 0):
            def lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf):
                # Extend the model creation section
                params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'bagging_fraction': bagging_fraction,
                    'min_data_in_leaf': min_data_in_leaf,
                    'num_leaves': pow(2, max_depth),
                    'objective' : 'quantile',
                    'alpha':lower,
                    'num_threads':50,
                    'metric': 'mean_absolute_error',
                    'verbose': -1,
                    'metric' : 'mae',
                    'first_metric_only': True}
    
                # Extend the return part
                cv = lgb.cv(params = params, train_set=train_data, num_boost_round = 10000,
                                                    nfold=5, stratified=False, early_stopping_rounds=10)
                return cv    

            
            results_list = []
            learning_rate_list = param_grid1['learning_rate']
            max_depth_list = param_grid1['max_depth']
            bagging_fraction_list = param_grid1['bagging_fraction']
            min_data_in_leaf_list = param_grid1['min_data_in_leaf']


            #initialize timer
            with alive_bar(len(grid)) as bar:
                for learning_rate in learning_rate_list:
                    for max_depth in max_depth_list:
                        for bagging_fraction in bagging_fraction_list:
                            for min_data_in_leaf in min_data_in_leaf_list:
                                cv_df = pd.DataFrame(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf))
                                cv_df_list =[cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1]]
                                results_list.append([learning_rate, max_depth, bagging_fraction, min_data_in_leaf, cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1] ])
                                bar()

            
            results_df = pd.DataFrame(results_list)
            

            results_df.columns = ['learning_rate', 'max_depth', 'bagging_fraction', 'min_data_in_leaf', 'n_estimators', 'mae']
            mae_lower = results_df[results_df['mae'] == results_df['mae'].min()]
            mae_lower.to_excel(path + name + '_best_params_' + str(k)  + '.xlsx', index = False)

            
            params_l = {'n_estimators' : int(mae_lower.iloc[0]['n_estimators']),
                                'max_depth': int(mae_lower.iloc[0]['max_depth']),
                                'learning_rate': float(mae_lower.iloc[0]['learning_rate']),
                                'bagging_fraction': float(mae_lower.iloc[0]['bagging_fraction']),
                                'min_data_in_leaf': int(mae_lower.iloc[0]['min_data_in_leaf']),
                                'verbose' : -1,
                                'num_threads':50,
                                'objective' : 'quantile',
                                'num_leaves': pow(2, int(mae_lower.iloc[0]['max_depth'])),
                                'alpha':lower
                                }
            model_l = lgb.train(params = params_l, verbose_eval=False, 
                        num_boost_round = int(mae_lower.iloc[0]['n_estimators']), train_set = train_data)

             # Lower quantile Prediction on testing data
            y_pred = model_l.predict(X_test)


            # Quantile test dataframe
            # make dataframe using predictions of lower, upper and median quatiles on testing dataset
            conf = pd.DataFrame(y_pred)
            conf.columns = ['y_pred']
            conf['y_test'] = pd.DataFrame(y_test)

        
            # Testing
           

            conf_t = conf[['y_test','y_pred']]
        
            conf_t.to_excel( path  + name + '_CI' + '_test_'  + str(k) + '.xlsx', index = False)
            # print(str(p))

            # Training
            # Prediction on Training Data
            # Check calibration by predicting the training data
            # y_autopred = model.predict(X_train)
            y_autolow = model_l.predict(X_train)

            # Quantile train dataframe 
            # make dataframe using predictions of lower, upper and median quantiles on training dataset
            conf_train = pd.DataFrame(y_autolow)
            conf_train.columns = ['y_train_pred']
            conf_train['y_train'] = pd.DataFrame(y_train)

            # Training 
            # Subtract the predictions on training data of lower and upper quantiles from 660(SKF Threshold) 
            conf_tr = conf_train[['y_train', 'y_train_pred']]
            # # Save the train confidence interval dataframe in a csv file 
            conf_tr.to_excel( path +  name + '_CI' + '_train_' + str(k) + '.xlsx', index = False)  
        else:
            def lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf):
                # Extend the model creation section
                params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'bagging_fraction': bagging_fraction,
                    'min_data_in_leaf': min_data_in_leaf,
                    'num_leaves': pow(2, max_depth),
                    'objective' : 'quantile',
                    'alpha':lower,
                    'num_threads':50,
                    'metric': 'mean_absolute_error',
                    'verbose': -1,
                    'metric' : 'mae',
                    'first_metric_only': True}
    
                # Extend the return part
                cv = lgb.cv(params = params, train_set=train_data, num_boost_round = 10000,
                                                    nfold=5, stratified=False, early_stopping_rounds=10)
                return cv

            
            results_list = []
            learning_rate_list = param_grid1['learning_rate']
            max_depth_list = param_grid1['max_depth']
            bagging_fraction_list = param_grid1['bagging_fraction']
            min_data_in_leaf_list = param_grid1['min_data_in_leaf']

            #initialize timer
            with alive_bar(len(grid)) as bar:
                for learning_rate in learning_rate_list:
                    for max_depth in max_depth_list:
                        for bagging_fraction in bagging_fraction_list:
                            for min_data_in_leaf in min_data_in_leaf_list:
                                cv_df = pd.DataFrame(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf))
                                cv_df_list =[cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1]]
                                results_list.append([learning_rate, max_depth, bagging_fraction, min_data_in_leaf, cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1] ])
                                bar()

            
            results_df = pd.DataFrame(results_list)
            results_df
            

            results_df.columns = ['learning_rate', 'max_depth', 'bagging_fraction','min_data_in_leaf', 'n_estimators', 'mae']
            mae_lower = results_df[results_df['mae'] == results_df['mae'].min()]
            mae_lower.to_excel(path + name + '_best_params_'+ str(k) + '_lower' + '.xlsx', index = False)

            
            params_l = {'n_estimators' : int(mae_lower.iloc[0]['n_estimators']),
                                'max_depth': int(mae_lower.iloc[0]['max_depth']),
                                'learning_rate': float(mae_lower.iloc[0]['learning_rate']),
                                'bagging_fraction': float(mae_lower.iloc[0]['bagging_fraction']),
                                'min_data_in_leaf': int(mae_lower.iloc[0]['min_data_in_leaf']),
                                'verbose' : -1,
                                'num_threads':50,
                                'objective' : 'quantile',
                                'num_leaves': pow(2, int(mae_lower.iloc[0]['max_depth'])),
                                'alpha':lower
                                }
            model_l = lgb.train(params = params_l, verbose_eval=False, 
                        num_boost_round = int(mae_lower.iloc[0]['n_estimators']), train_set = train_data)    
            # Lower quantile Prediction on testing data
            y_lower = model_l.predict(X_test)

            def lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf):
                # Extend the model creation section
                params = {
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'bagging_fraction': bagging_fraction,
                    'num_leaves': pow(2, max_depth),
                    'min_data_in_leaf':min_data_in_leaf,
                    'objective' : 'quantile',
                    'alpha':upper,
                    'num_threads':50,
                    'metric': 'mean_absolute_error',
                    'verbose': -1,
                    'first_metric_only': True}
    
                # Extend the return part
                cv = lgb.cv(params = params, train_set=train_data, num_boost_round = 10000,
                                                    nfold=5, stratified=False, early_stopping_rounds=10)
                return cv

           
            
            results_list = []
            learning_rate_list = param_grid1['learning_rate']
            max_depth_list = param_grid1['max_depth']
            bagging_fraction_list = param_grid1['bagging_fraction']
            min_data_in_leaf_list = param_grid1['min_data_in_leaf']


            #initialize timer
            with alive_bar(len(grid)) as bar:
                for learning_rate in learning_rate_list:
                    for max_depth in max_depth_list:
                        for bagging_fraction in bagging_fraction_list:
                            for min_data_in_leaf in min_data_in_leaf_list:
                                cv_df = pd.DataFrame(lgbreg_grid_search_cv(learning_rate, max_depth, bagging_fraction, min_data_in_leaf))
                                cv_df_list =[cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1]]
                                results_list.append([learning_rate, max_depth, bagging_fraction, min_data_in_leaf, cv_df.last_valid_index(), cv_df['l1-mean'].iloc[-1] ])
                                bar()

            
            results_df = pd.DataFrame(results_list)
            results_df
            

            results_df.columns = ['learning_rate', 'max_depth', 'bagging_fraction', 'min_data_in_leaf', 'n_estimators', 'mae']
            mae_lower = results_df[results_df['mae'] == results_df['mae'].min()]
            mae_lower.to_excel(path + name + '_best_params_'+ str(k) + '_upper'  + '.xlsx', index = False)

            
            params_u = {'n_estimators' : int(mae_lower.iloc[0]['n_estimators']),
                                'max_depth': int(mae_lower.iloc[0]['max_depth']),
                                'learning_rate': float(mae_lower.iloc[0]['learning_rate']),
                                'bagging_fraction': float(mae_lower.iloc[0]['bagging_fraction']),
                                'min_data_in_leaf': int(mae_lower.iloc[0]['min_data_in_leaf']),
                                'verbose' : -1,
                                'num_threads':50,
                                'objective' : 'quantile',
                                'num_leaves': pow(2, int(mae_lower.iloc[0]['max_depth'])),
                                'alpha':upper
                                }
            model_u = lgb.train(params = params_u, verbose_eval=False, 
                        num_boost_round = int(mae_lower.iloc[0]['n_estimators']), train_set = train_data)    
            # Upper quantile Prediction on testing data
            y_upper = model_u.predict(X_test)            

            # Quantile test dataframe
            # make dataframe using predictions of lower, upper and median quatiles on testing dataset
            conf = pd.DataFrame(y_lower)
            conf.columns = ['y_lower']
            # conf['y_pred'] = pd.DataFrame(y_pred)
            conf['y_upper'] = pd.DataFrame(y_upper)
            conf['y_test'] = pd.DataFrame(y_test)

        
            # Testing
            # Subtract the predictions on test data of lower and upper quantiles from 660(SKF Threshold) 

            conf_t = conf[['y_test','y_lower', 'y_upper']]
        
            conf_t.to_excel( path + name + '_'+ str(k) + '_test'   + '.xlsx', index = False)

            conf_t['lower-thresh'] = round(abs(conf_t['y_lower'] - thresh), 0)
            conf_t['upper-thresh'] = round(abs(conf_t['y_upper'] - thresh), 0)
            conf_t['bool'] = conf_t['lower-thresh'].ge(conf_t['upper-thresh'])


            # Assign the value of y_lower to y_interval if bool is true
            conf_t_lower = conf_t[conf_t['bool'] == True]
            conf_t_lower['ci'] = conf_t_lower['y_lower']
            # Assign the value of y_upper to y_interval if bool is False
            conf_t_upper = conf_t[conf_t['bool'] ==  False]
            conf_t_upper['ci'] = conf_t_upper['y_upper']
            # combine and sort the two dataframes (lower and upper)
            conf_t_final = pd.concat([conf_t_lower, conf_t_upper], ignore_index=False)
            conft = conf_t_final.sort_index(axis = 0)
            confte = pd.DataFrame()
            confte['orig'] = conft['y_test']
            confte['ci'] = round(conft['ci'], 0)
            # Save the test confidence interval dataframe in a csv file 
            confte.to_excel( path + name + '_CI_'  + str(k) + '_test'   + '.xlsx', index = False)



            # print(str(p))

            # Training
            # Prediction on Training Data
            # Check calibration by predicting the training data
            # y_autopred = model.predict(X_train)
        
            
            y_autolow = model_l.predict(X_train)
            y_autohigh = model_u.predict(X_train)

            # Quantile train dataframe 
            # make dataframe using predictions of lower, upper and median quantiles on training dataset
            conf_train = pd.DataFrame(y_autolow)
            conf_train.columns = ['y_train_lower']
            # conf_train['y_train_pred'] = pd.DataFrame(y_autopred)
            conf_train['y_train_upper'] = pd.DataFrame(y_autohigh)
            conf_train['y_train'] = pd.DataFrame(y_train)


            # Training 
            conf_tr = conf_train[['y_train', 'y_train_lower',  'y_train_upper']]
            conf_tr.to_excel( path + name + '_' + str(k) + '_train'   + '.xlsx', index = False)


            conf_tr['lower - thresh'] = round(abs(conf_tr['y_train_lower'] - thresh), 0)
            conf_tr['upper - thresh'] = round(abs(conf_tr['y_train_upper'] - thresh), 0)
            conf_tr['bool'] = conf_tr['lower - thresh'].ge(conf_tr['upper - thresh'])
            # Assign the value of y_train_lower to train_interval if bool is true
            conf_tr_lower = conf_tr[conf_tr['bool'] == True ]
            conf_tr_lower['ci'] = conf_tr_lower['y_train_lower']

            # Assign the value of y_train_upper to train_interval if bool is false
            conf_tr_upper = conf_tr[conf_tr['bool'] == False]
            conf_tr_upper['ci'] = conf_tr['y_train_upper']

            # combine and sort the two dataframes (tr_lower and tr_upper)
            conf_tr_final = pd.concat([conf_tr_lower, conf_tr_upper], ignore_index = False)
            conftr = conf_tr_final.sort_index(axis = 0)

            conftra = pd.DataFrame()
            conftra['orig'] = conftr['y_train']
            conftra['ci'] = round(conftr['ci'], 0)
            # Save the train confidence interval dataframe in a csv file 
            conftra.to_excel( path + name + '_CI_' + str(k) + '_train'  + '.xlsx', index = False)
            
    return conf_t, conf_tr


PI = [95, 90]
confidence_interval_test(PI)
