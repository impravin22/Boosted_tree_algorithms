

#%%
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from preprocessing import preprocessing
#%%

X_train, X_test, y_train, y_test = preprocessing()

#%%
alpha = 0.95
#%%
train_data = pd.read_excel("95_train.xlsx")
train_data
#%%
test_data = pd.read_excel("95_test.xlsx")
test_data
#%%
y_autohigh = train_data['y_train_upper'].to_numpy()
y_autolow = train_data['y_train_lower'].to_numpy()
y_upper = test_data['y_upper'].to_numpy()
y_lower = test_data['y_lower'].to_numpy()


#%%
# Convert frequency of training dataframe to a numpy array
y_train_arr = y_train.to_numpy()
y_train_arr

#%%
#%%
#  Estimation of predicted training data below and upper estimates of actual training data
frac_below_upper = round(np.count_nonzero(y_autohigh > y_train_arr) / len(y_train_arr),3)
frac_above_upper = round(np.count_nonzero(y_autohigh < y_train_arr) / len(y_train_arr),3)
frac_above_lower = round(np.count_nonzero(y_autolow < y_train_arr) / len(y_train_arr),3)
frac_below_lower = round(np.count_nonzero(y_autolow > y_train_arr) / len(y_train_arr),3)
#%%%
# Convert frequency of testing datframe to a numpy array
y_test_arr = y_test.to_numpy()
y_test_arr

#%%
# Estimation of predicted training data below and upper estimates of actual training data
test_below_upper = round(np.count_nonzero(y_upper > y_test_arr) / len(y_test_arr),3)
test_above_upper = round(np.count_nonzero(y_upper < y_test_arr) / len(y_test_arr),3)
test_above_lower = round(np.count_nonzero(y_lower < y_test_arr) / len(y_test_arr),3)
test_below_lower = round(np.count_nonzero(y_lower > y_test_arr) / len(y_test_arr),3)
#%%
# Print calibration on training and testing
print('fraction below upper estimate: \t actual: ' + str(frac_below_upper) + '\t ideal: ' + str(alpha))
print('fraction above lower estimate: \t actual: ' + str(frac_above_lower) + '\t ideal: ' + str(alpha))

print('test below upper estimate: \t actual: ' + str(test_below_upper) + '\t ideal: ' + str(alpha))
print('test above lower estimate: \t actual: ' + str(test_above_lower) + '\t ideal: ' + str(alpha))

# %%
# ------------------- PLOTTING ----------------- #
# Plotting the preictions of training and testing data
f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 7), sharex = True)
#ax1.plot(y_train, y_train, 'ro', label=u'Mean Prediction')
ax1.plot(y_train, y_autohigh, 'bo')
ax1.plot(y_train, y_autolow, 'ko')
ax1.scatter(x=y_train_arr[y_autohigh < y_train_arr], y=y_train_arr[y_autohigh < y_train_arr], s=20, marker='x', c = 'red', 
        label = str(round(100*frac_above_upper,1))+'% of training data above upper (expect '+str(round(100*(1-alpha),1))+'%)')            
ax1.scatter(x=y_train_arr[y_autolow > y_train_arr], y=y_train_arr[y_autolow > y_train_arr], s=20, marker='x', c = 'orange', 
        label = str(round(100*frac_below_lower,1))+ '% of training data below lower (expect '+str(round(100*(1-alpha),1))+'%)')
ax1.legend(loc='best', bbox_to_anchor=(1, 1.2))

ax1.set_title('train')
#ax2.plot(y_test, y_test, 'ro', label=u'Mean Prediction')
ax2.plot(y_test, y_upper, 'bo')
ax2.plot(y_test, y_lower, 'ko')
#plt.fill(np.concatenate([y_test, y_test[::-1]]),
#            np.concatenate([y_upper, y_lower[::-1]]),
#            alpha=.5, fc='b', ec='None', label=(str(round(100*(alpha-0.5)*2))+'% prediction interval'))
ax2.scatter(x=y_test_arr[y_upper < y_test_arr], y=y_test_arr[y_upper < y_test_arr], s=20, marker='x', c = 'red', 
        label = str(round(100*test_above_upper,1))+'% of testing data above upper (expect '+str(round(100*(1-alpha),1))+'%)')            
ax2.scatter(x=y_test_arr[y_lower > y_test_arr], y=y_test_arr[y_lower > y_test_arr], s=20, marker='x', c = 'orange', 
        label = str(round(100*test_below_lower,1))+ '% of testing data below lower (expect '+str(round(100*(1-alpha),1))+'%)')
ax2.legend(loc='best', bbox_to_anchor=(1, 1.2) )
ax2.set_title('test')

#%%

upper = pd.read_excel('_best_params_95_upper.xlsx')
lower = pd.read_excel('_best_params_95_lower.xlsx')
#%%
upper
#%%
lower
#%%
lgb_train = lgb.Dataset(train_data)

#%%
evals_result = {}

print('start training.....')
params_l = {'n_estimators' : int(lower.iloc[0]['n_estimators']),
                                'max_depth': int(lower.iloc[0]['max_depth']),
                                'learning_rate': float(lower.iloc[0]['learning_rate']),
                                'bagging_fraction': float(lower.iloc[0]['bagging_fraction']),
                                'min_data_in_leaf': int(lower.iloc[0]['min_data_in_leaf']),
                                'verbose' : -1,
                                'metrc': 'l1',
                                'num_threads':50,
                                'objective' : 'quantile',
                                'num_leaves': pow(2, int(lower.iloc[0]['max_depth'])),
                                'alpha':0.025
                                }
model_l = lgb.train(params = params_l, evals_result = evals_result, verbose_eval=10, valid_sets=[lgb_train],
                        num_boost_round = int(lower.iloc[0]['n_estimators']), train_set = lgb_train)

ax = lgb.plot_metric(evals_result, metric = 'l1')
plt.show()






# %%
import pickle
with open('model_lower.pkl', 'rb') as file:
    lower_model = pickle.load(file)

with open('model_upper.pkl', 'rb') as file:
    upper_model = pickle.load(file)
#%%

res = lower_model.predict(X_test)


#%%
lgb.plot_importance(lower_model)
#%%
lgb.create_tree_digraph(upper_model, tree_index=1000)

# %%
lgb.plot_metric()
# %%
