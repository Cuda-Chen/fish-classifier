
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
           41, 42]

lbp_train = pd.read_csv('../LBP_feature_train.csv')
lbp_val = pd.read_csv('../LBP_feature_val.csv')
fd_train = pd.read_csv('../fd_feature_train.csv')
fd_val = pd.read_csv('../fd_feature_val.csv')
color_som_train = pd.read_csv('../color_som_train.csv')
color_som_val = pd.read_csv('../color_som_val.csv')
train_label = pd.read_csv('../train_label.csv')
val_label = pd.read_csv('../val_label.csv')

# get your training and testing data here
# and to make sure to reshape and normalize!
x_train = pd.concat([lbp_train, fd_train, color_som_train], axis=1).values
x_test = pd.concat([lbp_val, fd_val, color_som_val], axis=1).values

# convert class vectors to binary class matrices (one-hot encoding)
# there are some missing class!
'''
train_label_list = train_label['class_no'].tolist()
onehot_train = []
for value in train_label_list:
    class_id = [0 for _ in range(len(classes))]
    class_id[value - 1] = 1 
    onehot_train.append(class_id)
y_train = pd.DataFrame(onehot_train)

val_label_list = val_label['class_no'].tolist()
onehot_val = []
for value in val_label_list:
    class_id = [0 for _ in range(len(classes))]
    class_id[value - 1] = 1 
    onehot_val.append(class_id)
y_test = pd.DataFrame(onehot_val)
'''
y_train = train_label['class_no'].values
y_test = val_label['class_no'].values

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 275, 'max_depth': 6, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0.008, 'reg_lambda': 0.1}

'''
xgbc = XGBClassifier()

xgbc.fit(x_train, y_train)

print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(x_test, y_test))
'''
model = XGBClassifier()
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=8)
optimized_GBM.fit(x_train, y_train)
evaluate_result = optimized_GBM.cv_results_['mean_test_score']
print('result of each iteration:{0}'.format(evaluate_result))
print('best params:{0}'.format(optimized_GBM.best_params_))
print('test score:{0}'.format(optimized_GBM.best_score_))
