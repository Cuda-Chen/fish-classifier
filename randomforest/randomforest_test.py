import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

lbp_train = pd.read_csv('../LBP_feature_train.csv')
#lbp_val = pd.read_csv('../LBP_feature_val.csv')
fd_train = pd.read_csv('../fd_feature_train.csv')
#fd_val = pd.read_csv('../fd_feature_val.csv')
color_som_train = pd.read_csv('../color_som_train.csv')
#color_som_val = pd.read_csv('../color_som_val.csv')
train_label = pd.read_csv('../train_label.csv')
#val_label = pd.read_csv('../val_label.csv')

x_train = pd.concat([lbp_train, fd_train, color_som_train], axis=1)
y_train = train_label['class_no']

pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier())])
scores = cross_val_score(pl_random_forest, x_train, y_train, cv=10, scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
