
import pandas as pd
from xgboost import XGBClassifier

lbp_train = pd.read_csv('../LBP_feature_train.csv')
lbp_val = pd.read_csv('../LBP_feature_val.csv')
color_som_train = pd.read_csv('../color_som_train.csv')
color_som_val = pd.read_csv('../color_som_val.csv')
train_label = pd.read_csv('../train_label.csv')
val_label = pd.read_csv('../val_label.csv')

# get your training and testing data here
# and to make sure to reshape and normalize!
x_train = pd.concat([lbp_train, color_som_train], axis=1).values
x_test = pd.concat([lbp_val, color_som_val], axis=1).values

# convert class vectors to binary class matrices (one-hot encoding)
# there are some missing class!
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

xgbc = XGBClassifier()

xgbc.fit(X_train, y_train)

print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))
