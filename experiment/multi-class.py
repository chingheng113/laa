import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn import over_sampling
from sklearn.model_selection import KFold
import xgboost as xgb

data = pd.read_csv(os.path.join('..', 'data', 'LAA_computation_lcms1_cli_class_s.csv'))
data = data.replace({'.':np.nan, '#N/A':np.nan})
# only gene data ===============
data = data.iloc[:, 3:198]
# gene data and clinical data ==
# data = data.iloc[:, 3:]
# data = data.drop(['AcSugar', 'HsCRP1', 'Hemoglobin'], axis=1)  # too much missing

how_many_null = data.isnull().sum().sort_values(ascending = False)
Y_data = data[['Label', 'S1', 'S2', 'S3', 'S4']]
label = 'S1'
if label == 'S1':
    c = 6
elif label == 'S2':
    c = 9
elif label == 'S3':
    c = 4
else:
    c = 7
y_data = pd.cut(Y_data[label], [-1, 0, c, 999], labels=[0, 1, 2]) #9
y_data = pd.DataFrame(y_data, columns=[label])
# a = pd.concat((y_data, Y_data[['C2']]), axis=1)
# a = pd.concat((a, Y_data[['S2']]), axis=1)

X_data = data.drop(['Label', 'S1', 'S2', 'S3', 'S4'], axis=1)

all_acc = []
# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=i, stratify=y_data)
for train_index, test_index in KFold(n_splits=10, random_state=42, shuffle=True).split(X_data):
    # print(len(train_index), len(test_index))
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    # imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # scaling
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    # over-sampling
    # print('before', y_train.groupby([label]).size())
    sm = over_sampling.SVMSMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # print('after', y_train.groupby([label]).size())

    # define the model
    # model = ExtraTreesClassifier(n_estimators=250,  random_state=0)
    # model = SVC(kernel='linear', gamma='auto', probability=True, class_weight='balanced')
    model = xgb.XGBClassifier(objective='multi:softmax', learning_rate=0.5)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # true_labels = y_test.apply(np.argmax, axis=1)
    # pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    all_acc.append(acc)
print(np.mean(all_acc), np.std(all_acc))
print('done')






# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]
# features = X_train.columns.to_list()
# for i in indices:
#     print(features[i], importances[i])