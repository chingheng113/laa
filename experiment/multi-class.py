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

data = pd.read_csv(os.path.join('..', 'data', 'LAA_computation_lcms1_cli_class_s.csv'))
data = data.replace({'.':np.nan, '#N/A':np.nan})
data = data.iloc[:, 3:202]
data = data.dropna(subset=['Label'], axis=0)

Y_data = data[['Label', 'C1', 'C2', 'C3', 'C4', 'S1', 'S2', 'S3', 'S4']]

label = 'C1'
y_data = Y_data[[label]]

# label = 'S2'
# y_data = pd.cut(Y_data['S2'], [-1, 0, 9, 999], labels=[0, 1, 2]) #9
# y_data = pd.DataFrame(y_data, columns=[label])
# a = pd.concat((y_data, Y_data[['C2']]), axis=1)
# a = pd.concat((a, Y_data[['S2']]), axis=1)

X_data = data.drop(['Label', 'C1', 'C2', 'C3', 'C4', 'S1', 'S2', 'S3', 'S4'], axis=1)
X_data = pd.get_dummies(X_data, columns=['sex'])

all_acc = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=i, stratify=y_data)
    # imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test= min_max_scaler.transform(X_test)

    # over-sampling
    # print('before', y_train.groupby([label]).size())
    sm = over_sampling.SVMSMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # print('after', y_train.groupby([label]).size())

    # define the model
    model = ExtraTreesClassifier(n_estimators=250,  random_state=0)
    # model = SVC(kernel='linear', gamma='auto', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
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