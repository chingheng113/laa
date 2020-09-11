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
from imblearn import over_sampling
from sklearn.model_selection import KFold
import xgboost as xgb

data = pd.read_csv(os.path.join('..', 'data', 'LAA_computation_lcms1_cli_class_s.csv'))
data = data.replace({'.':np.nan, '#N/A':np.nan})
data = data.dropna(subset=['Label'], axis=0)
Y_data = data[['Label', 'S1', 'S2', 'S3', 'S4']]
y_data = Y_data[['Label']]
X_data = data.drop(['Label', 'S1', 'S2', 'S3', 'S4'], axis=1)

# only LC mars ===============
# X_data = X_data.iloc[:, 3:7]

# only Metabolomics  ====================
X_data = X_data.iloc[:, 9:192]

# only clinical ==============
# X_data = X_data.iloc[:, 193:]
# X_data = pd.concat([X_data, data[['age', 'sex']]], axis=1)
# X_data = X_data.drop(['AcSugar', 'HsCRP1', 'Hemoglobin'], axis=1)  # too much missing

# all feature ==
# X_data = X_data.iloc[:, 3:]
# X_data = X_data.drop(['AcSugar', 'HsCRP1', 'Hemoglobin'], axis=1)  # too much missing


all_auroc = []
# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=i, stratify=y_data)
for train_index, test_index in KFold(n_splits=10, random_state=42, shuffle=True).split(X_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    # imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # scaling
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # over-sampling
    # print('before', y_train.groupby(['Label']).size())
    sm = over_sampling.SVMSMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # print('after', y_train.groupby(['Label']).size())

    # define the model
    # model = ExtraTreesClassifier(n_estimators=250,  random_state=42)
    # model = SVC(kernel='linear', probability=True)
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5, subsample=0.5)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    auroc = auc(fpr, tpr)
    print('auc', auroc)
    all_auroc.append(auroc)
print(np.mean(all_auroc), np.std(all_auroc))
print('done')






# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]
# features = X_train.columns.to_list()
# for i in indices:
#     print(features[i], importances[i])