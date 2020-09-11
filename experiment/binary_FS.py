import pandas as pd
import os, collections
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
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(os.path.join('..', 'data', 'LAA_computation_lcms1_cli_class_s.csv'))
data = data.replace({'.':np.nan, '#N/A':np.nan})
data = data.dropna(subset=['Label'], axis=0)
Y_data = data[['Label', 'S1', 'S2', 'S3', 'S4']]
y_data = Y_data[['Label']]
X_data = data.drop(['Label', 'S1', 'S2', 'S3', 'S4'], axis=1)

# all feature ==
X_data = X_data.iloc[:, 3:]
X_data = X_data.drop(['AcSugar', 'HsCRP1', 'Hemoglobin'], axis=1)  # too much missing


all_auroc = []
fs_elements = []
# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=i, stratify=y_data)
for train_index, test_index in KFold(n_splits=10, random_state=42, shuffle=True).split(X_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    # print(X_train.shape)
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    features = X_train.columns.to_list()
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

    # feature selection
    model = ExtraTreesClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    importances = model.feature_importances_
    no_zero_importance = importances[np.where(importances > 0)]
    cutoff = np.std(no_zero_importance) + np.min(no_zero_importance)
    indices = np.where(importances > cutoff)[0]
    fs_elements = np.concatenate((fs_elements, np.array(features)[indices]), axis=0)
    # for i in indices:
    #     print(features[i], importances[i])
    X_train_fs = X_train[:, indices]
    X_test_fs = X_test[:, indices]

    # define the model
    # model = ExtraTreesClassifier(n_estimators=250,  random_state=42)
    # model = SVC(kernel='linear', probability=True)
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5, subsample=0.5)
    model.fit(X_train_fs, y_train.values.ravel())
    y_pred = model.predict_proba(X_test_fs)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    auroc = auc(fpr, tpr)
    print('auc', auroc)
    all_auroc.append(auroc)
print(np.mean(all_auroc), np.std(all_auroc))
df = pd.DataFrame.from_dict(collections.Counter(fs_elements), orient='index', columns=['feq']).reset_index()
result = df.sort_values(['feq'], ascending=0)
result.set_index('index', inplace=True)
a = result[result.feq > 9]
print(result.shape)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))
sns.heatmap(result, annot=True, fmt="g", cmap='viridis', linewidths=0.3, yticklabels=True)
plt.show()

print('done')