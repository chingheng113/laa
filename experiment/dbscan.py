import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import os
# #############################################################################
data = pd.read_csv(os.path.join('..', 'data', 'LAA_computation_lcms1_cli_class_s.csv'))
data = data.replace({'.':np.nan, '#N/A':np.nan})
data = data.iloc[:, 3:198]
# data = data.dropna(subset=['Label','miR21ΔΔct ', 'H1'], axis=0)
how_many_null = data.isnull().sum().sort_values(ascending = False)
Y_data = data[['Label', 'S1', 'S2', 'S3', 'S4']]
label = 'S2'
y_data = pd.cut(Y_data['S2'], [-1, 0, 9, 999], labels=[0, 1, 2]) #9
y_data = pd.DataFrame(y_data, columns=[label])

X_data = data.drop(['Label', 'S1', 'S2', 'S3', 'S4'], axis=1)


# imputation or drop na
# X_data = X_data.dropna(axis=0)
imp = SimpleImputer(missing_values=np.nan, strategy='median')
X_data = imp.fit_transform(X_data)

# scaling
# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
X_data = scaler.fit_transform(X_data)

# Dimension reduction
X_data = PCA(n_components=2).fit_transform(X_data)


# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1, min_samples=3).fit(X_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_data, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(y_data, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(y_data, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(y_data, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(y_data, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_data, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()