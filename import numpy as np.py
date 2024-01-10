import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Maternal Health Risk Data Set.csv')
dataBersih = dataset.replace('?', np.NaN)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

dataBersihNew = dataBersih.dropna()
dataBersihNew



X = dataBersihNew.iloc[:, [1,2,3,4,5,6,7,8]].values
y = dataBersihNew.iloc[:, 0].values

#X = X.astype(float)
#y = y.astype(float)

#yBaru = pd.DataFrame(y)
#yBaru

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean', p = 2)
classifier.fit(X_train, y_train)

import pandas as pd

y_pred = classifier.predict(X_test)
df_pred = pd.DataFrame(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm)
df_cm