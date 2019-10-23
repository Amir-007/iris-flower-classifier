
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('IRIS.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

dataset.describe()
dataset.groupby('species').size()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
dataset.hist()
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

rep = classification_report(y_test,y_pred)