import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
sns.set_palette('husl')
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

#load the dataset
dataset = pd.read_csv('iris_dataset.csv')

label_mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

sns.FacetGrid(dataset, height=5,hue="Species").map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend().savefig('snsplot.png')

dataset_X = dataset.drop(['Species'], axis=1).values
dataset_Y = dataset.Species.replace(label_mapping).values.reshape(dataset.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.3, shuffle = True, random_state = 123)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train.ravel())

prediction = neigh.predict(X_test)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test)*100, 'percent')