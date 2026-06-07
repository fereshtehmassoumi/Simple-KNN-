import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
digi_dataset = load_digits()
print(digi_dataset.keys())

#Dataset beobachten
#print(digi_dataset['data'])
#print(digi_dataset.shape)
#print(digi_dataset['feature_names'])
#print(digi_dataset['images'][0])
#plt.imshow(digi_dataset['images'][0], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()
#print(digi_dataset['target_names'])

#KNN Modell bauen
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x=digi_dataset.data
y=digi_dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
print(knn_model.score(x_test, y_test))

#Prediction
prediction = knn_model.predict(x_test)
print("prediction:", prediction[0])
print("y_test:", y_test[0])