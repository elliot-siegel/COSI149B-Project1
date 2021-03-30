from imgdata import ProcessImage
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

imageProcessing = ProcessImage("Project1/train", "Project1/coordinates_train.csv")
imageNames = imageProcessing.getImgList()

X, y = imageProcessing.extractImages(1, 50, len(imageNames))
# W, z = imageProcessing.extractImages(len(imageNames) - 1, len(imageNames) - 10)
#
# X = np.concatenate((X, W), axis=0)
# y = np.concatenate((y, z), axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
print("here")
clf = SVC(max_iter = 1000, tol = 0.001)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(y_test)
print(y_pred)
print(y)
print(score)
