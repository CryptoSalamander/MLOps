from iris_classifier import IrisClassifier
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma='scale')
clf.fit(X,y)

iris_classifier_service = IrisClassifier()
iris_classifier_service.pack('model', clf)

saved_path = iris_classifier_service.save()