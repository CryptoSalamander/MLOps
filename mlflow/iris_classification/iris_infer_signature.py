from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd


iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

output_schema = Schema([ColSpec("long")])


signature = infer_signature(iris_train, clf.predict(iris_train))
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)