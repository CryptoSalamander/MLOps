from sys import version_info
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target
x_train, x_test, y_train, _ = train_test_split(x,y,test_size=0.2,random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

xgb_model = xgb.train(params={'max_depth': 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "xgb_model.pth"
xgb_model.save_model(xgb_model_path)

artifacts = {
    "xgb_model": xgb_model_path
}

import mlflow.pyfunc
class XGBWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        import xgboost as xgb
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])
        
    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)
    
    
import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'xgboost=={}'.format(xgb.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'xgb_env'
}

mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path, 
    python_model=XGBWrapper(), 
    artifacts=artifacts,
    conda_env=conda_env
)

loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

import pandas as pd
test_predictions = loaded_model.predict(pd.DataFrame(x_test))
print(test_predictions)