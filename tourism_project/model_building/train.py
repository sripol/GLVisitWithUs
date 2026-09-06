import pandas as pd
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

os.system("pip install mlflow")
import mlflow

mlflow.set_tracking_uri("file:./logs")
mlflow.set_experiment("Tourism_prediction")

# These files are generated in the current working directory by prep.py in the pipeline
x_train = pd.read_csv("tourism_project/model_building/x_train.csv")
y_train = pd.read_csv("tourism_project/model_building/y_train.csv").squeeze()

with mlflow.start_run():
    mlflow.log_param("Balanced y_train values", y_train.value_counts().to_dict())
    
    # Handle Class Imbalance
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

    categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
    numeric_features = ["Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups", "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome"]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=ratio)
    pipeline = make_pipeline(preprocessor, model)

    grid_search = GridSearchCV(pipeline, {'xgbclassifier__n_estimators': [100, 200]}, cv=3, scoring='f1')
    grid_search.fit(x_train, y_train)

    mlflow.log_param("Best model parameters", grid_search.best_params_)
    mlflow.log_metric("Best model score", grid_search.best_score_)

    deployment_path = "tourism_project/deployment"
    os.makedirs(deployment_path, exist_ok=True)
    model_file = os.path.join(deployment_path, "model.joblib")

    joblib.dump(grid_search.best_estimator_, model_file)
    print(f"Balanced model saved successfully to {model_file}")
