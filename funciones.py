import argparse
import subprocess
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_wine
from pyngrok import ngrok

def argumentos():
    parser = argparse.ArgumentParser(description='__main__ de la aplicación con argumentos de entrada.')
    parser.add_argument('--nombre_job', type=str, help='Nombre del experimento en MLflow.')
    return parser.parse_args()

def load_dataset():
    wine = load_wine()
    df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    df['target'] = wine['target']
    return df

def data_treatment(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test_target = test['target']
    test[['target']].to_csv('test-target.csv', index=False)
    del test['target']
    test.to_csv('test.csv', index=False)

    features = [x for x in list(train.columns) if x != 'target']
    x_raw = train[features]
    y_raw = train['target']

    X_train, X_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=0.20,
                                                        random_state=123,
                                                        stratify=y_raw)
    return X_train, X_test, y_train, y_test

def setup_ngrok(port=6000):
    ngrok.set_auth_token('2j9OBP1rEMPvAVn8lRDCAgr273D_7PvtaDu2Wt9ey8crPjJRi')
    ngrok.kill()
    public_url = ngrok.connect(port)
    print("ngrok URL:", public_url)
    return public_url

def mlflow_tracking(nombre_job, X_train, X_test, y_train, y_test, port=6000):
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', str(port)])
    print(mlflow_ui_process)
    time.sleep(5)
    mlflow.set_experiment(nombre_job)

    models_params = {
        'LogisticRegression': (LogisticRegression(max_iter=1000), {
            "model__C": [0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"]
        }),
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        })
    }

    for model_name, (model, params) in models_params.items():
        with mlflow.start_run(run_name=model_name) as run:
            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])
            
            grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            accuracy_train = grid_search.score(X_train, y_train)
            accuracy_test = grid_search.score(X_test, y_test)
            y_pred = grid_search.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_params(best_params)
            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.sklearn.log_model(grid_search.best_estimator_, model_name)

            report_df = pd.DataFrame(report).transpose()
            report_path = f"{model_name}_classification_report.csv"
            report_df.to_csv(report_path, index=True)
            mlflow.log_artifact(report_path)
    
    print("Se ha acabado el entrenamiento de los modelos correctamente")
