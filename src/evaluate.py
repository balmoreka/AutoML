import pandas as pd
import joblib
import json
import sys
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate(input_file, model_file, metrics_file, params_file):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los parámetros
    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # variables
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    tipo_problema = 'regression' if (y.nunique()>20) else 'classification'
    
    # Cargar el modelo entrenado
    model = joblib.load(model_file)

    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular métricas
    if tipo_problema == 'classification':
        acc = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        with open(metrics_file, 'w') as f:
            json.dump({'accuracy': acc, 'f1': f1}, f, indent=4)
        print(f"Métricas guardadas en {metrics_file}")
    
    else:
        rmse = root_mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        with open(metrics_file, 'w') as f:
            json.dump({'accuracy': acc, 'f1': f1}, f, indent=4)
        print(f"Métricas guardadas en {metrics_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de métricas, archivo de parámetros
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    evaluate(input_file, model_file, metrics_file, params_file)
