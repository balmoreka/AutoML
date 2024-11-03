import sys
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(params_file, output_file, output_prep):
    with open(params_file) as f:
        params = yaml.safe_load(f)
    input_file = 'datasets/'+params['dataset']['name']
    
    df = pd.read_csv(input_file)
    
    # separar caracteristicas
    features_name = [x for x in list(df.columns) if x!='target']
    features = df[features_name]
    target = df[['target']]

    # definición de tipo de features
    numeric_features = []
    categoric_features = []
    for f in features_name:
        if (features[f].nunique()<=20):
            categoric_features.append(f)
        else:
            numeric_features.append(f)   
    
    # Ajuste de features
    feat_num = features[numeric_features].select_dtypes('O')
    for f in feat_num.columns:
        feat_num.loc[(~feat_num[f].str.isnumeric()), f] = np.nan
    feat_num = feat_num.astype('float32')
    feat_num = pd.concat([features[numeric_features].select_dtypes('number'),feat_num], axis=1)
    feat_cat = features[categoric_features]
    features = pd.concat([feat_num, feat_cat], axis=1)
    
    # preprocesamiento
    numeric_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categoric_features)
        ]
    )
    df = pd.DataFrame(preprocessor.fit_transform(features))
    # guardar preprocesador
    joblib.dump(preprocessor, output_prep)
    
    df = pd.concat([df, target], axis=1, ignore_index=True)
    df.to_csv(output_file, index=False)
    print('Preprocessing stage is done')

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo de salida, características y objetivo
    params_file = sys.argv[1]
    output_file = sys.argv[2]
    output_prep = sys.argv[3]
    
    preprocess(params_file, output_file, output_prep)