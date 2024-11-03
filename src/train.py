import joblib
import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error

##############################

def train_cls(X_train, y_train, X_test, y_test, model_file, params, metric_score):
    # librerias
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV

    # RANDOM FOREST CON OPTUNA Y CV
    def objectiveRF(trial):
        # variables de optuna
        n_estimators = trial.suggest_int('n_estimators',20,200, step=20)
        max_depth = trial.suggest_int('max_depth', 2, 20, step=2)
        
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                   criterion='gini',
                                   max_depth=max_depth,
                                   n_jobs=-1,
                                   random_state=params['train']['random_state'])
        cv_score = np.mean(cross_val_score(rf, X_train, y_train, cv=params['train']['cv_folds']))
        return cv_score
    studyRF = optuna.create_study(direction='maximize')
    studyRF.optimize(objectiveRF, n_trials=params['train']['n_trials'])
    
    rf_model = RandomForestClassifier(n_estimators=studyRF.best_params['n_estimators'],
                               criterion='gini',
                               max_depth=studyRF.best_params['n_estimators'],
                               n_jobs=-1,
                               random_state=params['train']['random_state'])
    rf_model.fit(X_train, y_train)
    rf_metric = metric_score(y_test, rf_model.predict(X_test))
    joblib.dump(rf_model, 'models/trained_models/cls_rf.pkl')

    
    # XGBOOST con OPTUNA y CV de xgb
    obj_func = 'binary:logistic' if y_train.nunique()<=2 else 'multi:softprob'
    err_func = 'error' if y_train.nunique()<=2 else 'merror'
    err_col = 'test-error-mean' if y_train.nunique()<=2 else 'test-merror-mean'
    def objectiveXGB(trial):
        # variables de optuna
        learning_rate = trial.suggest_categorical('learning_rate',[0.01,0.05,0.1,0.5])
        max_depth = trial.suggest_int('max_depth', 2, 20, step=2)
        reg_lambda = trial.suggest_categorical('reg_lambda',[0.01,0.1,1,3])
        
        if err_func=='error':
                xg_params = {'objective':obj_func,
                     'learning_rate':learning_rate,
                     'max_depth':max_depth,
                     'reg_lambda':reg_lambda}
        else:            
            xg_params = {'objective':obj_func,
                         'num_class':y_train.nunique(),
                         'learning_rate':learning_rate,
                         'max_depth':max_depth,
                         'reg_lambda':reg_lambda}
    
        #Converting the data to DMatrix
        data = xgb.DMatrix(X_train, label=y_train)
    
        #Performing cross-validation
        cv_results = xgb.cv(xg_params, data, num_boost_round=10, nfold=5,metrics=err_func,\
                            seed=params['train']['random_state'])
        cv_score = cv_results[err_col].mean()
    
        return cv_score
    studyXGB = optuna.create_study(direction='minimize')
    studyXGB.optimize(objectiveXGB, n_trials=params['train']['n_trials'])
    
    xgb_model = xgb.XGBClassifier(
                        objective=obj_func,
                        learning_rate=studyXGB.best_params['learning_rate'],
                        max_depth=studyXGB.best_params['max_depth'],
                        reg_lambda=studyXGB.best_params['reg_lambda'],
                        random_state=params['train']['random_state'])
    xgb_model.fit(X_train, y_train)
    xgb_metric = metric_score(y_test, xgb_model.predict(X_test))
    joblib.dump(xgb_model, 'models/trained_models/cls_xgb.pkl')

    # SVM Classifier con RandomizedSearchCV
    from scipy.stats import uniform
    param_grid = {'C':uniform(0,2),
                  'kernel':['rbf','sigmoid']}
    
    SVM_model = RandomizedSearchCV(SVC(random_state=params['train']['random_state']),
                                  param_distributions=param_grid,
                                  n_iter=10,
                                  scoring='accuracy',
                                  n_jobs=-1)
    SVM_model.fit(X_train, y_train)
    SVM_metric = metric_score(y_test, SVM_model.predict(X_test))
    joblib.dump(SVM_model, 'models/trained_models/cls_svm.pkl')
    
    # Mejor modelo
    models_list = [rf_model, xgb_model, SVM_model]
    results = pd.DataFrame({'Modelo':['Random Forest','XGBoost','SVM'],'Metric':[rf_metric,xgb_metric,SVM_metric]})
    best_model_name = results.iloc[results.Metric.argmax(),0]
    print(f'Best Model {best_model_name} saved')
    best_model = models_list[results.Metric.argmax()]    
    joblib.dump(best_model, model_file)

def train_reg(X_train, y_train, X_test, y_test, model_file, params, metric_score, scoring):
    # librerias
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    
    # Ridge con RandomizedSearchCV
    from scipy.stats import uniform
    param_grid = {'alpha':uniform(0,2)}
    
    ridge_model = RandomizedSearchCV(Ridge(random_state=params['train']['random_state']),
                                  param_distributions=param_grid,
                                  n_iter=10,
                                  scoring=scoring,
                                  n_jobs=-1)
    ridge_model.fit(X_train, y_train)
    ridge_metric = metric_score(y_test, ridge_model.predict(X_test))
    joblib.dump(ridge_model, 'models/trained_models/reg_ridge.pkl')
    
    # Lasso con RandomizedSearchCV
    from scipy.stats import uniform
    param_grid = {'alpha':uniform(0,2)}
    
    lasso_model = RandomizedSearchCV(Lasso(random_state=params['train']['random_state']),
                                  param_distributions=param_grid,
                                  n_iter=10,
                                  scoring=scoring,
                                  n_jobs=-1)
    lasso_model.fit(X_train, y_train)
    lasso_metric = metric_score(y_test, lasso_model.predict(X_test))
    joblib.dump(lasso_model, 'models/trained_models/reg_lasso.pkl')
    
    # RandomForest con Optuna
    def objectiveRF(trial):
        # variables de optuna
        n_estimators = trial.suggest_int('n_estimators',20,200, step=20)
        max_depth = trial.suggest_int('max_depth', 2, 20, step=2)
        
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   criterion='squared_error',
                                   max_depth=max_depth,
                                   n_jobs=-1,
                                   random_state=params['train']['random_state'])
        cv_score = np.mean(cross_val_score(rf, X_train, y_train, cv=params['train']['cv_folds'], scoring=scoring))
        return cv_score
    studyRF = optuna.create_study(direction='maximize')
    studyRF.optimize(objectiveRF, n_trials=params['train']['n_trials'])
    
    rf_model = RandomForestRegressor(n_estimators=studyRF.best_params['n_estimators'],
                               criterion='squared_error',
                               max_depth=studyRF.best_params['n_estimators'],
                               n_jobs=-1,
                               random_state=params['train']['random_state'])
    rf_model.fit(X_train, y_train)
    rf_metric = metric_score(y_test, rf_model.predict(X_test))
    joblib.dump(rf_model, 'models/trained_models/reg_rf.pkl')
    
    # SVRegressor con Optuna
    def objectiveSVR(trial):
        # variables de optuna
        C = trial.suggest_categorical('C',[0.01,0.1,1,3])
        kernel = trial.suggest_categorical('kernel', ['linear','poly','rbf','sigmoid'])
        
        svr = SVR(kernel=kernel,
                   C=C)
        cv_score = np.mean(cross_val_score(svr, X_train, y_train, cv=params['train']['cv_folds'], scoring=scoring))
        return cv_score
    studySVR = optuna.create_study(direction='maximize')
    studySVR.optimize(objectiveSVR, n_trials=params['train']['n_trials'])
    
    svr_model = SVR(kernel=studySVR.best_params['kernel'],
                    C=studySVR.best_params['C'])
    svr_model.fit(X_train, y_train)
    svr_metric = metric_score(y_test, svr_model.predict(X_test))
    joblib.dump(svr_model, 'models/trained_models/reg_svr.pkl')
    
    # Mejor modelo
    models_list = [ridge_model, lasso_model, rf_model, svr_model]
    results = pd.DataFrame({'Modelo':['Ridge','Lasso','Random Forest','SVM'],\
                            'Metric':[ridge_metric,lasso_metric,rf_metric,svr_metric]})
    best_model_name = results.iloc[results.Metric.argmin(),0]
    print(f'Best Model {best_model_name} saved')
    best_model = models_list[results.Metric.argmax()]
    joblib.dump(best_model, model_file)

#####################################

def train(input_file, model_file, params_file):
  
    # Leer los archivos externos
    with open(params_file) as f:
        params = yaml.safe_load(f)
    df = pd.read_csv(input_file)

    # variables
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    tipo_problema = 'regression' if (y.nunique()>20) else 'classification'
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )
    
    # definición de metrica de desempeño de los modelos
    if tipo_problema == 'classification':
        metric_param = params['train']['metric_cls']
        if metric_param == 'accuracy':
            metric_score = accuracy_score
        elif metric_param == 'precision':
            metric_score = precision_score
        elif metric_param == 'f1':
            metric_score = f1_score
        else:
            metric_score = accuracy_score
    else:
        metric_param = params['train']['metric_reg']
        if metric_param == 'rmse':
            metric_score = root_mean_squared_error
            scoring = 'neg_root_mean_squared_error'
        elif metric_param == 'r2':
            metric_score = r2_score
            scoring = 'r2'
        elif metric_param == 'mape':
            metric_score = mean_absolute_percentage_error
            scoring = 'neg_mean_absolute_percentage_error'
        else:
            metric_score = root_mean_squared_error
            scoring = 'neg_root_mean_squared_error'
    
    if tipo_problema == 'classification':
        train_cls(X_train, y_train, X_test, y_test, model_file, params, metric_score)
    else:
        train_reg(X_train, y_train, X_test, y_test, model_file, params, metric_score, scoring)

############################################

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de hiperparámetros
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)