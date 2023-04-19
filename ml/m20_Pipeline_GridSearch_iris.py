import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes,load_iris, load_breast_cancer, load_digits, load_wine,fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_Pipeline
from sklearn.model_selection import GridSearchCV

# Define list of datasets to use
data_list = [load_iris, load_breast_cancer, load_digits, load_wine,load_wine, load_diabetes,fetch_california_housing]

# Define list of classifiers to use with their respective hyperparameters for GridSearch
algorithms_classifier = [('SVC', SVC), ('RandomForestClassifier', RandomForestClassifier)]
params_classifier = [    {'rf__n_estimators': [100, 200], 'rf__max_depth': [6, 8, 10]},
    {'rf__max_depth': [6, 8, 10, 12], 'rf__min_samples_leaf': [3, 5, 7, 10]},
    {'rf__min_samples_leaf': [3, 5, 7, 10], 'rf__min_samples_split': [2, 3, 5, 10]},
    {'rf__min_samples_split': [2, 3, 5, 10]},
    {'rf__min_samples_split': [2, 3, 5, 10]}
]

# Define list of scalers to use
scaler_list = [MinMaxScaler(), StandardScaler()]

for data in data_list:
    # Load data
    x, y = data(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)
    
    for scaler in scaler_list:
        # Scale data
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        for name, algorithm in algorithms_classifier:
            # pipeline = Pipeline([('scaler', scaler), (name, algorithm())])
            pipeline = make_Pipeline([('scaler', scaler), (name, algorithm())])
     
            params = {}
            for param in params_classifier:
                if name in param:
                    for key, value in param.items():
                        new_key = name + '__' + key
                        params[new_key] = value
            
            model = GridSearchCV(pipeline, params, cv=5)
            model.fit(x_train_scaled, y_train)
            score = model.score(x_test_scaled, y_test)
            print(f'{name}, {type(scaler).__name__}, {data.__name__}: {score:.4f}')