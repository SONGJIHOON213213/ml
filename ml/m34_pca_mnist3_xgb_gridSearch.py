from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.datasets import mnist
# define a function to run the model and print the results
def run_model(x, y, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, stratify=y)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    parameters = [
        {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
        {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1]},
        {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.8]}
    ]
    
    model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0), parameters, cv=kfold, verbose=1, n_iter=5, n_jobs=-1)
    model.fit(x_train, y_train)

    print('==========================' + name + '==========================')
    print('Accuracy score: ' + str(accuracy_score(y_test, model.best_estimator_.predict(x_test))))
 
# load and run the model for several datasets
for data_loader in [mnist]:
    x, y = data_loader(return_X_y=True)
    pca = PCA(n_components=x.shape[1] // 2)
    x = pca.fit_transform(x)
    run_model(x, y, data_loader.__name__)

# load and run the model for the diabetes dataset
datasets = pd.read_csv('c:/study/_data/dacon_diabete/train.csv', index_col=0)
x = datasets.drop(datasets.columns[-1], axis=1)
y = datasets.iloc[:, -1]
pca = PCA(n_components=x.shape[1] // 2)
x_pca = pca.fit_transform(x)
run_model(x_pca, y, 'dacon_diabetes')

# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# =====================================load_iris=====================================
# acc : 1.0
# best acc : 1.0
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# =====================================load_breast_cancer=====================================
# acc : 0.956140350877193
# best acc : 0.956140350877193
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# =====================================load_wine=====================================
# acc : 1.0
# best acc : 1.0
# Fitting 5 folds for each of 5 candidates, totalling 25 fits

