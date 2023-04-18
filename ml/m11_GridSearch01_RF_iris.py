from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import time

# load the data
X, y = load_iris(return_X_y=True)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=337, test_size=0.2, stratify=y)

# define the hyperparameter grid

parameters = [
    {'n_estimators': [200], 'max_depth': [12], 'min_samples_leaf': [10], 'min_samples_split': [10], 'n_jobs': [4]}
]

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [6, 8, 10, 12],
#     'min_samples_leaf': [3, 5, 7, 10],
#     'min_samples_split': [2, 3, 5, 10],
#     'n_jobs': [-1, 2, 4]
# }


# create a random forest classifier object
rfc = RandomForestClassifier(random_state=337)

# create a grid search object
grid_search = GridSearchCV(rfc, param_grid=parameters, cv=3)

# fit the grid search object to the data
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# print the best hyperparameters and score
print("최적의 매개변수:", grid_search.best_params_)
print("best_score_ :", grid_search.best_score_)

# evaluate the best estimator on the test set
y_pred_best = grid_search.best_estimator_.predict(X_test)
print('ACC 최적튠:', accuracy_score(y_test, y_pred_best))

print("걸린시간:", round(end_time - start_time, 2), '초')
