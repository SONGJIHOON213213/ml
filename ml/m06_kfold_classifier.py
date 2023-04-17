from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

data_list = [load_iris(return_X_y=True), load_breast_cancer(return_X_y=True), load_wine(return_X_y=True),load_digits(return_X_y=True)]
model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
data_name_list = ['아이리스이병헌 : ', '유빙암 : ', '와인먹고싶냐? : ','뒤짓래스? : ',]
model_name_list = ['LinearSVC : ', 'LogisticRegression :', 'DecisionTreeClassifier : ', 'RandomForestClassifier : ']

n_splits = 5  # number of folds for k-fold cross-validation
random_state = 42  # set random state for reproducibility
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

scaler_list = [RobustScaler()]

for data, data_name in zip(data_list, data_name_list):
    x, y = data
    print(data_name)
    best_model = None
    best_acc = 0.0
    
    for scaler in scaler_list:
        try:
            x_scaled = scaler.fit_transform(x)
        except ValueError:
            print(f"{scaler.__class__.__name__} scaler is not compatible with {data_name}")
            continue  # skip the rest of the loop if the scaler is not compatible with the dataset
        
        for model, model_name in zip(model_list, model_name_list):
            try:
                acc_list = []  # store accuracies for each fold

                for fold, (train_idx, test_idx) in enumerate(kf.split(x_scaled, y)):
                    x_train, x_test = x_scaled[train_idx], x_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model.fit(x_train, y_train)
                    y_predict = model.predict(x_test)
                    acc = accuracy_score(y_test, y_predict)
                    acc_list.append(acc)

                mean_acc = np.mean(acc_list)

                if mean_acc > best_acc:
                    best_model = model
                    best_acc = mean_acc

            except ValueError:
                print(f"{model_name} is not compatible with {data_name} dataset and {scaler.__class__.__name__} scaler")
                continue  # skip the rest of the loop if the model is not compatible with the dataset and scaler

        print(f"with {scaler.__class__.__name__} the best model for {data_name} dataset is {best_model.__class__.__name__}, accuracy: {round(best_acc, 3)}")