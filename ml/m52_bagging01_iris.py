from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
# dt = DecisionTreeClassifier(random_state=337)
# rf = RandomForestClassifier(random_state=337, n_jobs=-1)
# knn = KNeighborsClassifier()
# svm = SVC(probability=True, random_state=337)
# lr = LogisticRegression(random_state=337, n_jobs=-1)
# gb = GradientBoostingClassifier(random_state=337)
# xgb = XGBClassifier(random_state=337, n_jobs=-1)
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=10,
#                           n_jobs=-1,
#                           random_state=337,
#                           bootstrap=True,#디폴트
#                           )

# # RandomForestClassifier
# model_rf = BaggingClassifier(RandomForestClassifier(n_estimators=100, random_state=337),
#                              n_estimators=10,
#                              n_jobs=-1,
#                              random_state=337,
#                              bootstrap=True)

# # KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# model_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5),
#                               n_estimators=10,
#                               n_jobs=-1,
#                               random_state=337,
#                               bootstrap=True)



model = BaggingClassifier(
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('accuracy :', accuracy_score(y_test, y_pred))