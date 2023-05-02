import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


# 1.1 경로, 가져오기
path = 'c:/study/_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape) #(652, 9) (116, 8)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# # 1.5 train, test 분리 과적합방지
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True)
# 1.3 결측치 처리
num_cols = x_train.select_dtypes(include=np.number).columns
x_train[num_cols] = x_train[num_cols].fillna(x_train[num_cols].median())
x_test[num_cols] = x_test[num_cols].fillna(x_train[num_cols].median())

# 1.4 라벨인코딩
le = LabelEncoder()
for i in x_train.columns:
    if x_train[i].dtype == 'object':
        x_train[i] = le.fit_transform(x_train[i])
        x_test[i] = le.transform(x_test[i])

print(len(x_train.columns))
print(x_train.info())
print(x_test.info())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = BaggingClassifier(
    n_estimators=12,
    n_jobs=-1,
    random_state=337,
    bootstrap=True
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('accuracy :', accuracy_score(y_test, y_pred))
