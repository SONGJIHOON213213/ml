import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


# 1.1 경로, 가져오기
path = 'c:/study/_data/kaggle_bike/'
df = pd.read_csv(path + 'train.csv', index_col=0)
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

# 1.2 train, test split
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

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
