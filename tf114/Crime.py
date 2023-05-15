import pandas as pd
import numpy as np
import random
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(40) # Seed 고정

# pd.read_csv() 함수를 사용해서 데이터를 읽어오는 코드입니다.
train = pd.read_csv('c:/study/_data/crime/train.csv')
test = pd.read_csv('c:/study/_data/crime/test.csv')
submit = pd.read_csv('c:/study/_data/crime/sample_submission.csv')

# 데이터를 확인하기 위해 head() 함수를 사용합니다.
train.head(5)
x_train = train.drop(['ID', 'TARGET'], axis = 1)
y_train = train['TARGET']
print(train.columns)
x_test = test.drop('ID', axis = 1)
ordinal_features = ['요일', '범죄발생지','연기/연무','눈날림']

for feature in ordinal_features:
    le = LabelEncoder()
    le = le.fit(x_train[feature])
    x_train[feature] = le.transform(x_train[feature])

    # x_train데이터에서 존재하지 않았던 값이 x_test 데이터에 존재할 수도 있습니다.
    # 따라서 x_test 데이터를 바로 변형시키지 않고 고윳값을 확인후 x_test 데이터를 변환합니다.
    for label in np.unique(x_test[feature]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    x_test[feature] = le.transform(x_test[feature])
    # 모델 인자에 random_state를 넣음으로써 시드고정의 효과를 얻을 수 있습니다.
model = DecisionTreeClassifier(random_state = 337)
model.fit(x_train, y_train)
# predict() 함수는 독립변수(테스트데이터)를 입력받았을 때 종속변수를 예측합니다.
pred = model.predict(x_test)
# 제출 파일을 읽어옵니다.
submit = pd.read_csv('c:/study/_data/crime/sample_submission.csv')

# 예측한 값을 TARGET 컬럼에 할당합니다.
submit['TARGET'] = pred
submit.head()

# 테스트 데이터의 target 값 불러오기

# 모델 예측값과 실제값을 이용해 Macro F1 Score를 계산합니다.


# 결과를 출력합니다.\
    

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv('c:/study/_data/crime/4submit.csv', index = False)