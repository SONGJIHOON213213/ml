import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.datasets import load_wine, load_iris, fetch_covtype
from tensorflow.keras.datasets import cifar100
from sklearn.datasets import fetch_california_housing

# 데이터셋 리스트
datasets = [
    ("iris", load_iris(return_X_y=True)),
    ("covtype", fetch_covtype(return_X_y=True)),
    ("wine", load_wine(return_X_y=True)),
    ("digits", load_digits(return_X_y=True)),
    ("breast_cancer", load_breast_cancer(return_X_y=True)),
]

# 각 데이터셋에 대해 LDA 적용 후 결과 출력
for name, (x, y) in datasets:
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y) 
    print(f"LDA 모양 {name}: {x_lda.shape}")
    lda_EVR = lda.explained_variance_ratio_ 
    cumsum = np.cumsum(lda_EVR) 
    print(f"각 데이터들 {name}: {cumsum}")




#type x만 쓰는 이유
#y의 데이터 타입을 출력하는 것은 코드의 가독성을 높이지 않고, 오히려 불필요한 정보를 포함할 수 있습니다



# 정수형이라서 LDA에서 y의 클래스로 잘못인식해서 돌아감
# 성호는 캘리포니아 라운드처리 
# 그러다보니 그거도 정수형이라서 클래스로 인식
# 그래서 돌아감
# 회귀데이터 원칙적으로 에러인데
# 위처럼하면 가능


# np.cumsum() 함수는 배열의 원소들을 앞에서부터 차례대로 더해가며 누적 합계를 계산하는 함수입니다.

# 예를 들어, [1, 2, 3, 4]라는 배열이 있다면,

# 첫 번째 원소인 1을 그대로 더해줍니다. 결과는 1이 됩니다.
# 두 번째 원소인 2를 첫 번째 원소의 결과에 더해줍니다. 결과는 1 + 2 = 3이 됩니다.
# 세 번째 원소인 3을 두 번째 원소의 결과에 더해줍니다. 결과는 1 + 2 + 3 = 6이 됩니다.
# 네 번째 원소인 4를 세 번째 원소의 결과에 더해줍니다. 결과는 1 + 2 + 3 + 4 = 10이 됩니다. 
