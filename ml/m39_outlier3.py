import numpy as np
import pandas as pd

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    q2 = np.median(data_out)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outliers_val = data_out[(data_out > upper_bound) | (data_out < lower_bound)]
    return {
        'Q1': quartile_1,
        'Q3': quartile_3,
        'IQR': iqr,
        'outliers_loc': np.where((data_out > upper_bound) | (data_out < lower_bound))[0].tolist(),
        'outliers_val': outliers_val.tolist(),
    }

# 데이터프레임 생성
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [-100,200,30,400,500,600,70000,800,900,1000,210,420,350]])
df = pd.DataFrame(aaa.T, columns=['col1', 'col2'])

# 각 열에 대한 이상치 정보 출력
for col in df.columns:
    print(f"{col}:")
    info = outliers(df[col])
    print(f" - Q1: {info['Q1']:.2f}")
    print(f" - Q3: {info['Q3']:.2f}")
    print(f" - IQR: {info['IQR']:.2f}")
    print(f" - 이상치 위치: {info['outliers_loc']}")
    print(f" - 이상치 값: {info['outliers_val']}\n")
# 모든 열에 대한 이상치 위치 출력
outliers_loc_all = []
for col in df.columns:
    outliers_loc = outliers(df[col])
    outliers_loc_all.extend(outliers_loc)
print(f"모든 열의 이상치 위치: {list(set(outliers_loc_all))}")

# 즉, 첫 번째 row의 첫 번째 column 값(-10)과 두 번째 row의 일곱 번째 column 값(70000)이 이상치로 간주되어 이상치의 위치가 반환됩니다.

# 1사분위 : 6.25
# q2 :  21.0
# 3사분위 : 415.0
# iqr :  408.75

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show
# 첫 번째 column의 값(-10)은 lower bound인 -5보다 작으며, 두 번째 row의 일곱 번째 column 값(70000)은 
# upper bound인 19보다 크기 때문에 이상치로 판단됩니다. 이상치는 데이터의 분포와 동떨어져 있거나 
# 다른 값들과 크게 차이가 나는 값으로 정의됩니다.

############실습##############
# 각 컬럼별 이상치 표시
#(13,2) 가 나오게 수정 

#dataframe 컬럼별로 나올수있게 수정