import numpy as np

aaa = np.array([-10,2,3,4,5,6,7,8,1000,9,10,11,12,50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)
    print("q2 : ", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # -5
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    return np.where((data_out > upper_bound) | (data_out < lower_bound)) 
outliers_loc = outliers(aaa)
print("이상치의 위치: ", outliers_loc)  

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 이상치(outlier)란 무엇인가?
# 데이터 집합에서 일반적인 패턴에서 벗어나는 극단적인 값으로, 대부분의 값들이 위치한 범위에서 벗어난 값이다.
# 이상치는 측정값이 잘못되었거나, 실제로는 드문 경우지만 올바른 값일 수 있다.

# 이상치를 찾는 방법에는 무엇이 있나?
# Box Plot(상자그림)을 이용한 방법이 대표적이다.
# Box Plot은 데이터를 4분위(Quartile)로 분할하여, 데이터의 중심적 경향성과 분포도를 한눈에 볼 수 있는 그래프이다.
# 이상치를 찾기 위해, Box Plot에서 1사분위와 3사분위 사이의 범위를 IQR(Interquartile Range)라 하고,
# 이 범위를 벗어나는 데이터를 이상치로 간주한다.

# IQR * 1.5 값을 이용하여, 하한값(lower bound)과 상한값(upper bound)을 계산하고,
# 이 범위를 벗어나는 데이터를 이상치로 판단할 수 있다.
# 이상치를 찾는 방법에서 IQR * 1.5를 사용하는 이유는 무엇인가?
# 이상치를 찾는 방법 중 하나인 Box Plot에서 IQR * 1.5 값을 사용하는 이유는 통계학적인 규칙으로,

# 일반적으로 자료가 정규분포를 따르는 경우에는 대부분의 값이 평균과 가까운 범위에서 분포하고,
# 이에 비해 극단적인 값은 매우 드물게 발생하기 때문이다.
# 따라서 IQR * 1.5 값 이상 벗어나는 데이터를 이상치로 간주하는 것이 일반적인 규칙이다. 


# [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50] 데이터의 1사분위는 4.0이고, 중앙값은 7.0입니다. 

# 위에서 구한 1사분위(Q1)와 3사분위(Q3)를 이용하여 IQR을 계산합니다.

# Q1: 4.0
# Q3: 10.0
# IQR = Q3 - Q1 = 10.0 - 4.0 = 6.0

# 따라서, IQR은 6.0이 됩니다. 

# 데이터의 하위 75%에 해당하는 값이 10입니다.
# 즉, 13개의 데이터 중에서 가장 큰 값 50을 제외한 75%에 해당하는 값 중에서 가장 큰 값이 10입니다. 따라서 이 값은 10이 됩니다 

# 먼저 데이터를 정렬합니다: [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50]
# 25% 위치에 해당하는 값인 1사분위수(quartile_1)를 구합니다. 1사분위수는 4.0입니다.
# 50% 위치에 해당하는 값인 2사분위수(quartile_2)를 구합니다. 2사분위수는 7.0입니다.
# 75% 위치에 해당하는 값인 3사분위수(quartile_3)를 구합니다. 3사분위수는 10.0입니다.
# IQR(interquartile range)를 구합니다. IQR = quartile_3 - quartile_1 = 6.0
# lower bound와 upper bound를 구합니다. lower bound = quartile_1 - (iqr * 1.5) = -5, upper bound = quartile_3 + (iqr * 1.5) = 19
# 데이터에서 lower bound 보다 작거나 upper bound 보다 큰 값들을 이상치(outliers)로 간주합니다. 이 경우 -10과 50이 이상치입니다. 

# -10과 50은 각각 데이터의 최솟값과 최댓값으로, 데이터의 대부분의 값이 -10부터 12 사이에 분포하고 있기 때문에 
# -10보다 작거나 12보다 큰 값들은 이상치로 간주됩니다. 따라서 -10보다 작은 -10보다 큰 값 50은 이상치로 판단됩니다. 

# lower_bound = quartile_1 - (iqr * 1.5) = 4.0 - (6.0 * 1.5) = -5.0

# upper_bound = quartile_3 + (iqr * 1.5) = 10.0 + (6.0 * 1.5) = 19.0 

# 19보다 크거나 -5 보다 작은넘
# 이것은 위에서 계산한 상한선과 하한선을 기준으로, 해당 범위를 벗어나는 이상치 데이터를 찾기 위한 것입니다. 
# np.where 함수를 사용하여 이상치에 해당하는 데이터의 인덱스를 반환합니다. 
# 이상치를 제거하거나 대체하는 등의 후속 처리를 수행할 때 사용될 수 있습니다




