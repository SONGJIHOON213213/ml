import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')
#1.데이터
x,y, = load_boston(return_X_y=True)
# x_train,x_test,y_train,y_test = train_test_split(
#     x,y,shuffle=True, random_state=123,test_size=0,2,
    
# )

n_splits = 5
kfold = KFold()
# kfold = KFold(n_splits=n_splits, shuffle=True,random_state=123)

#2.모델구성
model = LinearSVC()


#3, 4. 컴파일,훈련,평가.예측
scores =cross_val_score(model,x,y, cv=kfold)
print('ACC: ', scores,'\n croos_val_score 평균 : ', round(np.mean(scores),4))