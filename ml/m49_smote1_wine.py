import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape,y.shape) 
print(np.unique(y,return_counts=True))
print(pd.Series(y).value_counts().sort_index())


# print(y) 
# x= x[:-25] 

x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.75,shuffle=True,random_state=3377,stratify=y
    
) 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=3377)

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predcit(x_test)
score = model.score(x_test,y_test)
print('model.score: ', score) 
print('accuracy_score : ',accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict,average='micro'))

print("=====================================SMOTE 적용후========")
smote = SMOTE(random_state=337)
x_train,y_train = smote.fit_resample(x_train,y_train)
print(x_train.shape,y_train.shape) 
print(pd.Series(y_train).value_counts().sort_index())
