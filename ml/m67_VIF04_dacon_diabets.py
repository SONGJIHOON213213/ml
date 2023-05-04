from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
# 1. Load the dataset
#1 데이터
# 1.1 경로, 가져오기
diabets_path = 'c:/study/_data/dacon_diabets/'

df = pd.read_csv(diabets_path + 'train.csv', index_col=0)

ddarung = pd.read_csv(diabets_path + 'train.csv', index_col=0).dropna()
df['Outcome'] = df.target
print(df)

# 2. Split the dataset into x and y
y = df['Outcome']
x = df.drop(['Outcome'], axis=1)

# 3. Perform VIF analysis
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
vif = pd.DataFrame()
vif['variables'] = x.columns
vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
print(vif)

# 4. Remove highly correlated features and scale the input features
x = x.drop(['Glucose', 'Age'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=337, test_size=0.2)
scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

# 5. Build and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=337)
model.fit(x_train, y_train)

# 6. Evaluate the model's performance
results = model.score(x_test, y_test)
print("Results: ", results)