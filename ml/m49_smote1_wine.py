import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

path = 'c:/study/_data/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

enc = LabelEncoder()
enc.fit(train_csv['type'])
train_csv['type'] = enc.transform(train_csv['type'])
test_csv['type'] = enc.transform(test_csv['type'])

ohe = OneHotEncoder()
y = train_csv['quality'].values
y = y.reshape(-1, 1)
y = ohe.fit_transform(y).toarray()

x = train_csv.drop(['quality'], axis=1)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

test_csv_df = pd.DataFrame(test_csv, columns=train_csv.columns[1:])
test_csv_df = scaler.transform(test_csv_df) 
test_csv_df = pd.DataFrame(test_csv_df, columns=train_csv.columns[1:])

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=3377, stratify=y)

smote = SMOTE(random_state=3377)
x_train, y_train = smote.fit_resample(x_train, y_train) 

model = RandomForestClassifier(random_state=3377)
model.fit(x_train, y_train)

y_pred = model.predict(x_val)
score = model.score(x_val, y_val)

print('After SMOTE')
print('model.score: ', score)
print('accuracy_score: ', accuracy_score(y_val, y_pred))
print('f1_score(macro): ', f1_score(y_val, y_pred, average='macro'))
print('f1_score(micro): ', f1_score(y_val, y_pred, average='micro'))
print(pd.Series(y_train).value_counts().sort_index()) 