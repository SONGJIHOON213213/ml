def Runmodel(x, y):
    imputer = IterativeImputer(RandomForestRegressor())
    scaler = MinMaxScaler()
    x = imputer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model.__class__.__name__, 'result : ', result)

path = 'c:/study/_data/kaggle_bike/'
kaggle_bike = pd.read_csv(path + 'train.csv', index_col=0)
kaggle_bike_x, kaggle_bike_y = split_xy(kaggle_bike)

outliers_loc = outliers(kaggle_bike_x)
for i in range(kaggle_bike_x.shape[1]):
    kaggle_bike_x[outliers_loc[i], i] = np.nan

Runmodel(kaggle_bike_x, kaggle_bike_y)