# Mice(Multiple Iputation by chained Equations)
import numpy as np
import pandas as pd 
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute  import SimpleImputer,KNNImputer # 결측치 책임 돌리기
from sklearn.impute  import IterativeImputer
from xgboost import XGBClassifier, XGBRegressor
from impyute.imputation.cs import mice
data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8],
                     [2,4,6,8,10],
                     [np.nan,4,8,np.nan]]
                    ).transpose()
data.columns = ['x1','x2','x3','x4']



impyute_df = mice(data.values)
print(impyute_df)




# [[ 2.          2.          2.         -0.02090859]
#  [ 4.02159461  4.          4.          4.        ]
#  [ 6.          6.01140005  6.          8.        ]
#  [ 8.          8.          8.         11.99895075]
#  [10.         10.         10.         16.00557053]]