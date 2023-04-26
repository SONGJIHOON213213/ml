import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 
                 800,900,1000,210,420,350]]
               )

aaa = np.transpose(aaa)
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results) 

