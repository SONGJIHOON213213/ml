from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import matplotlib.pyplot as plt

# 데이터 생성
x, y = make_blobs(n_samples=50, centers=2, cluster_std=1, random_state=337)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

ax1.set_title('오리지널', fontfamily='Malgun Gothic')
ax1.scatter(x[:, 0], x[:, 1], c=y, edgecolors='black')

qt = QuantileTransformer(n_quantiles=50)
x_qt = qt.fit_transform(x)

ax2.scatter(x_qt[:, 0], x_qt[:, 1], c=y, edgecolors='black')
ax2.set_title('QuantileTransformer')


pt = PowerTransformer()
x_pt = pt.fit_transform(x)

ax3.scatter(x_pt[:, 0], x_pt[:, 1], c=y, edgecolors='black')
ax3.set_title('PowerTransformer')


scaler = StandardScaler()
x_ss = scaler.fit_transform(x)

ax4.scatter(x_ss[:, 0], x_ss[:, 1], c=y, edgecolors='black')
ax4.set_title('StandardScaler')

plt.tight_layout()

plt.show()