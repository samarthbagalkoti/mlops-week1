import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.randn(200, 3)
y = (X[:, 0] + 0.5*X[:, 1] > 0).astype(int)

clf = LogisticRegression().fit(X, y)
print("Training accuracy:", clf.score(X, y))

