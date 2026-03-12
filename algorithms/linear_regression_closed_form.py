import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class LinearRegressionClosedForm:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

        A = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        self.intercept_ = A[0]
        self.coef_ = A[1:]

    def predict(self, X):
        X = np.array(X)

        return X @ self.coef_ + self.intercept_
    
X_train, y_train = fetch_california_housing(return_X_y=True)

my_model = LinearRegressionClosedForm()
sk_model = LinearRegression()

my_model.fit(X_train, y_train)
sk_model.fit(X_train, y_train)

y_pred_ = my_model.predict(X_train)
y_pred_sk = sk_model.predict(X_train)

print(f"The r2-score for my model: {r2_score(y_true=y_train, y_pred=y_pred_)}")
print(f"The r2-score for sklearn model: {r2_score(y_true=y_train, y_pred=y_pred_sk)}")