import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


class MultipleLinearRegression:

    def __init__(self, X_train, y_train, learning_rate=1e-10, epochs=30000):

        self.n, self.p_minus_one = X_train.shape

        self.X_train = np.ones((self.n, self.p_minus_one + 1)) # p_minus_one is the number of features we have, we need to add one because of the biais vector
        self.X_train[:, 1:] = X_train
        self.y_train = y_train
        self.parameters = np.zeros(self.p_minus_one + 1)

        self.learning_rate = learning_rate
        self.epochs = epochs


    def gradient_descent(self):

        predictions = np.dot(self.X_train, self.parameters)
        errors = predictions - self.y_train
        gradient = (2/self.n) * np.dot(self.X_train.T, errors)

        return self.parameters - self.learning_rate * gradient


    def train(self):

        for i in range(self.epochs):
            self.parameters = self.gradient_descent()


    def predict(self, X):

        n_samples = X.shape[0]

        X_with_bias = np.ones((n_samples, self.p_minus_one + 1))
        X_with_bias[:, 1:] = X

        return np.dot(X_with_bias, self.parameters)
    
X, y = make_regression(
    n_samples=1000,
    n_features=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultipleLinearRegression(X_train, y_train, learning_rate=0.001, epochs=5000)

model.train()

y_pred_custom = model.predict(X_test)

print("Custom MSE:", mean_squared_error(y_test, y_pred_custom))
print("Custom R2:", r2_score(y_test, y_pred_custom))
print("Custom coefficients:", model.parameters)
print("Custom prediction:", y_pred_custom[:5])