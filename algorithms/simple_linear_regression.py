import pandas as pd
import matplotlib.pyplot as plt
import random

class SimpleLinearRegression:
    def __init__(self):
        self.w_now = random.random() # Initializing current weight and bias randomly
        self.b_now = random.random()

    def gradient_descent(self, x_train, y_train, learning_rate):
        w_gradient = 0
        b_gradient = 0
        n = len(x_train)
        for i in range(n):
            w_gradient += -(2/n) * (y_train[i] - (self.w_now * x_train[i] + self.b_now)) * x_train[i]
            b_gradient += -(2/n) * (y_train[i] - (self.w_now * x_train[i] + self.b_now))
        w = self.w_now - learning_rate * w_gradient
        b = self.b_now - learning_rate * b_gradient
        return w, b
    
    def fit(self, x_train, y_train, learning_rate, epochs):
        for i in range(epochs):
            if (i % 50 == 0):
                print(f"epoch: {i}")
            self.w_now, self.b_now = self.gradient_descent(x_train, y_train, learning_rate)
        return self.w_now, self.b_now
    
    def predict(self, value):
        return self.w_now * value + self.b_now

data = pd.read_csv("C:/Users/pc/machine-learning-algorithms/datasets/Salary_Data.csv")
x_train = data['YearsExperience']
y_train = data['Salary']
L = 0.0001
epochs = 30000

lr = SimpleLinearRegression()
w, b = lr.fit(x_train, y_train, L, epochs)

plt.scatter(data.YearsExperience, data.Salary, color="black")
plt.plot(list(range(1, 11)), [lr.predict(x) for x in range(1, 11)], color="red")
plt.show()