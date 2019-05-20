import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# 定义Linear Regression
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self, X, y, method='gd'):
        if method == 'gd':
            return self.fit_gd(X, y)
        else:
            return self.fit_normal(X, y)

    def fit_normal(self, X, y):
        assert X.shape[0] == y.shape[0], "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X, y, eta=0.01, n_iters=1e4):
        assert X.shape[0] == y.shape[0], "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X), 1)), X])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X):
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b.dot(self._theta)

    def score(self, X, y):
        y_hat = self.predict(X)
        return r2_score(y, y_hat)

    def __repr__(self):
        return "Linear Regression"


# 读入数据文件，整个训练数据集
data = pd.read_csv('./train.csv')
# 只选择其中的PM2.5
data = data[data['observation'] == 'PM2.5']
# 去除掉其中的无关行
data = data.drop(['Date', 'stations', 'observation'], axis=1)
# 进行x和y的划分
X = []
y = []
for i in range(15):
    X.append(np.array(data.iloc[:, i:i + 9]))
    y.append(np.array(data.iloc[:, i + 9]))
X = np.reshape(X, (-1, 9))
y = np.reshape(y, (-1)).astype('int')
# 划分训练集和测试集
train_X = X[:int(len(X) * 0.8)]
test_X = X[int(len(X) * 0.8):]
train_y = y[:int(len(X) * 0.8)]
test_y = y[int(len(X) * 0.8):]
# 对所有的数据归一化
ss = StandardScaler()
ss.fit(train_X)
train_X = ss.transform(train_X)
test_X = ss.transform(test_X)
# 调用linear regression
LR = LinearRegression().fit(train_X, train_y)
print(LR.score(train_X, train_y))
print(LR.score(test_X, test_y))
