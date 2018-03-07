import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


class BaseTrainer(object):
    """
    base trainer:
    get data, standare it and train it then check proform
    """
    def __init__(self, X, Y, iter=1000, test_size=0.2):
        arr = np.arange(X.shape[0])
        np.random.shuffle(arr)
        X_c = X[arr]
        Y_c = Y[arr]
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X_c, Y_c, test_size=test_size, random_state=0)
        x_train_true = self.X_train[self.Y_train==1][:, 0]
        self.sc = StandardScaler()
        self.sc.fit(self.X_train)
        # self.X_train_std = self.sc.transform(self.X_train)
        # self.X_test_std = self.sc.transform(self.X_test)
        self.X_train_std = self.X_train
        self.X_test_std = self.X_test


