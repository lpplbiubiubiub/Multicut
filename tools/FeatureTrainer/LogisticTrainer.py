from BasicTrainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import numpy as np
import time
import pickle


class LogisticTrainer(BaseTrainer):
    def __init__(self, X, Y, iter=1000., test_size=0.2):
        super(LogisticTrainer, self).__init__(X, Y, iter, test_size)
        # self.lr = LogisticRegression(max_iter=iter, random_state=0, fit_intercept=False)
        self.lr = LogisticRegression(C=1000.0, random_state=0, fit_intercept=False)
        # self.lr = svm.SVC()
        # self.lr = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        np.random.seed(int(time.time()))

    def fit(self, balance_sample=False):
        """
        train data
        :return: 
        """
        if balance_sample:
            true_idx, false_idx = self.Y_train == 1, self.Y_train == 0
            true_train_sample, false_train_sample = self.X_train_std[true_idx], self.X_train_std[false_idx]
            sample_false_idx = np.random.choice(np.arange(self.X_train_std.shape[0])[false_idx], size=int(true_train_sample.shape[0]*0.5))
            false_train_sample = self.X_train_std[sample_false_idx]
            y_false = np.zeros(shape=(false_train_sample.shape[0], 1))
            y_true = np.ones(shape=(true_train_sample.shape[0], 1))
            self.Y_train = np.vstack((y_true, y_false))
            self.Y_train = self.Y_train.reshape(self.Y_train.shape[0], )
            self.X_train_std = np.vstack((true_train_sample, false_train_sample))
        self.lr.fit(self.X_train_std, self.Y_train)
        print(self.lr.coef_.shape)

    def val(self):
        # split by class
        X_test_true = self.X_test_std[self.Y_test == 1]
        Y_test_true = self.Y_test[self.Y_test == 1]
        X_test_false = self.X_test_std[self.Y_test == 0]
        Y_test_false = self.Y_test[self.Y_test == 0]
        pre_res_true = self.lr.predict(X_test_true)
        pre_res_false = self.lr.predict(X_test_false)
        res_true, res_false = pre_res_true == Y_test_true, pre_res_false == Y_test_false
        pre_res = self.lr.predict(self.X_test_std)
        pre_res = pre_res == self.Y_test
        print("correct ratio is {:.2f}".format(np.sum(pre_res) / (0. + pre_res.shape[0])),
              "true sample correct ratio is {:.2f}".format(np.sum(res_true) / (0. + pre_res_true.shape[0])),
              "false sample correct ratio is {:.2f}".format(np.sum(res_false) / (0. + pre_res_false.shape[0])),)

    def get_coef_vec(self, X):
        assert self.lr.coef_.shape[1] == X.shape[1], "dim1 {} doesn't equal dims {}".format(self.lr.coef_.shape[1] , X.shape[1])
        X_trandform = self.sc.transform(X)
        # X_trandform2 = (X - self.sc.mean_) / self.sc.std_
        raw_output = np.matmul(X_trandform, self.lr.coef_.reshape(self.lr.coef_.shape[1], ))
        return raw_output

    def guest_param_val(self, coef, mean, std):
        x_std = (self.X_test - mean) / std
        y_pre = np.matmul(x_std, coef) > 0
        res = y_pre == self.Y_test
        print "correct ratio is {}".format(np.sum(res) / (res.shape[0] + 0.))

    def save_model(self, dst_file):
        with open(dst_file, "w") as f:
            params = self.lr.coef_
            np.savetxt(f, params)

    def load_coef(self, coef_file):
        coef = np.loadtxt(coef_file)
        self.lr.coef_ = coef[np.newaxis, :]


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    lg_trainer = LogisticTrainer(X, y)
    lg_trainer.fit()
    lg_trainer.val()