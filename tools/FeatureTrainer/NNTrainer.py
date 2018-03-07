from BasicTrainer import BaseTrainer
import torch as T
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import numpy as np
import time


class Maxout(nn.Module):
    """
    
    """
    def __init__(self, d_in, d_out, pool_size):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class BasicNet(nn.Module):
    """
    Pass
    """
    def __init__(self, input_size=3, output=1):
        super(BasicNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output),
        )

    def forward(self, x):
        return self.model(x)


class NNTrainer(BaseTrainer):
    def __init__(self, X, Y, iter=1000, test_size=0.2, input_size=3, output=2):
        super(NNTrainer, self).__init__(X, Y, iter, test_size)
        self.lr = BasicNet(input_size, output)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.lr.parameters(), lr=1e-3, momentum=0.9)
        # self.optimizer = optim.Adam(self.lr.parameters(), lr=1e-4)
        self.epoch_size = iter
        self.scheduler = StepLR(self.optimizer, step_size=4000, gamma=0.1)
        np.random.seed(int(time.time()))

    def fit(self, balance_sample=False):
        """
        train data
        :return: 
        """
        #
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
        self.lr.train()
        for i in range(self.epoch_size):
            self.scheduler.step(i)
            X = Variable(T.from_numpy(self.X_train_std).float())
            Y = Variable(T.from_numpy(self.Y_train).long())
            self.optimizer.zero_grad()
            pred = self.lr(X)
            loss = self.criterion(pred, Y)
            loss.backward()
            self.optimizer.step()
            _, predicted = T.max(pred.data, 1)
            acc = T.sum(predicted == Y.data) / float(predicted.size()[0])
            # print("epoch {} has loss {} and acc {}".format(i, loss.data.numpy(), acc))
            if i > 0 and i % 100 == 0:
                print("train epoch {} has loss {} and acc {}".format(i, loss.data.numpy(), acc))
                self.val()

    def predict(self, x):
        self.lr.eval()
        X = Variable(T.from_numpy(x)).float()
        output = self.lr(X)
        return output

    def val(self):
        # split by class
        X_test_true = self.X_test_std[self.Y_test == 1]
        Y_test_true = self.Y_test[self.Y_test == 1]
        X_test_false = self.X_test_std[self.Y_test == 0]
        Y_test_false = self.Y_test[self.Y_test == 0]
        pre_res_true = self.predict(X_test_true)
        _, pre_res_true = T.max(pre_res_true.data, 1)
        pre_res_false = self.predict(X_test_false)
        _, pre_res_false = T.max(pre_res_false.data, 1)
        res_true, res_false = pre_res_true.numpy() == Y_test_true, pre_res_false.numpy() == Y_test_false
        pre_res = self.predict(self.X_test_std)
        _, pre_res = T.max(pre_res.data, 1)
        pre_res = pre_res.numpy() == self.Y_test

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
            params = np.vstack((self.lr.coef_, self.sc.mean_, self.sc.std_))
            np.savetxt(f, params)

if __name__ == "__main__":
    pass