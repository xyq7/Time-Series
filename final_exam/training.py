import numpy as np
import random
random.seed(0)
np.random.seed(0)


def sigmoid(x):
    return 1./(1 + np.exp(-x))


def training(data, label, lr=0.01, max_iter=1000, weight=None, bias=None, act=None):
    if weight is None:
        weight = np.random.uniform(-1, 1, data.shape[1])
    if bias is None:
        bias = np.random.uniform(-1, 1)
    for j in range(max_iter):
        correct_num = 0
        for i in range(data.shape[0]):
            value = np.sum(weight * data[i, :]) + bias
            if act is not None:
                value = act(value)

            if value >= 0 and label[i] == 1:
                correct_num += 1
                # continue
            if value < 0 and label[i] == 0:
                correct_num += 1
                # continue
            predict = value
            weight = weight + lr * (label[i] - predict) * data[i, :]
            print(weight)
            bias = bias + lr * (label[i] - predict)
            print(bias)
            # exit(0)
    correct_num = 0
    for i in range(data.shape[0]):
        value = np.sum(weight * data[i, :]) + bias
        if value >= 0 and label[i] == 1:
            correct_num += 1
            continue
        if value < 0 and label[i] == 0:
            correct_num += 1
            continue
    print("acc is ", correct_num / float(data.shape[0]))
    return weight, bias



if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 0, 0, 1, 0])
    print(training(x, y,
                   max_iter=1, lr=0.5, weight=np.array([0.1, 0.1]), bias=np.array([0.1]),
                   act=sigmoid))