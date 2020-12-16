import numpy as np
# note(yms) try to fix random seed when use random weights
import random
random.seed(0)
np.random.seed(0)


def sigmoid(x):
    return 1./(1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def simple_rnn(x, y, w=None, b=None):
    t = x.shape[0]
    s = 0.
    if w is None:
        w = np.random.uniform(-1, 1, x.shape[1] + 1)
    if b is None:
        b = np.random.uniform(-1, 1)
    for i in range(t):
        x_i = x[i]
        x_i = np.append(x_i, s)
        s_i = tanh(np.sum(x_i * w) + b)
        error = s_i - y[i]
        print("error", error)
        s = s_i


def lstm(x, y, w_f=None, b_f=None,
         w_i=None, b_i=None,
         w_a=None, b_a=None, w_o=None,
         b_o=None):
    t = x.shape[0]
    s = 0.
    y_t = 0
    for i in range(t):
        x_i = x[i]
        x_i = np.append(x_i, y_t)
        f_i = sigmoid(np.sum(x_i * w_f) + b_f)
        i_i = sigmoid(np.sum(x_i * w_i) + b_i)
        a_i = tanh(np.sum(x_i * w_a) + b_a)
        s_i = f_i * s + i_i * a_i
        o_i = sigmoid(np.sum(x_i * w_o) + b_o)
        y_t = o_i * tanh(s_i)
        error = y_t - y[i]
        print("error", error)
        s = s_i


def gru(x, y, w_r=None, b_r=None,
        w_a=None, b_a=None, w_u=None,
        b_u=None):
    t = x.shape[0]
    y_t = 0
    for i in range(t):
        x_i = x[i]
        x_i = np.append(x_i, y_t)
        r_i = sigmoid(np.sum(x_i * w_r) + b_r)
        a_input = np.append(x[i], y_t * r_i)
        a_i = tanh(np.sum(a_input * w_a) + b_a)
        u_i = sigmoid(np.sum(x_i * w_u) + b_u)
        y_t = (1 - u_i) * y_t + u_i * a_i
        error = y_t - y[i]
        print("error", error)


if __name__ == "__main__":
    # w = np.array([0.7, 0.3, 0.4])
    # b = 0.4
    # x = np.array([[0.1, 0.4], [0.7, 0.9]])
    # y = np.array([0.3, 0.5])
    # simple_rnn(x, y, w, b)

    # x = np.array([[0.1, 0.4], [0.7, 0.9]])
    # y = np.array([0.3, 0.5])
    # w_f = np.array([0.7, 0.3, 0.4])
    # b_f = 0.4
    #
    # w_i = np.array([0.2, 0.3, 0.4])
    # b_i = 0.2
    #
    # w_a = np.array([0.4, 0.2, 0.1])
    # b_a = 0.5
    #
    # w_o = np.array([0.8, 0.9, 0.2])
    # b_o = 0.3
    # lstm(x, y, w_f, b_f, w_i, b_i,
    #      w_a, b_a, w_o, b_o)

    x = np.array([[0.1, 0.4], [0.7, 0.9]])
    y = np.array([0.3, 0.5])
    w_r = np.array([0.7, 0.3, 0.4])
    b_r = 0.4

    w_a = np.array([0.2, 0.3, 0.4])
    b_a = 0.3

    w_u = np.array([0.4, 0.2, 0.1])
    b_u = 0.5

    gru(x, y, w_r, b_r, w_a, b_a, w_u, b_u)