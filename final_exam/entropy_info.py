import numpy as np


def entropy_info(prob_xy_matrix, x, y):
    if np.sum(prob_xy_matrix) != 1:
        print("error input")
        return

    print(prob_xy_matrix)
    x_size = prob_xy_matrix.shape[0]
    y_size = prob_xy_matrix.shape[0]

    mask = prob_xy_matrix == 0
    norm_prob = prob_xy_matrix.copy()
    norm_prob[mask] = 1e-5
    H_x_y = -np.log2(norm_prob) * prob_xy_matrix
    print("H_xy: ", np.sum(H_x_y))

    H_Y_X = 0.
    for i in range(x_size):
        print("H(Y|X=%d)=" % (x[i]), end="")
        prob_at_x = np.sum(prob_xy_matrix[i, :])
        cur_prob = prob_xy_matrix[i, :] / prob_at_x
        mask_zero = cur_prob == 0
        cur_prob_norm = cur_prob.copy()
        cur_prob_norm[mask_zero] = 1e-5
        out = np.sum(-np.log2(cur_prob_norm) * cur_prob)
        for j in range(y_size):
            print("%f * log(%f)"%(cur_prob[j], cur_prob[j]), end="")
            if j != y_size - 1:
                print("+", end="")
            else:
                print("=", end="")
        H_Y_X += out * prob_at_x
        print(out)
    print("H(Y|X)=%lf"%H_Y_X)

    H_X_Y = 0.
    for i in range(y_size):
        print("H(X|Y=%d)=" % (y[i]), end="")
        prob_at_y = np.sum(prob_xy_matrix[:, i])
        # print(prob_xy_matrix[:, i], prob_at_y)
        cur_prob = prob_xy_matrix[:, i] / prob_at_y
        mask_zero = cur_prob == 0
        cur_prob_norm = cur_prob.copy()
        cur_prob_norm[mask_zero] = 1e-5
        out = np.sum(-np.log2(cur_prob_norm) * cur_prob)
        H_X_Y += out * prob_at_y
        for j in range(x_size):
            print("%f * log(%f)"%(cur_prob[j], cur_prob[j]), end="")
            if j != x_size - 1:
                print("+", end="")
            else:
                print("=", end="")
        print(out)
    print("H(X|Y)=%lf" % H_X_Y)


if __name__ == "__main__":
    xy = np.array([[0., 1/8., 1/8.], [1/2., 0., 1/8.], [1/8., 0., 0.]])
    entropy_info(xy, x=[1, 2, 3], y=[1, 2, 3])