import numpy as np

from backprop import relu_diff, sigmoid_diff


def backprop_first(inter_outputs, weights, label, inputs):
    l, la, m, ma, o, oa = inter_outputs
    l3w, l2w, l1w = weights

    dw10 = -2 * (label - oa) * oa * (1 - oa) * ma[1]
    dw9 = -2 * (label - oa) * oa * (1 - oa) * ma[0]

    dw8 = -2 * (label - oa) * oa * (1 - oa) * la[1] * relu_diff(m[1]) * l3w[1][0]
    dw7 = -2 * (label - oa) * oa * (1 - oa) * la[1] * relu_diff(m[0]) * l3w[0][0]
    dw6 = -2 * (label - oa) * oa * (1 - oa) * la[0] * relu_diff(m[1]) * l3w[1][0]
    dw5 = -2 * (label - oa) * oa * (1 - oa) * la[0] * relu_diff(m[0]) * l3w[0][0]

    dw4 = (-2 * (label - oa) * oa * (1 - oa) * inputs[1]) * \
          (relu_diff(m[0]) * l3w[0][0] * relu_diff(l[1]) * l2w[0][1] + relu_diff(m[1]) * l3w[1][0] * relu_diff(l[1]) *
           l2w[1][1])
    dw3 = (-2 * (label - oa) * oa * (1 - oa) * inputs[1]) * \
          (relu_diff(m[0]) * l3w[0][0] * relu_diff(l[0]) * l2w[0][0] + relu_diff(m[1]) * l3w[1][0] * relu_diff(l[0]) *
           l2w[1][0])
    dw2 = (-2 * (label - oa) * oa * (1 - oa) * inputs[0]) * \
          (relu_diff(m[0]) * l3w[0][0] * relu_diff(l[1]) * l2w[0][1] + relu_diff(m[1]) * l3w[1][0] * relu_diff(l[1]) *
           l2w[1][1])
    dw1 = (-2 * (label - oa) * oa * (1 - oa) * inputs[0]) * \
          (relu_diff(m[0]) * l3w[0][0] * relu_diff(l[0]) * l2w[0][0] + relu_diff(m[1]) * l3w[1][0] * relu_diff(l[0]) *
           l2w[1][0])

    grad_l1 = np.array([[dw1.squeeze(), dw3.squeeze()],
                        [dw2.squeeze(), dw4.squeeze()]])
    grad_l2 = np.array([[dw5.squeeze(), dw7.squeeze()],
                        [dw6.squeeze(), dw8.squeeze()]])
    grad_l3 = np.array([[dw9.squeeze()],
                        [dw10.squeeze()]])

    return grad_l1, grad_l2, grad_l3


def backprop_second(intermediate_outputs, weights, label, inputs):
    l, la, m, ma, o, oa = intermediate_outputs
    l1w, l2w, l3w = weights

    n = oa.shape[0]
    dcdo = 2 * (oa - label) / n * sigmoid_diff(oa)  # dZ3
    dcw3 = dcdo @ ma.T  # dW3

    dodm = l3w * relu_diff(m.T)  # dZ2
    dcw2 = (dodm.T @ dcdo) @ la.T

    dmdl = l2w * relu_diff(l.T)
    dcw1 = ((dmdl.T @ dodm.T) @ dcdo) @ inputs.T

    return dcw1, dcw2, dcw3
