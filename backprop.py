import sys
from collections import deque
from functools import reduce

import numpy as np

import loss
from activations import Sigmoid


def relu_diff(x: np.array):
    a = np.zeros_like(x, dtype=np.int64)
    a[x > 0] = 1
    return a


def sigmoid_diff(x: np.array):
    sigmoid = Sigmoid()
    return sigmoid(x) * (1 - sigmoid(x))


def mse_diff(output: np.array, target: np.array):
    n = output.shape[0]
    return 2 * (output - target) / n


def backprop(outputs, weights, label, config):
    this = sys.modules[__name__]
    # Reverse weights order as we're traversing backwards
    weights = weights[::-1]

    # Calculate derivative of loss to activation(output) - common for all subsequent calc.
    dloss = getattr(this, config['loss_fn'].lower() + "_diff")(outputs[-1], label)
    dcdo = dloss * getattr(this, config['final_act_fn'][0].lower() + "_diff")(outputs[-2])

    # Gradient Cache of intermediate calculations (the reason for backprop)
    cache = deque([dcdo, np.ones(1)[None, :]])

    # Analytical Gradients for act(linear) * n type MLPs
    weight_grads, bias_grads = [], []
    for i, o in enumerate(outputs[:-2][::-1]):
        if i % 2 == 0:
            d_upto_now = reduce(lambda a, b: a @ b, cache)
            weight_grads.append(d_upto_now @ o.swapaxes(-1, -2))
            bias_grads.append(np.sum(d_upto_now, axis=-1, keepdims=True))
        else:
            # (w * relu_diff(o.T)).T
            cache.appendleft(
                (
                    weights[i % len(outputs) // 2] *
                    getattr(this, config['act_fn'][0].lower() + "_diff")(o.swapaxes(-1, -2))
                ).swapaxes(-1, -2)
            )

    # Return in original order
    return [np.sum(x, axis=0) for x in weight_grads[::-1] + bias_grads[::-1]]


def gradient_checking(model, dummy, dummy_label, eps=1e-6):
    weights = [x.weights for x in model.layers]
    biases = [x.biases for x in model.layers]
    config = model.config
    loss_fn = getattr(loss, config["loss_fn"])

    # Get implemented backpropagation gradient values
    output = model(dummy)
    gradients = backprop(output, weights, dummy_label, config)

    # See how the output reacts to small changes in each weight
    for i, param_group in enumerate(weights + biases):
        row, col = param_group.shape
        param_group_copy = param_group.copy()
        params = param_group_copy.reshape(-1).tolist()
        approx_grad = np.zeros_like(param_group)

        for j, p in enumerate(params):
            param_group[j // col if col > 1 else 0][j % col] = p - eps
            p_sub_out = model(dummy)
            p_sub_loss = loss_fn(p_sub_out[-1], dummy_label)

            param_group[j // col if col > 1 else 0][j % col] = p + eps
            p_add_out = model(dummy)
            p_add_loss = loss_fn(p_add_out[-1], dummy_label)

            # Calculate approximate do/dw for each weights
            approx_grad[j // col if col > 1 else 0][j % col] = (p_add_loss - p_sub_loss) / (2 * eps)

        name = "BIASES" if i // len(weights) > 0 else "WEIGHTS"
        print(f"Layer {(i % len(weights)) + 1} {name}:\n APPROX:\n {approx_grad} \n BACKPROP:\n {gradients[i]}")

        # Calculate difference
        diff_norm = np.linalg.norm(approx_grad - gradients[i])
        norm_sum = np.linalg.norm(approx_grad) + np.linalg.norm(gradients[i])
        difference = diff_norm / norm_sum
        if difference > 1e-5:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = "
                  + str(difference) + "\033[0m \n")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = "
                  + str(difference) + "\033[0m\n")


if __name__ == "__main__":
    from neural_network import MLPModel

    # Gradient Checking Example
    mlp_config = {
        "dims": [2, 2, 2, 1],
        "act_fn": "ReLU",
        "final_act_fn": "Sigmoid"
    }

    # Define input and label (N, IN, OUT)
    sample = np.random.randn(3, 2, 1)
    sample_label = np.random.rand(3, 1, 1)

    # Create model
    test_model = MLPModel(mlp_config)
    test_model.config.update({"loss_fn": "mse"})

    # Check backpropagation gradients against approximation
    gradient_checking(test_model, sample, sample_label)
