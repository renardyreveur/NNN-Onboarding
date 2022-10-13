# Numpy-only Neural Network Implementation

## What is it?

This repo contains a basic implementation of a fully-connected, multi-layer, feed-forward perceptron model.

The only dependency is numpy for array operations and matplotlib to plot the training loss graph.

This is an attempt at learning how DL frameworks such as PyTorch/Tensorflow works, and it serves as a reference for building neural networks in an object-orientated way.


## What does it do?

You can define a simple multi-linear-layer model with a configuration dictionary, and it learns to imitate the XOR logic gate via stochastic gradient descent.

Currently, every layer has an activation of either ReLU or Sigmoid. 

Mean Squared Error is used as the loss function, though Binary Cross-Entropy would be more suitable for the task.


## How is it Structured?

The model is defined in `neural_network.py`, and the other files are representative of their titles as well. 

Inside `backprop.py` there is also a gradient checking demo to ensure the correctness of the backpropagation implementation.

`backprop_hist.py` displays the previous iterations in analytically implementing backpropagation, that led up to the final vectorized and generalized function in the main file.

`train.py` is where the magic happens, and would be your *main* point of execution.