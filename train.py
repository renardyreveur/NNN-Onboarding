import matplotlib.pyplot as plt

from backprop import backprop
from dataloading import XORDataset
from loss import mse
from neural_network import MLPModel


def update(w_and_b, w_and_b_grads, lr=0.01):
    new_params = []
    for (param, param_grad) in zip(w_and_b, w_and_b_grads):
        new_params.append(param - lr * param_grad)

    return new_params


def train():
    # Model Configuration
    MLP_CONFIG = {
        "dims": [2, 2, 2, 1],
        "act_fn": "ReLU",
        "final_act_fn": "Sigmoid",
    }

    # Training Configuration
    LR = 0.05
    EPOCHS = 1000

    # Instantiate model with configuration
    model = MLPModel(MLP_CONFIG)
    model.config.update({"loss_fn": "mse"})

    loss_y = []
    # Training loop
    for epoch in range(EPOCHS):
        dataset = XORDataset(batch_size=256, num_batches_per_epoch=10)
        cum_loss = 0
        LR *= 0.9999
        for batch_idx, (data, label) in enumerate(dataset):
            output = model(data)  # Forward Pass

            loss = mse(output[-1], label)  # Loss Calculation
            cum_loss += loss

            gradients = backprop(  # Backpropagation
                outputs=output,
                weights=[x.weights for x in model.layers],
                label=label,
                config=model.config
            )

            new_params = update(  # Weight Update
                w_and_b=[x.weights for x in model.layers] + [x.biases for x in model.layers],
                w_and_b_grads=gradients,
                lr=LR
            )
            for i in range(len(new_params)):
                if i // (len(new_params)//2) < 1:
                    model.layers[i].weights = new_params[i]
                else:
                    model.layers[i % len(new_params)//2].biases = new_params[i]

            # Check training output
            if epoch % 100 == 99 and batch_idx % dataset.num_batches_per_epoch == dataset.num_batches_per_epoch - 1:
                print(f"Input:\n {data[0:4]}, \n\n Output:\n {output[-1][0:4]}, \n\n Target: \n {label[0:4]}")

        print(f"Epoch {epoch + 1}  --  Loss: {cum_loss}, lr: {LR}")
        loss_y.append(cum_loss)

    return list(range(EPOCHS)), loss_y


if __name__ == "__main__":
    x_ax, y_ax = train()

    # plot loss graph
    plt.plot(x_ax, y_ax)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
