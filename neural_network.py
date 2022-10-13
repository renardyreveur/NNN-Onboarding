import numpy as np

import activations
from base_module import Module


class Linear(Module):
    def __init__(self, in_nodes, out_nodes):
        super().__init__()
        self.weights = (np.random.rand(out_nodes, in_nodes) - 0.5) * 2 * np.sqrt(3/out_nodes)
        self.biases = (np.random.rand(out_nodes, 1) - 0.5) * 2 * np.sqrt(3/out_nodes)

    def forward(self, x):
        return self.weights @ x + self.biases


class MLPModel(Module):
    def __init__(self, config: dict):
        super().__init__()
        # Check configuration
        self.config = config
        self.check_config()

        # Create model attributes
        self.input_dim, self.output_dim = self.config['dims'][0], self.config['dims'][-1]

        # Stack Layers
        dim_pairs = [(self.config['dims'][i], self.config['dims'][i + 1]) for i in range(len(self.config['dims']) - 1)]
        self.layers = [Linear(*dims) for dims in dim_pairs]
        self.activations = [self.config['act_fn'][1]() if i < len(dim_pairs) - 1 else self.config['final_act_fn'][1]()
                            for i in range(len(dim_pairs))]

    def check_config(self):
        # Check model structure
        assert ["dims", "act_fn", "final_act_fn"] == list(self.config.keys()), \
            "Please provide a configuration dictionary with keys 'dims', 'intermediate_activations', " \
            "and 'final_activation' "
        assert len(self.config['dims']) > 1, "The MLP model should have at least an input layer and an output layer"
        assert all([x > 0 for x in self.config['dims']]), "All layers of the MLP should have at least one perceptron"

        # Parse Activation functions
        allowed_act_fns = [x for x in dir(activations) if x[:2] != "__" and x != "Module" and x != "np"]
        assert self.config['act_fn'] in allowed_act_fns, \
            f"The activations should be one of {allowed_act_fns}"
        assert self.config['final_act_fn'] in allowed_act_fns + ["None"], \
            f"The activations should be one of {allowed_act_fns}"

        self.config['act_fn'] = (self.config['act_fn'], getattr(activations, self.config['act_fn']))
        self.config['final_act_fn'] = (None, None) if self.config['final_act_fn'] == "None" \
            else (self.config['final_act_fn'], getattr(activations, self.config['final_act_fn']))

    def forward(self, x):
        intermediate_outputs = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            intermediate_outputs.append(x)

            x = self.activations[i](x)
            intermediate_outputs.append(x)

        return intermediate_outputs
