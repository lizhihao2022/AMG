import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=1, act='gelu', batch_norm=False, **kwargs):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = [nn.Linear(input_size, hidden_size)]
        activation = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(),
                      'sigmoid': nn.Sigmoid(), 'leaky_relu': nn.LeakyReLU(),
                      'elu': nn.ELU(), 'softplus': nn.Softplus()}.get(act, None)
        if activation is None:
            raise NotImplementedError("Activation function not supported.")

        for _ in range(num_layers):
            layers.append(activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Linear(hidden_size, hidden_size))

        layers.append(activation)
        layers.append(nn.Linear(hidden_size, output_size))
        self.processor = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.processor(x)
