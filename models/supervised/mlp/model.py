import torch.nn as nn
from torch.autograd.functional import jacobian

class Layer(nn.Module):
    def __init__(self, input_dim, output_dim, act_func=nn.Tanh()):
        super(Layer, self).__init__()
        if act_func is None:
            self.act_func = lambda x: x
        else:
            self.act_func = act_func
        self.linear_map = nn.Linear(input_dim, output_dim)
        self.out_features = output_dim
        self.in_features = input_dim

    def forward(self, x):
        return self.act_func(self.linear_map(x))


class MLP(nn.Module):
    def __init__(self, input_dim, num_layers, layer_width, output_dim=1):
        super(MLP, self).__init__()
        if isinstance(layer_width, int):
            layer_width = [layer_width for _ in range(num_layers-1)]
        assert len(layer_width) == num_layers-1
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [Layer(input_dim, layer_width[0])]
            + [Layer(layer_width[i], layer_width[i+1])
               for i in range(self.num_layers-2)]
            + [Layer(layer_width[-1], output_dim, act_func=nn.Sigmoid())])
    def init_forward(self):
        self.activations = []
        self.jacobians = []

    def get_activations(self):
        return self.activations
    
    def save_forward(self, coordinates):
        self.activations.append(coordinates)

    def save_jacobian(self, coordinates, func):
        def J(x): return jacobian(func, coordinates)

        self.jacobians.append(J(coordinates))

    def forward(self, x, save_activations=False, save_jacobians=False):
        self.init_forward()
        for layer in self.layers:
            if save_activations:
                self.save_forward(x)
            if save_jacobians:
                self.save_jacobians(x, layer)
            x = layer.forward(x)
        if save_activations:
            self.save_forward(x)
        if save_jacobians:
            self.save_jacobians(x, layer)

        return x

    def forward_layers(self, x, indx):
        # Go forward layers starting at indx, used for pullback metric mapping
        assert indx < self.num_layers
        for layer in self.layers[indx:]:
            x = layer.forward(x)
        return x
