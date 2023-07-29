import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian

from torch import nn
import torch

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


class VanillaNN(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_width, output_dim=1):
        super(VanillaNN, self).__init__()
        if isinstance(hidden_layer_width, list):
            if len(hidden_layer_width) == num_hidden_layers:
                variable_width = True
            else:
                raise ValueError("The number of hidden layer widths must be equal to the number of hidden layers.")

        elif isinstance(hidden_layer_width, int):
            variable_width = False
        else:
            return "The input is neither a list nor an integer."
        self.num_layers = num_hidden_layers+1
        if variable_width:
            self.layers = nn.ModuleList(
                [Layer(input_dim, hidden_layer_width[0])]
                + [Layer(hidden_layer_width[i], hidden_layer_width[i+1])
                   for i in range(self.num_layers-2)]
                + [Layer(hidden_layer_width[-1], output_dim, act_func=nn.Sigmoid())])
        else:
            self.layers = nn.ModuleList(
                [Layer(input_dim, hidden_layer_width)]
                + [Layer(hidden_layer_width, hidden_layer_width)
                   for _ in range(self.num_layers-2)]
                + [Layer(hidden_layer_width, output_dim, act_func=nn.Sigmoid())])

    def init_forward(self):
        self.activations = []

    def save_forward(self, coordinates):
        self.activations.append(coordinates)

    def forward(self, x, save_activations=False):
        if save_activations:
            self.init_forward()
            self.save_forward(x) 
        for layer in self.layers:
            x = layer(x)
            if save_activations:
                self.save_forward(x)

        return x


class EuclideanNN(nn.Module):
    def __init__(self, input_dim, num_layers, layer_width, output_dim=1):
        super(EuclideanNN, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [Layer(input_dim, layer_width, act_func=None)]
            + [Layer(layer_width, layer_width, act_func=None)
               for _ in range(self.num_layers-2)]
            + [Layer(layer_width, output_dim, act_func=None)])

    def init_forward(self):
        self.activations = []
        self.jacobians = []
        self.hessians = []

    def save_forward(self, coordinates, coord_change_func):
        new_coords = coord_change_func(coordinates)
        def J(x): return jacobian(coord_change_func, x)
        self.jacobians.append(J(coordinates))
        # Compute the second derivatives
        H = jacobian(J, coordinates)
        self.hessians.append(H)
        self.activations.append(new_coords)

    def forward(self, x, save_activations=False):
        self.init_forward()
        for layer in self.layers:
            a = layer(x)
            if save_activations:
                self.save_forward(x, layer)
            x = a
        return x
