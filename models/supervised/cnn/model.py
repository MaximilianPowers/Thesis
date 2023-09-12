import torch.nn as nn
import math

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_shape):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.act_func = nn.SiLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        H_in, W_in = self.input_shape
        
        H_out = math.floor((H_in - kernel_size) / stride) + 1
        W_out = math.floor((W_in - kernel_size) / stride) + 1

        self.in_features = H_in * W_in * self.in_channels
        self.out_features = H_out * W_out * self.out_channels



    def forward(self, x):
        return self.act_func(self.conv(x))
    
class FlatLayer(nn.Module):
    def __init__(self, dim):
        super(FlatLayer, self).__init__()
        self.in_features = dim
        self.out_features = dim

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class FCLayer(nn.Module):
    def __init__(self, in_features, out_features, act_func=nn.ReLU()):
        super(FCLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_func = act_func if act_func is not None else lambda x: x
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return self.act_func(self.linear(x))
    
class CNN(nn.Module):
    def __init__(self, cnn_layers, fc_layers, output_dim):
        super(CNN, self).__init__()
        assert fc_layers[-1][1] == output_dim
        self.cnn_layers = nn.ModuleList([CNNLayer(*params) for params in cnn_layers])
        self.flat_layers = nn.ModuleList([FlatLayer(self.cnn_layers[-1].out_features)])
        self.fc_layers = nn.ModuleList([FCLayer(*params) for params in fc_layers])
        self.output_dim = output_dim
        self.layers = self.cnn_layers + self.flat_layers + self.fc_layers
        self.num_layers = len(self.layers)
        
    def init_forward(self):
        self.activations = []
        
    def get_activations(self):
        return self.activations
    
    def save_forward(self, x):
        self.activations.append(x)
        
    def forward(self, x, save_activations=False):
        self.init_forward()
        for layer in self.layers:
            if save_activations: self.save_forward(x)
            x = layer(x)
        
        
        if save_activations: self.save_forward(x)
        output = nn.LogSoftmax(dim=1)(x)
        return output
    
    def forward_layers(self, x, indx):
        assert indx < self.num_layers
            
        for layer in self.layers[indx:]:
            x = layer(x)
        
        return x

