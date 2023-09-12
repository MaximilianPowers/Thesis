import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Define the Encoder module
class Encoder(nn.Module):
    def __init__(self, in_features, features, out_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList([
                Layer(in_features, features[0])
        ] + [
            Layer(features[i], features[i + 1]) for i in range(len(features) - 2)
        ] + [Layer(features[-2], features[-1], act_func=nn.Sigmoid())])
        self.fc_mu = Layer(features[-1], out_features)
        self.fc_log_var = Layer(features[-1], out_features)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


        
# Define the Decoder module
class Decoder(nn.Module):
    def __init__(self, in_features, features, out_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.layers = nn.ModuleList([
            Layer(in_features, features[0])
        ] + [
            Layer(features[i], features[i + 1]) for i in range(len(features) - 1)
        ] + [
            Layer(features[-1], out_features)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.layers = encoder.layers + [encoder.fc_mu]
        self.num_layers = len(self.layers)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def init_forward(self):
        self.activations = []

    def get_activations(self):
        return self.activations
    
    def forward(self, x, save_activations=False):
        if save_activations:
            self.init_forward()

            for layer in self.layers:
                self.activations.append(x)
                x = layer.forward(x)
            self.activations.append(x)
            return x
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
    
    def forward_layers(self, x, indx):
        assert indx < len(self.layers)
        for layer in self.layers[indx:]:
            x = layer.forward(x)
        return x