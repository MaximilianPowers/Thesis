import torch
from collections import defaultdict
from torch import nn


class KFAC():
    def __init__(self, model):
        self.model = model
        self.fisher = defaultdict(lambda: (torch.zeros(1), torch.zeros(1)))
        self.layer_names = {module: name for name,
                            module in model.named_modules()}

    def save_data(self, module, input, output):
        if isinstance(module, nn.Linear):
            self.fisher[self.layer_names[module]] = (
                input[0].data, output.data)

    def compute_fisher(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                x, y = self.fisher[self.layer_names[module]]
                # Since we've saved the data from the forward pass, we don't need gradients
                with torch.no_grad():
                    x = x - x.mean(0)
                    y = y - y.mean(0)
                    self.fisher[self.layer_names[module]] = x.t() @ x / \
                        x.size(0), y.t() @ y / y.size(0)

    def register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.save_data)
