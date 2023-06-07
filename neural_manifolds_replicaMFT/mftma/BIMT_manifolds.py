from manifold_analysis_correlation import manifold_analysis_corr
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from collections import defaultdict
from BIMT_2D import BioMLP2D
from utils.activation_extractor import extractor
from utils.make_manifold_data import make_manifold_data
import torch
import numpy as np
import json

np.random.seed(0)

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
sampled_per_class = 10
examples_per_class = 50

d_in = 784
d_out = 10

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    data_full = make_manifold_data(
        train_dataset, sampled_per_class, examples_per_class, seed=0)

json_file = json.load(
    open('neural_manifolds_replicaMFT/mftma/results/l1.json', 'r'))

for indx, model_name in enumerate(os.listdir('/Users/maxpowers/Library/Mobile Documents/com~apple~CloudDocs/MSc Thesis/Thesis.nosync/neural_manifolds_replicaMFT/mftma/models/l1_cnn')):
    if (w := model_name.split('-')[0]) in json_file.keys():
        continue
    print(f'Currently analysing model after {model_name} steps:')
    model_num = int(w)

    model = BioMLP2D(shp=[784, 100, 100, 10])
    model.load_state_dict(torch.load(
        '/Users/maxpowers/Library/Mobile Documents/com~apple~CloudDocs/MSc Thesis/Thesis.nosync/neural_manifolds_replicaMFT/mftma/models/l1_cnn/{}'.format(model_name), map_location=torch.device('cpu')))
    activations = extractor(model, data_full, layer_nums=[1, 2, 3])
    for layer, data, in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X
    radii, dimensions, capacities, corr, Ks = [], [], [], [], []

    for indx, (k, X) in enumerate(activations.items()):
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)
        a = 1/np.mean(1/a)
        r = np.mean(r)
        d = np.mean(d)
        radii.append(r)
        dimensions.append(d)
        capacities.append(a)
        corr.append(r0)
        Ks.append(K)

    print('Radius:')
    print(radii)
    print('Dimension:')
    print(dimensions)
    print('Capacity:')
    print(capacities)
    print('Correlation:')
    print(corr)
    print('K:')
    print(Ks)

    tmp_dict = {}
    tmp_dict['radius'] = radii
    tmp_dict['dimension'] = dimensions
    tmp_dict['capacity'] = capacities
    tmp_dict['correlation'] = corr

    json_file[model_num] = tmp_dict

    json.dump(json_file, open(
        'neural_manifolds_replicaMFT/mftma/results/l1.json', 'w'))
