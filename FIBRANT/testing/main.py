import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
import os
import pandas as pd
from model import BioMLP
from torch import nn


def gen_data():
    # HYPERPARAMETERS FOR DATASET
    n_samples_train = 400
    n_samples_test = 200
    noise = 0.1
    val_size = 0.1  # Validation set used to perform Fisher Information Based pruning
    # GENERATE DATASET
    X, y = make_moons(n_samples=n_samples_train, noise=noise)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_val, y_train, y_val = train_test_split(
        X.numpy(), y.numpy(), test_size=val_size)

    # Converting numpy arrays back to tensors
    X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float, requires_grad=True)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test, y_test = make_moons(n_samples=n_samples_test, noise=noise)

    X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    # HYPERPARAMETERS FOR TRAINING
    steps = 10000
    lr = 0.002
    swap_log = 200  # Swap every swap_log steps
    begin_swapping = 0  # Begin swapping after begin_swapping steps
    prune_perc = 0.1  # Percentage to be pruned every prune_log epochs
    prune_max = 0.1  # Minimum percentage to be pruned per hidden layer
    cc_ratio = 1 - prune_perc
    initial_cc = None
    curr_cc = 0

    # HYPERPARAMETERS FOR MODEL
    shp = [2, 20, 20, 2]  # Shape of the model
    topk = 6  # Number of candidate nodes to relocate per layer
    ystar = 0.2  # Distance between two nearby layers
    weight_factor = 2  # Weight factor for the bio-inspired trick
    lamb = 0.001  # Regularization parameter

    # INITIALIZE MODEL
    model = BioMLP(in_dim=shp[0], out_dim=shp[-1], w=max(shp), depth=len(shp)-1, ystar=ystar,
                   weight_factor=weight_factor, topk=topk, shp=shp, prune_perc=prune_perc, max_prune=prune_max)

    initial_cc = model.get_cc()
    # INITIALIZE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    for step in range(steps):
        # small lambda first, then large lambda, then small lambda
        if step == 2000:
            lamb = 0.01

        if step == 5000:
            lamb = 0.1
        CEL = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = CEL(y_pred, y_train)

        reg = model.get_cc()
        total_loss = loss + lamb*reg
        total_loss.backward()

        for layer, mask, bias_mask in zip(model.linears, model.masks, model.bias_masks):
            layer.linear.weight.grad *= mask
            layer.linear.bias.grad *= bias_mask
        optimizer.step()

        if ((step+1) % swap_log == 0) and (step+1) > begin_swapping:
            # pass
            model.relocate()
            # swap_record.append([copy.deepcopy(model.swapped), copy.deepcopy(
            #    model.swappee), copy.deepcopy(model.swap_self)])

        curr_cc = model.get_cc()
        if curr_cc/initial_cc < cc_ratio:
            model.architectural_prune()
            initial_cc = model.get_cc()


if __name__ == "__main__":
    t = []
    for i in range(10):
        X_train, y_train, X_val, y_val, X_test, y_test = gen_data()
        start = time.process_time()

        train_model("bimt_test_plain", X_train,
                    y_train, X_val, y_val, X_test, y_test)
        end = time.process_time()
        t.append(end-start)

    print("Average time: ", np.mean(t))
    print("Standard deviation: ", np.std(t))
