import sklearn
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

from models.supervised.bimt.model import BioMLP
from models.data.sklearn_datasets import MoonDataset, SpiralDataset, BlobsDataset, CirclesDataset
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='vanilla', help='Name of the experiment')

# Data related arguments
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--n_samples_test', type=int, default=200, help='Number of samples to generate for testing')
parser.add_argument('--noise', type=float, default=0.01, help='Noise level of the data')
parser.add_argument('--n_features', type=int, default=2, help='Number of features to generate')
parser.add_argument('--seed', type=int, default=2, help='Random seed')
parser.add_argument('--dataset', type=str, default='moon', help='Dataset to use')


# Model related arguments
parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of the model')
parser.add_argument('--width', type=int, default=20, help='Width of the hidden layers')
parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--output_dim', type=int, default=2, help='Output dimension of the model')

# Training related arguments
parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
parser.add_argument('--lamb', type=int, default=0.001, help='Regularisation parameter')
parser.add_argument('--swap_interval', type=int, default=200, help='Run swapping check every .. epochs')
parser.add_argument('--scheduler', type=str, default='custom', help='Use a scheduler for the learning rate')

# Misc arguments
parser.add_argument('--save_interval', type=int, default=1, help='Save model every .. epochs')
parser.add_argument('--log_interval', type=int, default=100, help='Log model stats every .. epochs')
parser.add_argument('--plot', type=bool, default=True, help='Plot the results')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
sklearn.utils.check_random_state(args.seed)
cur_dir = f"./models/supervised/bimt/saved_models/{args.name}/{args.dataset}/"

# Check if directory exists
if not os.path.exists(f'{cur_dir}'):
    os.makedirs(f'{cur_dir}')

# Generate the data
n_sample = args.n_samples
n_sample_test = args.n_samples_test

if args.dataset == 'moon':
    dataset = MoonDataset(n_samples=args.n_samples, noise=args.noise)
    dataset_test = MoonDataset(n_samples=args.n_samples, noise=args.noise)
elif args.dataset == 'spiral':
    dataset = SpiralDataset(n_samples=args.n_samples, length=args.length, noise=args.noise)
    dataset_test = SpiralDataset(n_samples=args.n_samples, length=args.length, noise=args.noise)
elif args.dataset == 'blobs':
    dataset = BlobsDataset(n_samples=args.n_samples, noise=args.noise, centers=args.clusters)
    dataset_test = BlobsDataset(n_samples=args.n_samples, noise=args.noise, centers=args.clusters)
elif args.dataset == 'circles':
    dataset = CirclesDataset(n_samples=args.n_samples, noise=args.noise)
    dataset_test = CirclesDataset(n_samples=args.n_samples, noise=args.noise)
else:
    raise ValueError("Dataset not supported.")

X, y = dataset.X, dataset.y
X = torch.tensor(X, dtype=torch.float, requires_grad=True)
y = torch.tensor(y, dtype=torch.long)

X_test, y_test = dataset_test.X, dataset_test.y
X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.long)

with open(f"{cur_dir}/dataset.pkl", 'wb') as f:
    pkl.dump(dataset, f)
# Generate the model

shp = [args.input_dim] + [args.width] * args.depth + [args.output_dim]

model = BioMLP(shp=shp)

# Define the training loop

EPOCHS = args.epochs
LAMBDA = args.lamb
SWAP_INTERVAL = args.swap_interval
SAVE_INTERVAL = args.save_interval
LOG_INTERVAL = args.log_interval

def scheduler(epoch, lr):
    if epoch < args.epochs // 4:
        return lr
    elif epoch < args.epochs // 2:
        return lr * 10
    else:
        return lr
    
CEL = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
if args.plot:
    res_acc = []
    res_loss = []

for epoch in range(EPOCHS):
    if args.scheduler == 'default':
        pass
    elif args.scheduler == 'custom':
        lr = scheduler(epoch, args.lr)
    else:
        raise ValueError('Scheduler not recognised')
    
    optimizer.zero_grad()
    y_pred = model(X)
    loss = CEL(y_pred, y)
    acc = torch.mean((torch.argmax(y_pred, dim=1) == y).float())



    reg = model.get_cc(weight_factor=1)
    total_loss = loss + LAMBDA*reg
    total_loss.backward()
    optimizer.step()

    if epoch % LOG_INTERVAL == 0:
        pred_test = model(X_test)
        loss_test = CEL(pred_test, y_test)
        acc_test = torch.mean((torch.argmax(pred_test, dim=1) == y_test).float())
        print("epoch = %d | total loss: %.2e | train loss: %.2e | test loss %.2e | train acc : %.2f | test acc : %.2f | reg: %.2e "%(epoch, total_loss.detach().numpy(), loss.detach().numpy(), loss_test.detach().numpy(), acc.detach().numpy(), acc_test.detach().numpy(), reg.detach().numpy()))
        if args.plot:
            res_acc.append(acc.detach().numpy())
    if args.plot:
            res_loss.append(total_loss.detach().numpy())
    if epoch % SAVE_INTERVAL == 0:
        torch.save(model, f'{cur_dir}/model_{epoch}.pt')
        
    
    if (epoch+1) % SWAP_INTERVAL == 0:
        model.relocate()

if args.plot:
    plt.plot(res_acc)
    plt.savefig(f'{cur_dir}/acc.png')
    plt.clf()
    plt.plot(res_loss)
    plt.savefig(f'{cur_dir}/loss.png')
