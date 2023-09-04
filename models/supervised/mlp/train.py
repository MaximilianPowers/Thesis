from math import pi
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.data.sklearn_datasets import MoonDataset, SpiralDataset, BlobsDataset, CirclesDataset
import matplotlib.pyplot as plt
from models.supervised.mlp.model import MLP
import os
import glob
from utils.metrics.metrics import class_accuracy
import torch.nn.functional as F
# Import scripts to set up seeds for sklearn torch and numpy
from numpy.random import seed
import pickle

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for inputs, targets in loader:
        if model.output_dim > 1:
           targets = torch.tensor(F.one_hot(targets.clone().detach().to(
               torch.int64), num_classes=model.output_dim).reshape(-1, model.output_dim), dtype=torch.float32)
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = criterion(torch.squeeze(outputs), targets)
        running_loss += loss.item() * inputs.size(0)

        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def main(args):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    seed(args.seed)
    
    if args.dataset == 'moon':
        dataset = MoonDataset(n_samples=args.n_samples, noise=args.noise)
    elif args.dataset == 'spiral':
        dataset = SpiralDataset(n_samples=args.n_samples, length=args.length, noise=args.noise)
    elif args.dataset == 'blobs':
        dataset = BlobsDataset(n_samples=args.n_samples, noise=args.noise, centers=args.clusters)
    elif args.dataset == 'circles':
        dataset = CirclesDataset(n_samples=args.n_samples, noise=args.noise)
    else:
        raise ValueError("Dataset not supported.")
    cur_dir = f"./models/supervised/mlp/saved_models/{args.name}"

    # Check if directory exists
    if not os.path.exists(f'{cur_dir}/mlp_{args.dataset}/'):
        os.makedirs(f'{cur_dir}/mlp_{args.dataset}/')
    # Remove all files in directory
    files = glob.glob(f'{cur_dir}/mlp_{args.dataset}/*')
    for f in files:
        os.remove(f)
    with open(f'{cur_dir}/mlp_{args.dataset}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLP(input_dim=args.input_dim, output_dim=args.output_dim, num_layers=args.num_layers,
                      layer_width=args.layer_width).to(device)
    if args.output_dim > 2:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate)
    if args.plot:
        res_acc = []
        res_loss = []
    for epoch in range(args.num_epochs):
        torch.save(
            model.state_dict(), f'{cur_dir}/mlp_{args.dataset}/model_{epoch}.pth')
        train_loss = train(model, loader, criterion, optimizer, device)
        if args.plot:
            res_loss.append(train_loss)
        if epoch % (args.num_epochs//20) == 0:
            acc = class_accuracy(model, loader, device)
            print(
                f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss:.4f}, Accuracy: {acc:.4f}")
            if args.plot:
                res_acc.append(acc)
    if args.plot:
        plt.plot(res_acc)
        plt.savefig(f'{cur_dir}/mlp_{args.dataset}/acc.png')
        plt.clf()
        plt.plot(res_loss)
        plt.savefig(f'{cur_dir}/mlp_{args.dataset}/loss.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_layers', type=int, default=7)
    parser.add_argument('--layer_width', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='blobs')
    parser.add_argument('--noise', type=float, default=0.4)
    parser.add_argument('--clusters', type=float, default=4)
    parser.add_argument('--length', type=float, default=2*pi)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--name', type=str, default="2_wide")
    args = parser.parse_args()
    main(args)
