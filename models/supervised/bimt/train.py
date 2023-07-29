from sklearn.datasets import make_moons
import sklearn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np
import torch
from model import BioMLP
import torch.nn as nn
import os
import argparse
import json
import csv

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
sklearn.utils.check_random_state(seed)
save_model_indx = len(os.listdir('./saved_models'))
os.mkdir(f"./saved_models/{save_model_indx}")
os.mkdir(f"./figures/{save_model_indx}")

log_ref_file = 'log_ref.json'
if os.path.exists(log_ref_file):
    # If it does, load the existing data
    with open(log_ref_file, "r") as file:
        data = json.load(file)
else:
    # If it doesn't, create an empty dictionary
    data = {}
parser = argparse.ArgumentParser()

# Data related arguments
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--n_samples_test', type=int, default=1000, help='Number of samples to generate for testing')
parser.add_argument('--noise', type=float, default=0.05, help='Noise level of the data')
parser.add_argument('--n_features', type=int, default=2, help='Number of features to generate')

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
parser.add_argument('--log_interval', type=int, default=100, help='Run test every .. epochs')
parser.add_argument('--save_interval', type=int, default=100, help='Save model every .. epochs')
parser.add_argument('--fig_interval', type=int, default=50, help='Output figures every .. epochs')


args = parser.parse_args()

# Generate the data
n_sample = args.n_samples
n_sample_test = args.n_samples_test

X, y = make_moons(n_samples=n_sample, noise=args.noise)
X = torch.tensor(X, dtype=torch.float, requires_grad=True)
y = torch.tensor(y, dtype=torch.long)

X_test, y_test = make_moons(n_samples=n_sample_test, noise=args.noise)
X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.long)
# Generate the model

shp = [args.input_dim] + [args.width] * args.depth + [args.output_dim]

model = BioMLP(shp=shp)

# Define the training loop

EPOCHS = args.epochs
LAMBDA = args.lamb
SWAP_INTERVAL = args.swap_interval
LOG_INTERVAL = args.log_interval
SAVE_INTERVAL = args.save_interval
FIG_INTERVAL = args.fig_interval

def scheduler(epoch, lr):
    if epoch < args.epochs // 4:
        return lr
    elif epoch < args.epochs // 2:
        return lr * 10
    else:
        return lr
    
CEL = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

data[save_model_indx] = {
    'n_samples': args.n_samples,
    'n_samples_test': args.n_samples_test,
    'noise': args.noise,
    'n_features': args.n_features,
    'input_dim': args.input_dim,
    'width': args.width,
    'depth': args.depth,
    'output_dim': args.output_dim,
    'lr': args.lr,
    'epochs': args.epochs,
    'lamb': args.lamb,
    'swap_interval': args.swap_interval,
    'scheduler': args.scheduler,
    'log_interval': args.log_interval,
    'save_interval': args.save_interval,
    'fig_interval': args.fig_interval
}

with open(log_ref_file, "w") as file:
        json.dump(data, file, indent=4)

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
        log_store = [epoch, total_loss.detach().numpy(), loss.detach().numpy(), loss_test.detach().numpy(), acc.detach().numpy(), acc_test.detach().numpy(), reg.detach().numpy()]
        with open(f'./logs/{save_model_indx}.csv', 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(log_store)
         
    if epoch % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f'./saved_models/{save_model_indx}/model_{epoch}.pt')
        
    if epoch % FIG_INTERVAL == 0:
        plt.figure(figsize=(3, 7)) 

        plt.subplot(2,1,1)

        N = 2
        s = 1/(2*max(shp))
        for j in range(len(shp)):
            N = shp[j]
            for i in range(N):
                circle = Ellipse((1/(2*N)+i/N, 0.1*j), s, s/10*((len(shp)-1)+0.4), color='black')
                plt.gca().add_patch(circle)


        plt.ylim(-0.02,0.1*(len(shp)-1)+0.02)
        plt.xlim(-0.02,1.02)

        ii = 0
        for p in model.parameters():


            if len(p.shape) == 2:
                p_shp = p.shape
                p = p/torch.abs(p).max()
                for i in range(p_shp[0]):
                    for j in range(p_shp[1]):
                        if p[i,j] > 0:
                            plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*p_shp[1])+j/p_shp[1]], [0.1*(ii+1),0.1*ii], lw=1*np.abs(p[i,j].detach().numpy()), color="blue")
                        else:
                            plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*p_shp[1])+j/p_shp[1]], [0.1*(ii+1),0.1*ii], lw=1*np.abs(p[i,j].detach().numpy()), color="red")

                formulas = ["Class 1", "Class 2"]
                if ii == 0:
                    for j in range(p_shp[1]):
                        plt.text(1/(2*p_shp[1])+j/p_shp[1]-0.05, 0.1*ii-0.04, "$x_{}$".format(model.in_perm[j].long()+1), fontsize=15)
                ii += 1


        for j in range(p_shp[0]):
            plt.text(1/(2*p_shp[0])+j/p_shp[0]-0.15, 0.1*ii+0.02, formulas[model.out_perm[j].long()], fontsize=15)

        plt.gca().axis('off')
        plt.title("step={}".format(epoch), fontsize=15, y=1.1)


        plt.subplot(2,1,2)


        start_x = X[:,0].min()-0.1
        end_x = X[:,0].max()+0.1
        start_y = X[:,1].min()-0.1
        end_y = X[:,1].max()+0.1
        n_values = 30

        x_vals = np.linspace(start_x.detach().numpy(), end_x.detach().numpy(), n_values)
        y_vals = np.linspace(start_y.detach().numpy(), end_y.detach().numpy(), n_values)
        XX, YY = np.meshgrid(x_vals, y_vals)
        pred = model(torch.tensor([XX.reshape(-1,), YY.reshape(-1,)], dtype=torch.float).permute(1,0))
        pred = pred[:,1] - pred[:,0]

        #ZZ = np.sqrt(XX**2 + YY**2)

        cp = plt.contourf(XX, YY, pred.reshape(n_values,n_values).detach().numpy(), [-100,0.,100.], colors=["green","orange"], alpha=0.2)
        color = ['green', 'orange']

        for i in range(n_sample):
            plt.scatter(X[i,0].detach().numpy(),X[i,1].detach().numpy(),color=color[y[i]])


        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r"$x_1$", fontsize=15)
        plt.ylabel(r"$x_2$", fontsize=15)
    
        plt.savefig(f"figures/{save_model_indx}/{epoch}.png")
        
    if (epoch+1) % SWAP_INTERVAL == 0:
        model.relocate()
