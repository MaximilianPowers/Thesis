import torch
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
import math
import argparse
import os
import sys
from models.data.mog import MixtureOfGaussiansDataset
from models.unsupervised.gmm_vae.model import GMVAE
import json
import utils
import csv

rel_path = os.path.dirname(os.path.realpath(__file__))

save_model_indx = len(os.listdir(f'{rel_path}/saved_models'))
os.mkdir(f"{rel_path}/saved_models/{save_model_indx}")

log_ref_file = rel_path + '/log_ref.json'
if os.path.exists(log_ref_file):
    # If it does, load the existing data
    with open(log_ref_file, "r") as file:
        data = json.load(file)
else:
    # If it doesn't, create an empty dictionary
    data = {}

supported_datasets = ['toy']


parser = argparse.ArgumentParser(description='Gaussian Mixture VAE')

# Data related arguments
parser.add_argument('--dataset', default='toy', 
                    help='dataset to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--n_samples', type=int, default=1000, 
                    help='Number of samples to generate')
parser.add_argument('--n_samples_test', type=int, default=200, 
                    help='Number of samples to generate for testing')
parser.add_argument('--x-size', type=int, default=2, metavar='N', 
                    help='dimension of x')
parser.add_argument('--K', type=int, default=4, metavar='N', 
                    help='number of clusters')
parser.add_argument('--continuous', default=True, help='data is continuous',
					action='store_true')
# Model related arguments

parser.add_argument('--hidden-size', type=int, default=200, metavar='N',
					help='dimension of hidden layer')
parser.add_argument('--w-size', type=int, default=20, metavar='N',
					help='dimension of latent variable')

# Training related arguments

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--learning-rate', type=float, default=1e-4,
					help='learning rate for optimizer')

# Misc arguments
parser.add_argument('--log_interval', type=int, default=1, help='Run test every .. epochs')
parser.add_argument('--save_interval', type=int, default=1, help='Save model every .. epochs')

args = parser.parse_args()

if args.dataset not in supported_datasets:
	raise ValueError('Unsupported dataset: ' + args.dataset)

if args.dataset == 'toy':
    means = torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=torch.float32)
    covariances = 0.1*torch.stack([torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)])
    dataset = MixtureOfGaussiansDataset(means, covariances, args.n_samples)
    test_dataset = MixtureOfGaussiansDataset(means, covariances, args.n_samples_test)       
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    args.K = len(means)

# select gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.device = device

data[save_model_indx] = {
    'dataset': args.dataset,
    'seed': args.seed,
    'n_samples': args.n_samples,
    'n_samples_test': args.n_samples_test,
    'x_size': args.x_size,
    'K': args.K,
    'continuous': args.continuous,
    'hidden_size': args.hidden_size,
    'w_size': args.w_size,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'save_interval': args.save_interval
}

with open(log_ref_file, "w") as file:
    json.dump(data, file, indent=4)
	
EPOCHS = args.epochs
LOG_INTERVAL = args.log_interval
SAVE_INTERVAL = args.save_interval

torch.manual_seed(args.seed)
gmvae = GMVAE(args).to(device)
optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)

def loss_function(recon_X, X, mu_w, logvar_w, qz,
	mu_x, logvar_x, mu_px, logvar_px, x_sample, lambda_=1):
	N = X.size(0) # batch size

	# 1. Reconstruction Cost = -E[log(P(y|x))]
	# for dataset such as mnist
	if not args.continuous:
		recon_loss = F.binary_cross_entropy(recon_X, X,
			size_average=False)
	# for datasets such as tvsum, spiral
	elif args.continuous:
		# unpack Y into mu_Y and logvar_Y
		mu_recon_X, logvar_recon_X = recon_X

		# use gaussian criteria
		# negative LL, so sign is flipped
		# log(sigma) + 0.5*2*pi + 0.5*(x-mu)^2/sigma^2
		recon_loss = 0.5 * torch.sum(logvar_recon_X + math.log(2*math.pi) \
			+ (X - mu_recon_X).pow(2)/logvar_recon_X.exp())

	# 2. KL( q(w) || p(w) )
	KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

	# 3. KL( q(z) || p(z) )
	KLD_Z = torch.sum(qz * torch.log(args.K * qz + 1e-10))

	# 4. E_z_w[KL(q(x)|| p(x|z,w))]
	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
	mu_x = mu_x.unsqueeze(-1)
	mu_x = mu_x.expand(-1, args.x_size, args.K)

	logvar_x = logvar_x.unsqueeze(-1)
	logvar_x = logvar_x.expand(-1, args.x_size, args.K)

	# shape (-1, x_size, K)
	KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + \
		((logvar_x.exp() + (mu_x - mu_px).pow(2))/logvar_px.exp())) \
		- 1)

	# transpose to change dim to (-1, x_size, K)
	# KLD_QX_PX = KLD_QX_PX.transpose(1,2)
	qz = qz.unsqueeze(-1)
	qz = qz.expand(-1, args.K, 1)

	E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

	# 5. Entropy criterion
	
	# CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
	# compute likelihood
	
	x_sample = x_sample.unsqueeze(-1)
	x_sample =  x_sample.expand(-1, args.x_size, args.K)

	temp = 0.5 * args.x_size * math.log(2 * math.pi)
	# log likelihood
	llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2))/logvar_px.exp(), dim=1) \
			- 0.5 * torch.sum(logvar_px, dim=1) - temp

	lh = F.softmax(llh, dim=1)

	# entropy
	CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh))
	
	loss = recon_loss + lambda_*(KLD_W + KLD_Z + E_KLD_QX_PX)
	return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV

def train(epoch):
    lambda_ = min(1, epoch/50)
    gmvae.train()
    store_batch = torch.randint(0, len(train_loader), (1,))
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.to(device)

        target = target.to(device)
        optimizer.zero_grad()

        mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
            x_sample = gmvae(data)

        loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
            = loss_function(Y, data, mu_w, logvar_w, qz,
            mu_x, logvar_x, mu_px, logvar_px, x_sample, lambda_)

        loss.backward()

        optimizer.step()
        
        
        if epoch % LOG_INTERVAL == 0 and batch_idx == store_batch.item():
            str_= 'Train Epoch: {:5d}[{:5d}/{:5d} loss: {:.6f} ReconL: {:.6f} E(KLD(QX||PX)): {:.6f} CV: {:.6f} KLD_W: {:.6f} KLD_Z: {:.6f}]'.format(
                    epoch, batch_idx+1, len(train_loader),
	        		loss.item(), BCE.item(), E_KLD_QX_PX.item(),
	        		CV.item(), KLD_W.item(), KLD_Z.item())
            vals = [epoch, batch_idx+1, "Train",
                loss.item(), BCE.item(), E_KLD_QX_PX.item(), CV.item(),
                KLD_W.item(), KLD_Z.item()]
            print(str_)
    return vals



def test(epoch):
    gmvae.eval()
    store_batch = torch.randint(0, len(test_loader), (1,))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
			
            target = target.to(device)

            mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
				x_sample = gmvae(data)

            loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
				= loss_function(Y, data, mu_w, logvar_w, qz,
				mu_x, logvar_x, mu_px, logvar_px, x_sample)


            if epoch % LOG_INTERVAL == 0 and batch_idx == store_batch.item():
                    str_= 'Test Epoch:   {:5d}[{:5d}/{:5d} loss: {:.6f} ReconL: {:.6f} E(KLD(QX||PX)): {:.6f} CV: {:.6f} KLD_W: {:.6f} KLD_Z: {:.6f}]'.format(
                        epoch, batch_idx+1, len(test_loader),
	        	    	loss.item(), BCE.item(), E_KLD_QX_PX.item(),
	        	    	CV.item(), KLD_W.item(), KLD_Z.item())
                    vals = [epoch, batch_idx+1, "Test",
		    			loss.item(), BCE.item(), E_KLD_QX_PX.item(), CV.item(),
		    			KLD_W.item(), KLD_Z.item()]
                    print(str_)
    return vals
			
log_store = ["Epoch", "BatchIndex", "Mode", "Loss", "ReconLoss", "E(KLD(QX||PX))", "CV", "KLD_W", "KLD_Z"]
with open(f'{rel_path}/logs/{save_model_indx}.csv', 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(log_store)
res = []

for epoch in range(0, args.epochs):
    # train the network
    vals = train(epoch)
    res.append(vals)

    # test the network
    vals = test(epoch)
    res.append(vals)
        
    if epoch % args.save_interval == 0:
        torch.save(gmvae.state_dict(), f'{rel_path}/saved_models/{save_model_indx}/model_{epoch}.pth')

with open(f'{rel_path}/logs/{save_model_indx}.csv', 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(res)