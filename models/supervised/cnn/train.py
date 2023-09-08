import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.supervised.cnn.model import CNN
from models.data.mnist import MNISTDataset
import os

def train(model, train_loader, val_loader, criterion, optimizer, epoch, num_epochs, save_step, save_dir):
    batch_size = train_loader.batch_size
    model.train()
    running_loss = 0.0
    store_loss = []
    for step, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        store_loss.append(loss.item())
        if step % save_step == 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{int(epoch*len(train_loader)/batch_size+step)}.pth"))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {(100 * correct / total):.2f}%')
    return running_loss / len(train_loader), store_loss

def main(args):
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters from args
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_val_split = args.train_val_split

    # Prepare data
    mnist_data = MNISTDataset()
    n_train = int(len(mnist_data) * train_val_split)
    n_val = len(mnist_data) - n_train
    train_data, val_data = random_split(mnist_data, [n_train, n_val])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Initialize the model
    cnn_layers = [(1, 16, 3, 1), (16, 32, 3, 1)]
    fc_layers = [(32 * 24 * 24, 128, nn.ReLU()), (128, 64, nn.ReLU())]
    output_dim = 10
    
    model = CNN(cnn_layers, fc_layers, output_dim).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory if it doesn't exist
    save_dir = f"./models/supervised/cnn/saved_models/{args.name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_loss = []
    for epoch in range(num_epochs):
        train_loss, store_loss = train(model, train_loader, val_loader, criterion, optimizer, epoch, num_epochs, args.save_interval, save_dir)
        total_loss += store_loss
        
    if args.plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(total_loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(save_dir, "loss.png"))
        plt.close()

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Train-Validation split ratio')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval to save model state')
    parser.add_argument('--name', type=str, default="cnn_mnist", help='Name to save the model under')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot_loss', type=bool, default=True, help='Plot training loss')
    
    args = parser.parse_args()
    main(args)
