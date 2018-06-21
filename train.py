import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import model
import utils
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='FashionSimpleNet', help="model")
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
args = parser.parse_args()

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)



# Define transforms.
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create dataloaders. Use pin memory if cuda.
kwargs = {'pin_memory': True} if device=='cuda:0' else {}

trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.nworkers, **kwargs)
valset = datasets.FashionMNIST('data', train=False, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)

def train(epochs, patience):
    patience = patience
    best_loss = 1e4
    for e in range(epochs):
        start = time.time()
        train_loss, train_acc = run_model(net, train_loader,
                                      criterion, optimizer)
        val_loss, val_acc = run_model(net, train_loader,
                                      criterion, optimizer, False)
        end = time.time()

        # print stats
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format(e, train_loss, train_acc, val_loss,
                                        val_acc, end - start)
        print(stats)

        # early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break



if __name__ == '__main__':

    # Init network, criterion and early stopping
    net = model.__dict__[args.model]().to(device)
    criterion = torch.nn.CrossEntropyLoss()



    # Define optimizer
    optimizer = optim.Adam(net.parameters())

    train(args.nepochs, args.patience)
