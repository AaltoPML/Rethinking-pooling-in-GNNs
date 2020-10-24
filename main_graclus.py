import torch
import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from graclus.train import train, evaluate, train_regression, evaluate_regression
from utils import set_seed
from data.datasets import get_data
from graclus.graclus import Graclus
from graclus.params import get_params


parser = argparse.ArgumentParser(description='Graclus - PyTorch Geometric.')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--logdir', type=str, default='results/graclus', help='Log directory.')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['SMNIST', 'ZINC', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY', 'DD', 'ogbg-molhiv'])
parser.add_argument('--reproduce', action='store_true', default=False)
parser.add_argument('--cleaned', action='store_true', default=False, help='Used to eliminate isomorphisms in IMDB')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--early_stop_patience', type=int, default=50)
parser.add_argument('--lr_decay_patience', type=int, default=10)

# model
parser.add_argument('--pooling_type', type=str, choices=['graclus', 'complement', 'none'], default='graclus')
parser.add_argument('--no_cat', action='store_true', default=True, help='Residual links.')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=64)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

if args.reproduce:
    args.num_layers, args.hidden_dim, batch_size = get_params(args.dataset)

args.logdir = f'{args.logdir}/{args.dataset}/{args.pooling_type}/' \
              f'{args.num_layers}_layers/{args.hidden_dim}_dim'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/'):
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

train_loader, val_loader, test_loader, stats, evaluator, encode_edge = get_data(args.dataset, args.batch_size,
                                                                                rwr=False, cleaned=args.cleaned)

model = Graclus(num_features=stats['num_features'], num_classes=stats['num_classes'],
                num_layers=args.num_layers, hidden=args.hidden_dim, no_cat=args.no_cat,
                pooling_type=args.pooling_type, encode_edge=encode_edge).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                              patience=args.lr_decay_patience, verbose=True)

if args.dataset == 'ZINC':
    train = train_regression
    evaluate = evaluate_regression

train_losses, val_losses, test_losses = [], [], []
val_accuracies, test_accuracies = [], []

epochs_no_improve = 0  # used for early stopping
for epoch in range(1, args.max_epochs + 1):

    # train
    train_loss = train(model, optimizer, train_loader, device)

    # validation
    val_acc, val_loss = evaluate(model, val_loader, device, evaluator=evaluator)

    # test
    test_acc, test_loss = evaluate(model, test_loader, device, evaluator=evaluator)

    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    if (epoch - 1) % args.interval == 0:
        print(f'{epoch:03d}: Train Sup Loss: {train_loss:.3f},'
              f' Val Sup Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, '
              f'Test Sup Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}')

    scheduler.step(val_acc)

    if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2-epochs_no_improve]:
        epochs_no_improve = epochs_no_improve + 1

    else:
        epochs_no_improve = 0
        best_model = model.state_dict()

    if epochs_no_improve >= args.early_stop_patience:
        print('Early stopping!')
        break

if args.save:
    torch.save(best_model, f'{args.logdir}/models/graclus_{args.seed}.model')

torch.save({
    'train_losses': tensor(train_losses),
    'val_accuracies': tensor(val_accuracies),
    'val_losses': tensor(val_losses),
    'test_accuracies': tensor(test_accuracies),
    'test_losses': tensor(test_losses),
}, f'{args.logdir}/graclus_{args.seed}.results')
