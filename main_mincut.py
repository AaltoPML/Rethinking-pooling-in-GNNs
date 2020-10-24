import torch
import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from mincut.train import train, evaluate, train_regression, evaluate_regression
from mincut.mincutpool import MincutPool
from mincut.params import get_params
from utils import set_seed
from data.datasets import get_data

parser = argparse.ArgumentParser(description='MincutPool - PyTorch Geometric')
parser.add_argument('--seed', type=int, default=157)
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
parser.add_argument('--pooling_type', type=str, choices=['mlp', 'random'], default='random')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=32)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

if args.reproduce:
    args.hidden_dim, args.num_layers, args.batch_size = get_params(args.dataset)

args.logdir = f'{args.logdir}/{args.dataset}/{args.pooling_type}/' \
              f'{args.num_layers}_layers/{args.hidden_dim}_dim'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/') and args.save:
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

train_loader, val_loader, test_loader, stats, evaluator, encode_edge = get_data(args.dataset, args.batch_size,
                                                                                rwr=False, cleaned=args.cleaned)

model = MincutPool(num_features=stats['num_features'], num_classes=stats['num_classes'],
                   max_num_nodes=stats['max_num_nodes'], hidden=args.hidden_dim,
                   pooling_type=args.pooling_type, num_layers=args.num_layers, encode_edge=encode_edge).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                              patience=args.lr_decay_patience, verbose=True)

if args.dataset == 'ZINC':
    train = train_regression
    evaluate = evaluate_regression

train_sup_losses, train_lp_losses, train_entropy_losses = [], [], []
val_sup_losses, val_lp_losses, val_entropy_losses = [], [], []
test_sup_losses, test_lp_losses, test_entropy_losses = [], [], []
val_accuracies, test_accuracies = [], []

epochs_no_improve = 0  # used for early stopping
for epoch in range(1, args.max_epochs + 1):

    # train
    train_sup_loss, train_lp_loss, train_entropy_loss = \
        train(model, optimizer, train_loader, device)

    # validation
    val_acc, val_sup_loss, val_lp_loss, val_entropy_loss \
        = evaluate(model, val_loader, device, evaluator=evaluator)

    # test
    test_acc, test_sup_loss, test_lp_loss, test_entropy_loss = \
        evaluate(model, test_loader, device, evaluator=evaluator)

    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

    train_sup_losses.append(train_sup_loss)
    train_lp_losses.append(train_lp_loss)
    train_entropy_losses.append(train_entropy_loss)

    val_sup_losses.append(val_sup_loss)
    val_lp_losses.append(val_lp_loss)
    val_entropy_losses.append(val_entropy_loss)

    test_sup_losses.append(test_sup_loss)
    test_lp_losses.append(test_lp_loss)
    test_entropy_losses.append(test_entropy_loss)

    if (epoch-1) % args.interval == 0:
        print(f'{epoch:03d}: Train Sup Loss: {train_sup_loss:.3f}, '
          f'Val Sup Loss: {val_sup_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, '
          f'Test Sup Loss: {test_sup_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}')

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
    torch.save(best_model, f'{args.logdir}/models/mincut_{args.seed}.model')

torch.save({
    'train_sup_losses': tensor(train_sup_losses),
    'train_lp_losses': tensor(train_lp_losses),
    'train_entropy_losses': tensor(train_entropy_losses),
    'val_accuracies': tensor(val_accuracies),
    'val_sup_losses': tensor(val_sup_losses),
    'val_lp_losses': tensor(val_lp_losses),
    'val_entropy_losses': tensor(val_entropy_losses),
    'test_accuracies': tensor(test_accuracies),
    'test_sup_losses': tensor(test_sup_losses),
    'test_lp_losses': tensor(test_lp_losses),
    'test_entropy_losses': tensor(test_entropy_losses)
}, f'{args.logdir}/mincut_{args.seed}.results')
