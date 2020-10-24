import torch

import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import set_seed
import os
import json
from data.datasets import get_data
from gmn.GMN import GMN
from gmn.train import train, evaluate, kl_train, train_regression, eval_regression, kl_train_regression
from gmn.params import get_params


parser = argparse.ArgumentParser(description='Graph Memory Nets (with PyTorch Geometric)')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--logdir', type=str, default='results/gmn', help='Log directory')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['ogbg-molhiv','SMNIST', 'ZINC', 'DD', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY'])
parser.add_argument('--reproduce', action='store_true', default=False)
parser.add_argument('--cleaned', action='store_true', default=False, help='Used to eliminate isomorphisms in IMDB')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--early_stop_patience', type=int, default=50)
parser.add_argument('--lr_decay_patience', type=int, default=10)

# model
parser.add_argument('--variant', type=str, default='random', choices=['gmn', 'random', 'distance'])
parser.add_argument('--kl_period', type=int, default=5)
parser.add_argument('--num_heads', type=int, default=5)
parser.add_argument('--num_keys', type=int, nargs='+', default=[10, 1])
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimensionality of final MLP')
parser.add_argument('--pos_hidden_dim', type=int, default=16, help='Positional hidden dimensionality')
parser.add_argument('--mem_hidden_dim', type=int, default=5, help='Memory-layer hidden dimensionality')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

args.logdir = f'{args.logdir}/{args.dataset}/{args.variant}'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/'):
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

train_loader, val_loader, test_loader, stats, evaluator, encode_edge = get_data(args.dataset, args.batch_size,
                                                                                rwr=False, cleaned=args.cleaned)

if args.reproduce:
    args.num_heads, args.num_keys, args.hidden_dim, args.pos_hidden_dim, args.mem_hidden_dim = get_params(args.dataset)

model = GMN(stats['num_features'], stats['max_num_nodes'], stats['num_classes'], args.num_heads, args.hidden_dim,
            args.num_keys, args.mem_hidden_dim, variant=args.variant, encode_edge=encode_edge).to(device)

# Optimizers
no_keys_param_list = [param for name, param in model.named_parameters() if 'keys' not in name]
optimizer = Adam(no_keys_param_list, lr=args.lr)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=args.lr_decay_patience)

kl_optimizer = Adam(model.parameters(), lr=args.lr)
kl_scheduler = ReduceLROnPlateau(kl_optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=args.lr_decay_patience)

if args.dataset == 'ZINC':
    train = train_regression
    evaluate = eval_regression
    kl_train = kl_train_regression

train_sup_losses, train_kl_losses = [], []
val_sup_losses, val_kl_losses = [], []
test_sup_losses, test_kl_losses = [], []
val_accuracies, test_accuracies = [], []

kl_period = args.kl_period
epochs_no_improve = 0  # used for early stopping
for epoch in range(1, args.max_epochs + 1):
    # train
    if epoch % kl_period == 0 and args.variant == 'gmn':
        train_sup_loss, train_kl_loss = kl_train(model, kl_optimizer, train_loader, device)
    else:
        train_sup_loss, train_kl_loss = train(model, optimizer, train_loader, device)

    # validation
    acc, val_sup_loss, val_kl_loss = evaluate(model, val_loader, device, evaluator=evaluator)

    # test
    test_acc, test_sup_loss, test_kl_loss = evaluate(model, test_loader, device, evaluator=evaluator)

    val_accuracies.append(acc)
    train_sup_losses.append(train_sup_loss)
    train_kl_losses.append(train_kl_loss)

    val_sup_losses.append(val_sup_loss)
    val_kl_losses.append(val_kl_loss)

    test_sup_losses.append(test_sup_loss)
    test_kl_losses.append(test_kl_loss)
    test_accuracies.append(test_acc)

    print(f'{epoch:03d}: Train Sup Loss: {train_sup_loss:.3f},'
          f' Val Sup Loss: {val_sup_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, '
          f'Test Sup Loss: {test_sup_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}')

    scheduler.step(acc)

    if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2 -epochs_no_improve]:
        epochs_no_improve = epochs_no_improve + 1
    else:
        epochs_no_improve = 0
        best_model = model.state_dict()

    if epochs_no_improve >= args.early_stop_patience:
        print('Early stopping!')
        break

if args.save:
    torch.save(best_model, f'{args.logdir}/models/gmn_{args.seed}.model')

torch.save({
    'train_sup_losses': tensor(train_sup_losses),
    'train_kl_losses': tensor(train_kl_losses),
    'val_accuracies': tensor(val_accuracies),
    'val_sup_losses': tensor(val_sup_losses),
    'val_kl_losses': tensor(val_kl_losses),
    'test_accuracies': tensor(test_accuracies),
    'test_sup_losses': tensor(test_sup_losses),
    'test_kl_losses': tensor(test_kl_losses),
}, f'{args.logdir}/{args.seed}.results')

