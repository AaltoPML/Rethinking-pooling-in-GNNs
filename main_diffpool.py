import torch
import argparse
from torch.optim import Adam
from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from diffpool.train import train, evaluate, train_regression, evaluate_regression
from utils import set_seed
from data.datasets import get_data

parser = argparse.ArgumentParser(description='DiffPool - PyTorch Geometric.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--logdir', type=str, default='results/diffpool', help='Log directory')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['ogbg-molhiv','SMNIST', 'ZINC', 'DD', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY'])
parser.add_argument('--reproduce', action='store_true', default=False)
parser.add_argument('--cleaned', action='store_true', default=False, help='Used to eliminate isomorphisms in IMDB')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--early_stop_patience', type=int, default=50)
parser.add_argument('--lr_decay_patience', type=int, default=10)

# model
parser.add_argument('--v2', action='store_true', default=True, help='Alternative version of DiffPool')
parser.add_argument('--pooling_type', type=str, default='gnn',
                    choices=['gnn', 'bernoulli', 'categorical', 'normal', 'uniform'])
parser.add_argument('--invariant', action='store_true', default=False, help='Invariant version of random pooling')
parser.add_argument('--gnn_dim', type=int, default=32)
parser.add_argument('--mlp_hidden_dim', type=int, default=50)
parser.add_argument('--num_pooling_layers', type=int, default=2)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

if args.v2:
    from diffpool.diffpool_v2 import DiffPool
    from diffpool.params_v2 import get_params
    args.logdir = args.logdir + '_v2'
else:
    from diffpool.diffpool import DiffPool
    from diffpool.params import get_params

linkpred = False
if args.pooling_type == 'gnn':
    linkpred = True

if args.reproduce:
    args.gnn_dim, args.mlp_hidden_dim, \
    args.num_pooling_layers, args.batch_size = get_params(args.dataset)

args.logdir = f'{args.logdir}/{args.dataset}/{args.pooling_type}/' \
              f'{args.num_pooling_layers}_layers/{args.gnn_dim}_dim'

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/') and args.save:
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


train_loader, val_loader, test_loader, stats, evaluator, encode_edge = get_data(args.dataset, args.batch_size,
                                                                                rwr=False, cleaned=args.cleaned)

model = DiffPool(num_features=stats['num_features'], num_classes=stats['num_classes'],
                 max_num_nodes=stats['max_num_nodes'], num_layers=args.num_pooling_layers,
                 gnn_hidden_dim=args.gnn_dim, gnn_output_dim=args.gnn_dim,
                 mlp_hidden_dim=args.mlp_hidden_dim, pooling_type=args.pooling_type,
                 invariant=args.invariant, encode_edge=encode_edge, pre_sum_aggr=args.dataset=='IMDB-BINARY').to(device)

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
        train(model, optimizer, train_loader, linkpred, device)

    # validation
    val_acc, val_sup_loss, val_lp_loss, val_entropy_loss \
        = evaluate(model, val_loader, device, evaluator=evaluator)

    # test
    test_acc, test_sup_loss, test_lp_loss, test_entropy_loss = \
        evaluate(model, test_loader, device, evaluator=evaluator)

    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

    train_sup_losses.append(train_sup_loss)  # training losses
    train_lp_losses.append(train_lp_loss)
    train_entropy_losses.append(train_entropy_loss)

    val_sup_losses.append(val_sup_loss)  # validation losses
    val_lp_losses.append(val_lp_loss)
    val_entropy_losses.append(val_entropy_loss)

    test_sup_losses.append(test_sup_loss)  # test losses
    test_lp_losses.append(test_lp_loss)
    test_entropy_losses.append(test_entropy_loss)

    if (epoch - 1) % args.interval == 0:
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
    torch.save(best_model, f'{args.logdir}/models/diffpool_{args.seed}.model')

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
}, f'{args.logdir}/diffpool_{args.seed}.results')
