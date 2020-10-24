import torch
import torch.nn.functional as F


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data.to(device)
        out = model(data)
        out = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, data.y.view(-1), reduction='mean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, evaluator=None):
    model.eval()
    loss, correct = 0, 0
    y_pred, y_true = [], []
    for data in loader:
        data.to(device)
        out = model(data)

        y_pred.append(out[:, 1])
        y_true.append(data.y)

        out = F.log_softmax(out, dim=-1)
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction='mean').item()* data.num_graphs

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if evaluator is None:
        acc = correct/len(loader.dataset)
    else:
        acc = evaluator.eval({'y_pred': y_pred.view(y_true.shape), 'y_true': y_true})[evaluator.eval_metric]
    return acc, loss / len(loader.dataset)


def train_regression(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data.to(device)
        out = model(data)
        loss = F.l1_loss(out, data.y.unsqueeze(1), reduction='mean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_regression(model, loader, device, evaluator=None):
    model.eval()
    loss, lp_loss, e_loss, mae = 0, 0, 0, 0
    for data in loader:
        data.to(device)
        out = model(data)
        loss += F.l1_loss(out, data.y.unsqueeze(1), reduction="mean")*data.num_graphs
    return -loss / len(loader.dataset), loss / len(loader.dataset)
