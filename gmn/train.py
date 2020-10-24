import torch
import torch.nn.functional as F


def train(model, optimizer, loader, device):
    model.train()
    total_ce_loss, total_kl_loss = 0, 0
    for data in loader:
        data.to(device)

        optimizer.zero_grad()

        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        out = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, data.y.view(-1), reduction='mean')

        loss.backward()
        optimizer.step()

        total_ce_loss += loss.item() * data.y.size(0)
        total_kl_loss += kl.item() * data.y.size(0)
    return total_ce_loss / len(loader.dataset), total_kl_loss / len(loader.dataset)


def kl_train(model, optimizer, loader, device):
    total_kl_loss = 0.0
    total_ce_loss = 0.0

    optimizer.zero_grad()
    for data in loader:
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        out = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, data.y.view(-1), reduction='mean')
        kl.backward()

        total_kl_loss += kl.item() * data.y.size(0)
        total_ce_loss += loss.item() * data.y.size(0)
    optimizer.step()

    return total_ce_loss/len(loader.dataset), total_kl_loss/len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, evaluator=None):
    model.eval()
    loss, kl_loss, correct = 0, 0, 0
    y_pred, y_true = [], []
    for data in loader:
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)

        y_pred.append(out[:, 1])
        y_true.append(data.y)

        out = F.log_softmax(out, dim=-1)
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction='mean').item()* data.y.size(0)
        kl_loss += kl.item() * data.y.size(0)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if evaluator is None:
        acc = correct/len(loader.dataset)
    else:
        acc = evaluator.eval({'y_pred': y_pred.view(y_true.shape), 'y_true': y_true})[evaluator.eval_metric]

    return acc, loss / len(loader.dataset), kl_loss / len(loader.dataset)


################## regression stuff
def train_regression(model, optimizer, loader, device):
    model.train()
    total_ce_loss, total_kl_loss = 0, 0
    for data in loader:
        data.to(device)

        optimizer.zero_grad()

        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = F.l1_loss(out, data.y.unsqueeze(1), reduction='mean')

        loss.backward()
        optimizer.step()

        total_ce_loss += loss.item() * data.num_graphs
        total_kl_loss += kl.item() * data.num_graphs
    return -total_ce_loss / len(loader.dataset), total_kl_loss / len(loader.dataset)


def kl_train_regression(model, optimizer, loader, device):
    total_kl_loss = 0.0
    total_ce_loss = 0.0

    optimizer.zero_grad()
    for data in loader:
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = F.l1_loss(out, data.y.unsqueeze(1), reduction='mean')
        kl.backward()

        total_kl_loss += kl.item() * data.num_graphs
        total_ce_loss += loss.item() * data.num_graphs
    optimizer.step()

    return -total_ce_loss/len(loader.dataset), total_kl_loss/len(loader.dataset)


@torch.no_grad()
def eval_regression(model, loader, device):
    model.eval()
    loss, kl_loss, correct = 0, 0, 0
    for data in loader:
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss += F.l1_loss(out, data.y.unsqueeze(1), reduction='mean').item()* data.num_graphs
        kl_loss += kl.item() * data.num_graphs
    return -loss / len(loader.dataset), -loss / len(loader.dataset), kl_loss / len(loader.dataset)
