def get_params(dataset):
    if dataset == 'NCI1':
        num_layers, hidden_dim, batch_size = 3, 64, 8
    elif dataset == 'SMNIST':
        num_layers, hidden_dim, batch_size = 2, 64, 8
    elif dataset == 'ZINC':
        num_layers, hidden_dim, batch_size = 3, 64, 8
    elif dataset == 'IMDB-BINARY':
        num_layers, hidden_dim, batch_size = 2, 64, 8
    elif dataset == 'ogbg-molhiv':
        num_layers, hidden_dim, batch_size = 2, 128, 64
    elif dataset == 'PROTEINS':
        num_layers, hidden_dim, batch_size = 3, 64, 64
    elif dataset == 'DD':
        num_layers, hidden_dim, batch_size = 3, 64, 64
    elif dataset == 'NCI109':
        num_layers, hidden_dim, batch_size = 3, 64, 64
    else:
        num_layers, hidden_dim, batch_size = 3, 64, 16
    return num_layers, hidden_dim, batch_size
