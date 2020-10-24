def get_params(dataset):
    if dataset == 'NCI1' or dataset == 'NCI109':
        hidden, num_layers, batch_size = 32, 3, 16
    elif dataset == 'SMNIST':
        hidden, num_layers, batch_size = 32, 3, 16
    elif dataset == 'ZINC':
        hidden, num_layers, batch_size = 32, 3, 16
    elif dataset == 'IMDB-BINARY':
        hidden, num_layers, batch_size = 32, 3, 16
    elif dataset == 'ogbg-molhiv':
        hidden, num_layers, batch_size = 64, 3, 32
    else:
        hidden, num_layers, batch_size = 32, 3, 16
    return hidden, num_layers, batch_size
