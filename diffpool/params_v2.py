def get_params(dataset):
    if dataset == 'NCI1' or dataset == 'NCI109':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 16 
    elif dataset == 'SMNIST':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 64
    elif dataset == 'ZINC':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 56, 56, 2, 64
    elif dataset == 'IMDB-BINARY':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 16
    elif dataset == 'ogbg-molhiv':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 64
    elif dataset == 'PROTEINS':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 16
    else:
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 16
    return gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size
