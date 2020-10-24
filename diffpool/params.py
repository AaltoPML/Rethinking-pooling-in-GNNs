def get_params(dataset):
    if dataset == 'NCI1':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 32, 1, 16
    elif dataset == 'SMNIST':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 64
    elif dataset == 'ZINC':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 56, 56, 1, 64
    elif dataset == 'IMDB-BINARY':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 1, 8
    elif dataset == 'ogbg-molhiv':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 2, 32
    elif dataset == 'PROTEINS':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 2, 8
    elif dataset == 'NCI109':
        gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 32, 1, 16
    else:
        gnn_dim, gnn_output_dim, mlp_hidden_dim, num_pooling_layers, batch_size = 32, 50, 2, 8
    return gnn_dim, mlp_hidden_dim, num_pooling_layers, batch_size
