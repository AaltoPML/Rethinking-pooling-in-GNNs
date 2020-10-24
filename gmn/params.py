def get_params(dataset):
    if dataset=='NCI1':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [10, 1], 64, 16, 100
    elif dataset == 'IMDB-BINARY':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 1, [32, 1], 64, 16, 16
    elif dataset == 'SMNIST':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 10, [32, 1], 64, 16, 16
    elif dataset == 'ZINC':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [10, 1], 64, 16, 100
    elif dataset == 'PROTEINS':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [32, 1], 64, 16, 16
    elif dataset == 'NCI109':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [32, 1], 64, 16, 16
    elif dataset == 'DD':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [32, 8, 1], 64, 16, 64
    elif dataset == 'ogbg-molhiv':
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [32, 1], 64, 16, 16
    else:
        num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim = 5, [10, 1], 64, 16, 64
    return num_heads, num_keys, hidden_dim, pos_hidden_dim, mem_hidden_dim
