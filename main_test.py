import numpy as np
import random
import data_processing
from torch_geometric.loader import DataLoader
import torch
from model import GCN
import functional

if __name__ == '__main__':
    params = {
        'dataset_dir': 'data/tree_dataset/trial',
        'seed': 0,
        'model_weights': 'model/best_model.pt'
    }

    np.random.seed(params['seed'])
    random.seed(params['seed'])

    
    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    for i in range(2,9):
        d = params['dataset_dir']+str(i)
        X_edges, X_force, X_pos, Y_pos = data_processing.load_npy(d)
        X_force_list.append(X_force)
        X_pos_list.append(X_pos)
        Y_pos_list.append(Y_pos)
    X_force_arr = np.concatenate(X_force_list)
    X_pos_arr = np.concatenate(X_pos_list)
    Y_pos_arr = np.concatenate(Y_pos_list)

    X_force_arr, X_pos_arr, Y_pos_arr = data_processing.shuffle_in_unison(X_force_arr, X_pos_arr, Y_pos_arr)

    train_val_split = int(len(X_force_arr)*0.9)

    X_force_train = X_force_arr[:train_val_split] 
    X_pos_train = X_pos_arr[:train_val_split] 
    Y_pos_train = Y_pos_arr[:train_val_split] 

    X_force_val = X_force_arr[train_val_split:] 
    X_pos_val = X_pos_arr[train_val_split:] 
    Y_pos_val = Y_pos_arr[train_val_split:] 


    val_dataset = data_processing.make_dataset(X_edges, X_force_val, X_pos_val, Y_pos_val, 
                    make_directed=True, prune_augmented=False, rotate_augmented=False)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    model.load_state_dict(torch.load(params['model_weights']))
    functional.test(model, test_loader, device)
    