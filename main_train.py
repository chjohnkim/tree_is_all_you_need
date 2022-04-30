import numpy as np
import random
import data_processing
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch
import copy
from model import GCN
import functional

if __name__ == '__main__':
    seed = 0
    params = {
        'dataset_dir': 'dataset_fullTree/trial',
        'seed': 0,
        'num_epochs': 3,
        'batch_size': 256, 
        'lr': 2e-3,
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


    train_dataset = data_processing.make_dataset(X_edges, X_force_train, X_pos_train, Y_pos_train, 
                    make_directed=True, prune_augmented=True, rotate_augmented=True)
    val_dataset = data_processing.make_dataset(X_edges, X_force_val, X_pos_val, Y_pos_val, 
                    make_directed=True, prune_augmented=True, rotate_augmented=False)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, min_lr=5e-4)

    train_loss_history = []
    val_loss_history = []
    best_loss = 1e9
    for epoch in range(1, params['num_epochs']+1):
        train_loss = functional.train(model, optimizer, criterion, train_loader, epoch, device)
        val_loss = functional.validate(model, criterion, val_loader, epoch, device)
        if val_loss<best_loss:
            best_loss=val_loss
            best_model = copy.deepcopy(model)
        scheduler.step(best_loss)
        print('Epoch {} | Train Loss: {} | Val Loss: {} | LR: {}'.format(epoch, train_loss, val_loss, scheduler._last_lr))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
    ax.plot(train_loss_history, 'r', label='train')
    ax.plot(val_loss_history, 'b', label='validation')
    ax.legend(loc="upper right")
    plt.show()
    functional.test(best_model, test_loader, device)
    torch.save(best_model.state_dict(), 'model.pt')