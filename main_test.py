import numpy as np
import os
import random
import data_processing
from torch_geometric.loader import DataLoader
import torch
from model import LearnedSimulator
import functional

if __name__ == '__main__':

    params = {
        'run_name': 'sim_entire_dataset',
        'dataset_dir': ['data/10Nodes_by_tree/trial', 'data/20Nodes_by_tree/trial'],
        'num_trees_per_dir': [27, 43],
        'simulated_dataset': True,
        'seed': 0,
        'train_validation_split': 0.9,
        'visualize': True,
    }

    output_dir = 'output/{}'.format(params['run_name'])
    model_weights_path = os.path.join(output_dir, 'best_model.pt')
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    test_dataset = []
    print('Loading data from...')
    for i_dir, dataset_dir in enumerate(params['dataset_dir']):
        train_val_split = int(params['num_trees_per_dir'][i_dir]*params['train_validation_split'])
        for i in range(0,params['num_trees_per_dir'][i_dir]):
            if i<train_val_split:
                continue
            else:
                d = dataset_dir+str(i)
                print(d)
                X_edges, X_force, X_pos, Y_pos = data_processing.load_npy(d, params['simulated_dataset'])
                test_dataset += data_processing.make_dataset(X_edges, X_force, X_pos, Y_pos, 
                                make_directed=True, prune_augmented=False, rotate_augmented=False)
    print('Train dataset size: {}'.format(len(test_dataset)))

    if params['visualize']:
        batch_size = 1
    else:
        batch_size = 256
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedSimulator().to(device)
    model.load_state_dict(torch.load(model_weights_path))
    functional.test(model, test_loader, device, visualize=params['visualize'])
    