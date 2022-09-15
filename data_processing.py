import os
import numpy as np
import copy
import random
from queue import LifoQueue, Queue
from scipy.spatial.transform import Rotation
import torch
from torch_geometric.data import Data

def remove_duplicate_nodes(edges, init_positions, final_positions, X_force):
    tree_representative = init_positions[0]
    tree_representative = np.around(tree_representative, decimals=4)
    duplicates = [(0,1)] #treat 0 and 1 as duplicates, as 0 represents the base_link aka the floor, which should behave like the root
    for i, node in enumerate(tree_representative):
        for j, nodec in enumerate(tree_representative):
            if (node[:3] == nodec[:3]).all() and i != j and has_same_parent(i,j,edges):
                if i < j:
                    duplicates.append((i,j))
                else:
                    duplicates.append((j,i))
    duplicates = list(set(duplicates))
    while len(duplicates) > 0:
        original, duplicate = duplicates.pop()
        edges, init_positions, final_positions, duplicates, X_force = remove_duplicate(original, duplicate, edges, init_positions, final_positions, duplicates, X_force)
        duplicates = list(set(duplicates))
        duplicates = adjust_indexing(duplicates, duplicate)
        edges = adjust_indexing(edges, duplicate)
    edges = np.array(edges)
    return edges, init_positions[:,:,:3], final_positions[:,:,:3], X_force

def has_same_parent(i,j,edges):
    for parent, child in edges:
        if child == i:
            parent_i = parent
        if child == j:
            parent_j = parent
    return parent_i == parent_j

def remove_duplicate(original, duplicate, edge_def, init_positions, final_positions, duplicates, forces):
    init_positions = np.delete(init_positions, duplicate, axis=1)
    final_positions = np.delete(final_positions, duplicate, axis=1)

    new_edge_def = []
    new_duplicates = []
    for orig, dup in duplicates:
        if orig == duplicate:
            new_duplicates.append((original,dup))
        elif duplicate != dup and duplicate != orig:
            new_duplicates.append((orig,dup))

    for parent, child in edge_def:
        if duplicate == parent:
            new_edge_def.append((original,child))
        elif duplicate != parent and duplicate != child:
            new_edge_def.append((parent,child))

    for idx, force in enumerate(forces):
        if np.linalg.norm(force[duplicate]) != 0:
            forces[idx][original] += forces[idx][duplicate]
    
    forces = np.delete(forces, duplicate, axis=1)

    return new_edge_def, init_positions, final_positions, new_duplicates, forces

def adjust_indexing(tuple_list, deleted_index):
    new_tuple_list = []
    for i, j in tuple_list:
        if i > deleted_index:
            i = i-1
        if j > deleted_index:
            j = j-1
        new_tuple_list.append((i,j))
    return new_tuple_list

def get_topological_order(neighbor_dict, root=0):
    """
    Find the topological order of the tree.
    :param neighbor_dict: dict (key:int, val: set); the dictionary saving neighbor nodes.
    :param root: int; the root node index.
    :return topological_order: list of int; the topological order of the tree (start from root).
    """
    topological_order = []
    queue = LifoQueue()
    queue.put(root)
    expanded = [False] * len(neighbor_dict)
    while not queue.empty():
        src = queue.get()
        topological_order.append(src)
        expanded[src] = True
        for tgt in neighbor_dict[src]:
            if not expanded[tgt]:
                queue.put(tgt)
    return topological_order

def get_parents(neighbor_dict, root = 0):
    """
    Find the parents of each node in the tree.
    :param neighbor_dict: dict (key:int, val: set); the dictionary saving neighbor nodes.
    :param root: int; the root node index.
    :return parents: list of int; the parent indices of each node in the tree.
    """
    parents = [None] * len(neighbor_dict)
    parents[root] = -1
    queue = Queue()
    queue.put(root)
    while not queue.empty():
        src = queue.get()
        for tgt in neighbor_dict[src]:
            if parents[tgt] is None:
                parents[tgt] = src
                queue.put(tgt)
    return parents

def get_trunk(parents, leaf, root=0):
    """
    Get the trunk of the tree from leaf to root.
    :param parents: list of int; the parent indices of each node in the tree.
    :param leaf: int; the leaf node.
    :param root: int; the root node.
    :return trunk: set of tuple of int; the set of trunk edges.
    """
    trunk = set([])
    tgt = leaf
    while tgt != root:
        trunk.add((tgt, parents[tgt]))
        tgt = parents[tgt]
    return trunk

def make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos, make_directed=True, prune_augmented=True):
    """
    Make the dataset edge connections directed and augment the dataset by random pruning.
    Note that this function assumes the input coming from graphs in same topology and same node ordering.
    :param X_edges: np.ndarray (n_edges, 2); the edge connection of the graph.
    :param X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
    :param X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
    :param Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    :param make_directed: bool; whether augment data by making it directed. If this is set to False, the
        graph is constructed as a undirected graph.
    :param prune_augmented: bool; whether augment the data by random pruning.
    :return:
        new_X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the augmented graph.
        new_X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
        new_X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
        new_Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    """
    num_graphs = len(X_force)
    # Construct a neighboring dictionary
    neighbor_dict = {}
    for src, tgt in X_edges:
        if src not in neighbor_dict:
            neighbor_dict[src] = set([tgt])
        else:
            neighbor_dict[src].add(tgt)
        if tgt not in neighbor_dict:
            neighbor_dict[tgt] = set([src])
        else:
            neighbor_dict[tgt].add(src)

    # Topologically sort the indices
    topological_order = get_topological_order(neighbor_dict)
    # Find the parents to the nodes
    parents = get_parents(neighbor_dict)

    # Data augmentation
    new_X_edges = []
    new_X_force = []
    new_X_pos = []
    new_Y_pos = []
    for i in range(num_graphs):
        # Find the node that force is applied on
        force_index = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        # Only keep the edges from force_index to the root
        trunk = get_trunk(parents, force_index)
        # Find leaf of the trunk
        trunk_nodes = set([])
        for edge in trunk:
            trunk_nodes.add(edge[0])
            trunk_nodes.add(edge[1])
        leaf_nodes = set([])
        for node in trunk_nodes:
            for child in neighbor_dict[node]:
                if child not in trunk_nodes:
                    leaf_nodes.add(child)
        # Add the directed/undirected graph without pruning
        if make_directed:
            edges = []
            for tgt in range(1, len(neighbor_dict)):
                src = parents[tgt]
                if src == 0:
                    edges.append([src, tgt])
                elif (tgt, src) in trunk:
                    edges.append([tgt, src])
                    edges.append([src, tgt])
                else:
                    edges.append([src, tgt])
            new_X_edges.append(np.array(edges))
        else:
            edges = []
            for tgt in range(1, len(neighbor_dict)):
                src = parents[tgt]
                edges.append([tgt, src])
                edges.append([src, tgt])
            new_X_edges.append(np.array(edges))
        new_X_force.append(X_force[i])
        new_X_pos.append(X_pos[i])
        new_Y_pos.append(Y_pos[i])
        # Add the graph with pruning
        if prune_augmented:
            for edge_size in range(len(trunk), len(neighbor_dict) - 1):
                # Get new tree edges
                add_size = edge_size - len(trunk)
                nodes = copy.copy(trunk_nodes)
                node_candidates = copy.copy(leaf_nodes)
                for _ in range(add_size):
                    new_node = random.sample(node_candidates, 1)[0]
                    nodes.add(new_node)
                    node_candidates.remove(new_node)
                    for child in neighbor_dict[new_node]:
                        if child not in nodes:
                            node_candidates.add(child)
                # Re-indexing while keeping root
                reindex_mapping = list(nodes)
                random.shuffle(reindex_mapping)
                root_index = reindex_mapping.index(0)
                reindex_mapping[root_index] = reindex_mapping[0]
                reindex_mapping[0] = 0
                inverse_mapping = [-1] * len(neighbor_dict)
                for new_idx, old_idx in enumerate(reindex_mapping):
                    inverse_mapping[old_idx] = new_idx
                # Add edges to the dataset
                if make_directed:
                    edges = []
                    for tgt in nodes:
                        src = parents[tgt]
                        new_src = inverse_mapping[src]
                        new_tgt = inverse_mapping[tgt]
                        if src == -1:
                            pass
                        elif src == 0:
                            edges.append([new_src, new_tgt])
                        elif (tgt, src) in trunk:
                            edges.append([new_tgt, new_src])
                            edges.append([new_src, new_tgt])
                        else:
                            edges.append([new_src, new_tgt])
                    new_X_edges.append(np.array(edges))
                else:
                    edges = []
                    for tgt in nodes:
                        src = parents[tgt]
                        new_src = inverse_mapping[src]
                        new_tgt = inverse_mapping[tgt]
                        if src == -1:
                            pass
                        else:
                            edges.append([new_tgt, new_src])
                            edges.append([new_src, new_tgt])
                    new_X_edges.append(np.array(edges))
                # Add to the dataset
                new_X_force.append(X_force[i][reindex_mapping])
                new_X_pos.append(X_pos[i][reindex_mapping])
                new_Y_pos.append(Y_pos[i][reindex_mapping])
    return new_X_edges, new_X_force, new_X_pos, new_Y_pos

def rotate_augment(X_edges, X_force, X_pos, Y_pos, rotate_augment_factor=5, stddev_x_angle=0.2, stddev_y_angle=0.2):
    """
    Augment the graph by random rotation.
    :param X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the graph.
    :param X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
    :param X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
    :param Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    :param rotate_augment_factor: int; number of random rotation per graph.
    :param stddev_x_angle: float; the stddev of random rotation in x direction.
    :param stddev_y_angle: float; the stddev of random rotation in y direction.
    :return:
        new_X_edges: np.ndarray (n_graphs, n_edges, 2); the edge connection of the augmented graph.
        new_X_force: np.ndarray (n_graphs, n_nodes, 3); the force applied on the graph.
        new_X_pos: np.ndarray (n_graphs, n_nodes, 3); the initial pose of the graph.
        new_Y_pos: np.ndarray (n_graphs, n_nodes, 3); the end pose of the graph.
    """
    num_graphs = len(X_force)
    # Augment the data by rotation
    new_X_edges = []
    new_X_force = []
    new_X_pos = []
    new_Y_pos = []
    for i in range(num_graphs):
        X_edge = X_edges[i]
        for _ in range(rotate_augment_factor):
            theta_x = np.random.normal(0., stddev_x_angle)
            theta_y = np.random.normal(0., stddev_y_angle)
            theta_z = np.random.uniform(0., 2. * np.pi)
            R = Rotation.from_euler('zyx', [theta_z, theta_y, theta_x]).as_matrix()
            
            X_R = Rotation.from_quat(X_pos[i][:,3:]).as_matrix()
            Y_R = Rotation.from_quat(X_pos[i][:,3:]).as_matrix()
            
            X_R = R@X_R
            Y_R = R@Y_R
            
            X_q = Rotation.from_matrix(X_R).as_quat()
            Y_q = Rotation.from_matrix(Y_R).as_quat()
            
            X_pos_quat = np.concatenate((np.dot(R, X_pos[i][:,:3].T).T, X_q), axis=1)
            Y_pos_quat = np.concatenate((np.dot(R, Y_pos[i][:,:3].T).T, Y_q), axis=1)
            
            new_X_edges.append(X_edge)
            new_X_force.append(np.dot(R, X_force[i].T).T)
            
            
            new_X_pos.append(X_pos_quat)
            new_Y_pos.append(Y_pos_quat)
            
    return new_X_edges, new_X_force, new_X_pos, new_Y_pos


def load_npy(data_dir, sim=True):
    if sim:
        # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
        #X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp.npy'))
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'))
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'))
        # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, n_features)
        X_pos = X_pos[:, :7, :].transpose((0,2,1))
        Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
        X_force = X_force.transpose((0,2,1))
    else:
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'), allow_pickle=True)
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'), allow_pickle=True)

        invalid_graphs = []
        for i, graph in enumerate(X_pos):
            for j, node in enumerate(graph):
                if node is None:
                    invalid_graphs.append(i)

        X_force = np.delete(X_force, invalid_graphs, axis=0)
        X_pos = np.delete(X_pos, invalid_graphs, axis=0)
        Y_pos = np.delete(Y_pos, invalid_graphs, axis=0)
        
        X_pos_list = []
        for graph in X_pos:
            for node in graph:
                for feature in node:
                    X_pos_list.append(feature)
        X_pos = np.array(X_pos_list)
        X_pos = X_pos.reshape(Y_pos.shape[0],Y_pos.shape[1],Y_pos.shape[2]) 
        X_force = X_force.transpose((0,2,1))
    return X_edges, X_force, X_pos, Y_pos

def _make_dataset(X_edges, X_force, X_pos, Y_pos, 
                 make_directed=True, prune_augmented=False, rotate_augmented=False):
    num_graphs = len(X_pos)
    X_edges, X_force, X_pos, Y_pos = make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos,
                                                                     make_directed=make_directed, 
                                                                     prune_augmented=prune_augmented)
    if rotate_augmented:
        X_edges, X_force, X_pos, Y_pos = rotate_augment(X_edges, X_force, X_pos, Y_pos)

    num_graphs = len(X_pos)
    dataset = []
    for i in range(num_graphs): 
        # Combine all node features: [position, force, stiffness] with shape (num_nodes, xyz(3)+force(3)+stiffness_damping(4)) 
        # stiffness damping is (4) because of bending stiffness/damping and torsional stiffness/damping
        root_feature = np.zeros((len(X_pos[i]), 1))
        #root_feature[0, 0] = 1.0
        #X_data = np.concatenate((X_pos[i], X_force[i], root_feature), axis=1) # TODO: Add stiffness damping features later
        X_data = np.concatenate((X_pos[i], X_force[i]), axis=1) # TODO: Add stiffness damping features later

        edge_index = torch.tensor(X_edges[i].T, dtype=torch.long)
        x = torch.tensor(X_data, dtype=torch.float)
        y = torch.tensor(Y_pos[i], dtype=torch.float)
        force_node = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        graph_instance = Data(x=x, edge_index=edge_index, y=y, force_node=force_node)
        dataset.append(graph_instance)
    return dataset

def shuffle_in_unison(a,b,c):
    assert len(a)==len(b)==len(c)
    order = np.arange(len(a))
    np.random.shuffle(order)
    return a[order],b[order],c[order]

def make_dataset(X_edges, X_force, X_pos, Y_pos, tree_pts,
                 make_directed=True, prune_augmented=False, rotate_augmented=False):
    num_graphs = len(X_pos)
    X_edges, X_force, X_pos, Y_pos = make_directed_and_prune_augment(X_edges, X_force, X_pos, Y_pos,
                                                                     make_directed=make_directed, 
                                                                     prune_augmented=prune_augmented)
    if rotate_augmented:
        X_edges, X_force, X_pos, Y_pos = rotate_augment(X_edges, X_force, X_pos, Y_pos)
    
    num_graphs = len(X_pos)
    dataset = []
    for i in range(num_graphs): 
        # Normalize tree by making root node [0,0,0]
        X_pos[i][:,:3] = X_pos[i][:,:3] - X_pos[i][0,:3] 
        Y_pos[i][:,:3] = Y_pos[i][:,:3] - Y_pos[i][0,:3] 

        # node-level features: position, force
        node_features = np.concatenate((X_pos[i][:,:3], X_force[i]), axis=1)
        
        # edge-level features: displacement, distance
        edge_features = []
        for edge in X_edges[i]:
            displacement = X_pos[i][edge[1],:3] - X_pos[i][edge[0],:3]
            distance = np.linalg.norm(displacement)
            edge_features.append(np.concatenate((displacement, [distance]))) 
        edge_features = np.asarray(edge_features)
        # ground truth: final position
        final_positions = Y_pos[i][:,:3]

        # Combine all node features: [position, force, stiffness] with shape (num_nodes, xyz(3)+force(3)+stiffness_damping(4)) 
        # stiffness damping is (4) because of bending stiffness/damping and torsional stiffness/damping
        #root_feature = np.zeros((len(X_pos[i]), 1))
        #root_feature[0, 0] = 1.0
        #X_data = np.concatenate((X_pos[i], X_force[i], root_feature), axis=1) # TODO: Add stiffness damping features later
        #X_data = np.concatenate((X_pos[i], X_force[i]), axis=1) # TODO: Add stiffness damping features later

        edge_index = torch.tensor(X_edges[i].T, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) 
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(final_positions, dtype=torch.float)
        #force_node = np.argwhere(np.sum(np.abs(X_force[i]), axis=1))[0,0]
        graph_instance = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr, tree_pts=tree_pts)
        dataset.append(graph_instance)
    return dataset
