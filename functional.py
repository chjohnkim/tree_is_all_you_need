import torch
import visualization

def train(model, optimizer, criterion, train_loader, epoch, device):
    model.train()
    running_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    return train_loss

def validate(model, criterion, val_loader, epoch, device):
    model.eval()
    running_l2_norm = 0
    num_graphs = 0
    for batch in val_loader:
        batch.to(device)
        out = model(batch)
        running_l2_norm += torch.sum(torch.norm(out-batch.y, dim=1)).item()
        num_graphs+=out.size()[0]
    val_loss = running_l2_norm/num_graphs
    return val_loss

def test(model, test_loader, device, visualize=False):
    model.eval()
    running_l2_norm_by_root_dist = {}
    root_dist_accumulator = {}
    running_l2_base_by_root_dist = {}
    l2_norms = []
    l2_norm_base = []
    l2_norm_averaged = []
    l2_norm_base_averaged = []
    running_l2_norm = 0
    running_l2_norm_base = 0
    num_graphs = 0
    idx = 0
    for batch in test_loader:
        root_dist_dict = generate_root_distance_dict(batch.edge_index.detach().cpu().numpy().T)
        batch.to(device)
        with torch.no_grad():
            out = model(batch)
        y = batch.y.detach()
        x = batch.x.detach()
        force = batch.x[:,-3:].detach()

        for i,node_dist in enumerate(torch.norm(out[:,:3]-y[:,:3], dim=1).detach().tolist()):
            root_distance = root_dist_dict[i]
            if root_distance in running_l2_norm_by_root_dist.keys():
                running_l2_norm_by_root_dist[root_distance].append(node_dist)
                root_dist_accumulator[root_distance] += 1
            else:
                running_l2_norm_by_root_dist[root_distance] = [node_dist]
                root_dist_accumulator[root_distance] = 1

        for i,base_dist in enumerate(torch.norm(x[:,:3]-y[:,:3], dim=1).detach().tolist()):
            root_distance = root_dist_dict[i]
            if root_distance in running_l2_base_by_root_dist.keys():
                running_l2_base_by_root_dist[root_distance].append(base_dist)
            else:
                running_l2_base_by_root_dist[root_distance] = [base_dist]

        norm = torch.sum(torch.norm(out[:,:3]-y[:,:3], dim=1))
        base = torch.sum(torch.norm(x[:,:3]-y[:,:3], dim=1))

        l2_norms.append(norm.detach().item())
        l2_norm_base.append(base.detach().item())

        l2_norm_averaged.append(norm.detach().item()/out.size()[0])
        l2_norm_base_averaged.append(base.detach().item()/out.size()[0])

        running_l2_norm += norm
        running_l2_norm_base += base
        num_graphs+=out.size()[0]
        if idx < 10 and visualize:
            visualization.visualize_graph(out[:,:3], 
                            y[:,:3], 
                            x[:,:3], 
                            batch.edge_index, batch.force_node[0], 
                            force, results_path+"prediction_example%s"%idx)
        idx += 1  

    l2_norm_by_node = []
    l2_norm_base_by_node = []
    root_dists = list(root_dist_accumulator.keys())
    root_dists.sort()
    for root_dist in root_dists:
        l2_norm_by_node.append(sum(running_l2_norm_by_root_dist[root_dist])/root_dist_accumulator[root_dist])
        l2_norm_base_by_node.append(sum(running_l2_base_by_root_dist[root_dist])/root_dist_accumulator[root_dist])

    l2_std_dev_by_node = []
    l2_std_dev_base_by_node = []
    for root_dist in root_dists:
        l2_std_dev_by_node.append(calc_variance(l2_norm_by_node[root_dist], running_l2_norm_by_root_dist[root_dist])**(1/2))
        l2_std_dev_base_by_node.append(calc_variance(l2_norm_base_by_node[root_dist], running_l2_base_by_root_dist[root_dist])**(1/2))


    X_axis = np.arange(len(root_dists))

    plt.bar(X_axis - 0.2, l2_norm_by_node, 0.4, label = 'prediction error', yerr=l2_std_dev_by_node)
    plt.bar(X_axis + 0.2, l2_norm_base_by_node, 0.4, label = 'baseline error', yerr=l2_std_dev_base_by_node)

    plt.xticks(X_axis, root_dists)
    plt.xlabel("Distance from root node")
    plt.ylabel("L2 distance")
    plt.title("error by distance to root node")
    plt.legend()

    plt.savefig(results_path+"error_by_node")
    plt.close()

    l2_norm = running_l2_norm/num_graphs
    l2_base = running_l2_norm_base/num_graphs

    l2_std_dev = calc_variance(l2_norm, l2_norms, N=num_graphs)**2
    l2_base_std_dev = calc_variance(l2_base, l2_norm_base, N=num_graphs)**2

    visualization.print_loss_by_tree(l2_norm_base_averaged, l2_norm_averaged)

    print('Mean node distance error: {}'.format(l2_norm))
    print('Mean base node displacement: {}'.format(l2_base))
    print('Variance of node distance error: {}'.format(l2_std_dev))
    print('Variance of base node displacement: {}'.format(l2_base_std_dev))

    return l2_norm, l2_base

def make_gif(model, test_loader, device):
    model.eval()
    running_l2_norm = 0
    num_graphs = 0
    for i, batch in enumerate(test_loader):
        batch.to(device)
        out = model(batch)
        running_l2_norm += torch.sum(torch.norm(out[:,:3]-batch.y[:,:3], dim=1))
        num_graphs+=out.size()[0]
        visualization.make_gif(out[:,:3], 
                        batch.y[:,:3], 
                        batch.x[:,:3], 
                        batch.edge_index, batch.force_node[0], 
                        batch.x[:,-3:], i)
    l2_norm = running_l2_norm/num_graphs
    print('Average node distance error: {}'.format(l2_norm))