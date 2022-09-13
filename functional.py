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
    running_l2_norm = 0
    num_graphs = 0
    for batch in test_loader:
        batch.to(device)
        out = model(batch)
        running_l2_norm += torch.sum(torch.norm(out-batch.y, dim=1)).item()
        num_graphs+=out.size()[0]
        if visualize:
            force_node = torch.argmax(torch.abs(torch.sum(batch.x[:,3:], dim=1))).item()
            visualization.visualize_graph(out[:,:3], 
                            batch.y[:,:3], 
                            batch.x[:,:3], 
                            batch.edge_index, force_node,
                            batch.x[:,-3:])
    l2_norm = running_l2_norm/num_graphs
    print('Average node distance error: {}'.format(l2_norm))

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