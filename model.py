import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gcn_conv = GCNConv(hidden_size, hidden_size) # (hidden, num_out_features_per_node)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
  
    def forward(self, x, edge_index):
        x_block = self.gcn_conv(x, edge_index)
        x_block = F.relu(x_block)
        x_block = self.linear(x_block)
        x_block = F.relu(x_block)
        return x_block+x


# Residual connection
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1280
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_size),
            torch.nn.ReLU(),
        )

        self.block1 = GCNResidualBlock(hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.do1 = torch.nn.Dropout(p)
        
        self.block2 = GCNResidualBlock(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.do2 = torch.nn.Dropout(p)
        
        self.block3 = GCNResidualBlock(hidden_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.do3 = torch.nn.Dropout(p)

        self.block4 = GCNResidualBlock(hidden_size)
        self.bn4 = torch.nn.BatchNorm1d(hidden_size)
        self.do4 = torch.nn.Dropout(p)

        self.block5 = GCNResidualBlock(hidden_size)
        self.bn5 = torch.nn.BatchNorm1d(hidden_size)
        self.do5 = torch.nn.Dropout(p)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 7)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.stem(x)
        x = self.block1(x, edge_index)
        x = self.bn1(x)
        x = self.do1(x)
        
        x = self.block2(x, edge_index)
        x = self.bn2(x)
        x = self.do2(x)
        
        x = self.block3(x, edge_index)
        x = self.bn3(x)
        x = self.do3(x)
        
        x = self.block4(x, edge_index)
        x = self.bn4(x)
        x = self.do4(x)
        
        x = self.block5(x, edge_index)
        x = self.bn5(x)
        x = self.do5(x)
                
        x = self.out(x)        
        return x