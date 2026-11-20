import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_classifier(nn.Module):
   def __init__(self, input_channels, hidden_channels, output_channels, n_hidden = 0, dropout = 0.25):
      super().__init__()
      self.conv1 = GCNConv(input_channels, hidden_channels)
      self.hid = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(n_hidden)])
      self.conv2 = GCNConv(hidden_channels, output_channels)
      self.dropout = dropout

   def forward(self, data):
      x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,0]
      x = self.conv1(x, edge_index, edge_weight = edge_weight)
      x = F.relu(x)
      x = F.dropout(x, training = self.training, p = self.dropout)
      for i in range(len(self.hid)):
         x = self.hid[i](x, edge_index, edge_weight = edge_weight)
         x = F.relu(x)
         x = F.dropout(x, training = self.training, p = self.dropout)
      x = self.conv2(x, edge_index, edge_weight = edge_weight)
      x = F.dropout(x, p = self.dropout, training = self.training)
      return F.log_softmax(x, dim = 1)
