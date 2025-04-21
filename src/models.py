import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, APPNP, SGConv, ChebConv, GraphConv, BatchNorm


class GNNModel(torch.nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.3, K=2):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout
        
        if self.model_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.model_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, concat=True))
        elif self.model_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif self.model_type == "gin":
            mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
        elif self.model_type == "appnp":
            self.propagation = APPNP(K, alpha=0.1)
        elif self.model_type == "sgc":
            self.convs.append(SGConv(input_dim, output_dim, K=K))
        elif self.model_type == "cheb":
            self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        elif self.model_type == "graphconv":
            self.convs.append(GraphConv(input_dim, hidden_dim))
        else:
            raise Exception("Model " + model_type + " ne postoji." )
        
        if self.model_type not in ["appnp", "sgc"]:
            self.bns.append(BatchNorm(hidden_dim))
            for _ in range(num_layers - 2):
                if self.model_type == "gcn":
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif self.model_type == "gat":
                    self.convs.append(GATConv(hidden_dim, hidden_dim, concat=True))
                elif self.model_type == "sage":
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                elif self.model_type == "gin":
                    mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    self.convs.append(GINConv(mlp))
                elif self.model_type == "cheb":
                    self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
                elif self.model_type == "graphconv":
                    self.convs.append(GraphConv(hidden_dim, hidden_dim))
                self.bns.append(BatchNorm(hidden_dim))
            
            if self.model_type in ["gcn", "gat", "sage", "gin", "cheb", "graphconv"]:
                self.convs.append(GCNConv(hidden_dim, output_dim) if self.model_type == "gcn" else 
                                  GATConv(hidden_dim, output_dim, concat=False) if self.model_type == "gat" else 
                                  SAGEConv(hidden_dim, output_dim) if self.model_type == "sage" else 
                                  GINConv(nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))) if self.model_type == "gin" else 
                                  ChebConv(hidden_dim, output_dim, K=K) if self.model_type == "cheb" else
                                  GraphConv(hidden_dim, output_dim))
                self.bns.append(BatchNorm(output_dim))

        self.final = nn.Linear(output_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.model_type == "appnp":
            x = self.propagation(x, edge_index)
        else:
            for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                
            x = self.convs[-1](x, edge_index)
            x = self.bns[-1](x)
        
        x = self.final(x)
        return x




"""
#bez dropouta
class GNNModel(torch.nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, num_layers=4, K=2):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Initializing the first convolution layer based on model type
        if self.model_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.model_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, concat=True))
        elif self.model_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif self.model_type == "gin":
            mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
        elif self.model_type == "appnp":
            self.propagation = APPNP(K, alpha=0.1)
        elif self.model_type == "sgc":
            self.convs.append(SGConv(input_dim, output_dim, K=K))
        elif self.model_type == "cheb":
            self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        elif self.model_type == "graphconv":
            self.convs.append(GraphConv(input_dim, hidden_dim))
        else:
            raise Exception("Model " + model_type + " does not exist.")

        # For models other than APPNP and SGC, add layers and batch normalization
        if self.model_type not in ["appnp", "sgc"]:
            self.bns.append(BatchNorm(hidden_dim))  # BatchNorm for the first hidden layer
            for _ in range(num_layers - 2):
                if self.model_type == "gcn":
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif self.model_type == "gat":
                    self.convs.append(GATConv(hidden_dim, hidden_dim, concat=True))
                elif self.model_type == "sage":
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                elif self.model_type == "gin":
                    mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    self.convs.append(GINConv(mlp))
                elif self.model_type == "cheb":
                    self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
                elif self.model_type == "graphconv":
                    self.convs.append(GraphConv(hidden_dim, hidden_dim))
                self.bns.append(BatchNorm(hidden_dim))  # BatchNorm for each hidden layer
            
            if self.model_type in ["gcn", "gat", "sage", "gin", "cheb", "graphconv"]:
                self.convs.append(GCNConv(hidden_dim, output_dim) if self.model_type == "gcn" else 
                                  GATConv(hidden_dim, output_dim, concat=False) if self.model_type == "gat" else 
                                  SAGEConv(hidden_dim, output_dim) if self.model_type == "sage" else 
                                  GINConv(nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))) if self.model_type == "gin" else 
                                  ChebConv(hidden_dim, output_dim, K=K) if self.model_type == "cheb" else
                                  GraphConv(hidden_dim, output_dim))
                self.bns.append(BatchNorm(output_dim))  # BatchNorm for the output layer

        self.final = nn.Linear(output_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # If the model is APPNP, apply propagation
        if self.model_type == "appnp":
            x = self.propagation(x, edge_index)
        else:
            # For the other models, apply convolutions and batch normalization
            for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
                x = conv(x, edge_index)
                x = bn(x)  # Apply batch normalization
                x = F.relu(x)  # Apply ReLU activation
                
            # Apply final convolution and batch normalization
            x = self.convs[-1](x, edge_index)
            x = self.bns[-1](x)
        
        # Final output layer
        x = self.final(x)
        return x
    

"""