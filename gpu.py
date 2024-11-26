
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader 
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import BatchNorm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_features", type=int, default=20)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

import random
import numpy as np

#seed = 42
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)  # If using CUDA

print(args.num_features, args.device, args.batch_size, args.verbose)

device = torch.device(args.device)

print(f"Using device: {device}")

num_epochs=100000
velicina_batcha=args.batch_size


rucne_znacajke_dim = 2
emb_dim = args.num_features


def parse_coverage_file(file_path):
    with open(file_path, 'r') as file:
        elements=[]
        while True:
            # Read the first line to get n and e
            line = file.readline().strip()
            if not line:  # End of file
                break
            
            # Parse n and e
            aug_edges_size, solution = map(int, line.split())
            
            # Read the next n lines
            lines = [[aug_edges_size,solution]]
            for _ in range(aug_edges_size):
                line = file.readline().strip()
                if line:  # If line is not empty
                    numbers = list(map(int, line.split()))
                    lines.append(numbers)

            elements.append(lines)
            
    return elements

def cover_work(data, n, weights_redundant , probs_double ):

    weights = weights_redundant[(n-1)*2:]
    
    probs=[]
    for i in range(0,len(probs_double),2):
        probs.append( [probs_double[i],i] )
    
    
    probs.sort()
    probs.reverse()
    
    aug_edges_size=data[0][0]
    real_solution=data[0][1]

    covered=[False]*(n-1)

    ML_solution=0
    
    for prob_index in probs:#prob_index[0] nije bitan jer bitno sam koja je najveca vjer ne kakva je
        ind=prob_index[1]
        w=int(weights[ind])
        ok=False
        #print(data[ind//2+1])
        for el in data[ind//2+1]:#+1 jer prvi data je aug_edges i real_solution
            #print(str(el) + " " + str(n))
            if not covered[el]:
                ok=True
                covered[el]=True
        if ok:
            #print(covered)
            ML_solution+=w

    #print(weights)
    print( "rjesenja " + str(real_solution) + " " + str(ML_solution) )

    
# Parsing function
def parse_graph_file(filename):
    data_list = []  # List to store each graph's Data object
    with open(filename, 'r') as file:
        lines = file.readlines()
    line_idx = 0  # Initialize the line index

    while line_idx < len(lines):
        # Parse number of nodes and edges
        n, e = map(int, lines[line_idx].split())
        line_idx += 1

        # Parse edges
        ei1 = list(map(int, lines[line_idx].split()))
        line_idx += 1
        ei2 = list(map(int, lines[line_idx].split()))
        line_idx += 1
        edge_index = torch.tensor([ei1, ei2], dtype=torch.long)

        # Parse edge weights
        edge_weights = list(map(int, lines[line_idx].split()))
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        line_idx += 1

        # Parse chosen labels
        chosen = list(map(int, lines[line_idx].split()))
        y = torch.tensor(chosen, dtype=torch.long)
        line_idx += 1

        # Parse node features
        node_features = []
        for i in range(n):  # Read n lines of node features
            features = list(map(int, lines[line_idx].split()))
            assert len(features) == rucne_znacajke_dim  # Ensure correct feature dimension
            node_features.append(features)
            line_idx += 1

        # Convert node features to tensor and add random features
        x = torch.tensor(node_features, dtype=torch.float)
        random_features = torch.randn((n, emb_dim - rucne_znacajke_dim), dtype=torch.float)
        x = torch.cat((x, random_features), dim=1)

        # Create a Data object and add it to the data list
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
        data_list.append(data)

    return data_list  # Return the list of Data objects



class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout=0.0):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Initial GCN layer (input dimension to hidden dimension)
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers (Hidden dimension to hidden dimension)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # Output layer (Hidden dimension to output dimension)
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns.append(BatchNorm(output_dim))

        # Final linear layer for the output
        self.final = nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr  # Edge weights are optional in GCN

        # Apply GCN layers with Batch Normalization, LeakyReLU, and dropout
        for bn, conv in zip(self.bns[:-1], self.convs[:-1]):
            x = conv(x, edge_index)  # GCNConv
            x = F.leaky_relu(x, negative_slope=0.2)  # LeakyReLU activation
            x = bn(x)  # Batch normalization
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # Final GCN layer (no activation or batch norm here)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.final(x)
        return x


    
# samo vrati vjerojatnosti
def vjerojatnosti_bridova(output_features, edge_index):
    node_a_features = output_features[edge_index[0]]#node_a_features = output_features[edge_index[0][2*(n-1):]]
    node_b_features = output_features[edge_index[1]]
    
    # Compute logits (dot product of node features)
    logits = (node_a_features * node_b_features).sum(dim=1)
    
    return torch.sigmoid(logits)# Convert logits to probabilities (between 0 and 1)

def custom_loss(output_features, edge_index, target_values):
    probs=vjerojatnosti_bridova(output_features, edge_index)
    return F.binary_cross_entropy(probs, target_values.float(), reduction="mean")

# Training function
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    output_features = model(data) 
    loss = custom_loss(output_features, data.edge_index, data.y, data.x.size(0) )
    loss.backward()
    optimizer.step() 
    return loss.item()


def create_support_list(data_list):

    support_list=[]

    for data in data_list:
        n=data.x.size(0)

        # Create the new Data object
        new_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, 2*(n-1):],  # Slice edge_index to remove the first `n` edges
            edge_attr=data.edge_attr[2*(n-1):],  # Slice edge_attr correspondingly
            y=data.y
        )
        support_list.append(new_data)
    return support_list

def save_model(model, loss, filename='model.pth'):
    # Save model weights, loss, and epoch information
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, filename)
    print(f"Model and training state saved to {filename}")

# Load model weights and loss (or other training states)
def load_model(model, filename='model.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    print(f"Model loaded from {filename}")
    return loss  # Return loss and epoch if you need them




# Load data and model setup
train_list = parse_graph_file('data4python.txt')
support_list=create_support_list(train_list)

train_loader = DataLoader(train_list, batch_size=velicina_batcha, shuffle=False)
support_loader = DataLoader(support_list, batch_size=velicina_batcha, shuffle=False)

validation_list = parse_graph_file('validation.txt')
coverage_list = parse_coverage_file("coverage.txt")

input_dim = emb_dim
hidden_dim = 512
output_dim = 512

model = GCN(input_dim, hidden_dim, output_dim, num_layers=3)


import os

file_path = 'model.pth'

if os.path.exists(file_path):
    mali_los=load_model(model)
    print(mali_los)
else:
    mali_los=50

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

"""
# Print the weights of the model
for name, param in model.named_parameters():
    if 'weight' in name:  # Filter only weight parameters
        print(f"Weights of {name}: {param.data}")
    elif 'bias' in name:  # Filter only bias parameters
        print(f"Biases of {name}: {param.data}")
"""

# Initial evaluation (unchanged)
model.eval() 
with torch.no_grad():
    suma = 0
    for data,supp in zip(train_list,support_list):
        data.to(device)
        supp.to(device)
        initial_output_features = model(data) 
        initial_loss = custom_loss(initial_output_features, supp.edge_index, data.y)
        suma += initial_loss.item()
    print(f'Initial Average Loss for a graph before training: {suma / len(train_list)}')


# Training loop with batches
for epoch in range(num_epochs):
    total_loss = 0
    model.train()  # Set model to training mode
    for batch, batch_support in zip(train_loader,support_loader):  # Iterate over batches
        batch.to(device)
        batch_support.to(device)
        optimizer.zero_grad()
        output_features = model(batch)  # Forward pass with batch
        loss = custom_loss(output_features, batch_support.edge_index, batch.y)  # Compute loss
        
        loss.backward()

        print(output_features)

        
        for name, param in model.named_parameters():
            if param.grad is not None:  # Ensure the parameter has a gradient
                print(f"Gradient for {name}: {param.grad}")  # Print the gradient
        
        exit(0)
        
        optimizer.step()
        total_loss += loss.item()

    mean_gubitak=total_loss / len(train_loader) 
    # Log average training loss
    if epoch % 10 == 0 or mean_gubitak < mali_los:
        print(f'Epoch {epoch + 1}')
        
        print(f'Average Train Loss: {total_loss / len(train_loader)}')

        if mean_gubitak < mali_los:
            mali_los=mean_gubitak
            save_model(model, mali_los)
        
        # Validation loop (unchanged)
        model.eval()
        with torch.no_grad():
            suma = 0
            for data, cover_data in zip(validation_list, coverage_list):
                data.to(device)
                skraceni_ei=data.edge_index[:, 2*(data.x.size(0)-1):]
                probs = vjerojatnosti_bridova(model(data), skraceni_ei)
                cover_work(cover_data, data.x.size(0), data.edge_attr, probs.tolist())
                initial_output_features = model(data)
                initial_loss = custom_loss(initial_output_features, skraceni_ei, data.y)
                suma += initial_loss.item()
            print(f'Average Validation Loss: {suma / len(validation_list)}')


