
import torch
from torch_geometric.loader import DataLoader
from data_utils import *
from model_logic import *
import wandb
import argparse
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--load_model_parameters_from_file", type=bool, default=False)
parser.add_argument("--wandb_log", type=bool, default=True)
parser.add_argument("--model_type", type=str, default="sage")
parser.add_argument("--embedding_type", type=str, default="zeros")
parser.add_argument("--loss_type", type=str, default="unsupervised")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--reproducible", type=bool, default=False)
args = parser.parse_args()

if args.wandb_log:
    wandb.init(project="e1-aug", name="zeros")  # change project/name_of_run as needed
device = torch.device(args.device)
print(f"Using device: {device}")
reproducibility_settings(args.reproducible)

train_list = parse_graph_file('data4python.txt', args.num_features, args.embedding_type)
agumenting_edges_list = create_augmenting_edges_list(train_list)
train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=False)
augmenting_edges_loader = DataLoader(agumenting_edges_list, batch_size=args.batch_size, shuffle=False)

validation_list = parse_graph_file('validation.txt', args.num_features, args.embedding_type)
coverage_list = parse_coverage_file("coverage.txt")

input_dim = args.num_features
hidden_dim = 128
output_dim = 64
model = GNNModel( args.model_type, input_dim, hidden_dim, output_dim, num_layers=5)

lowest_optimality_gap = initialize_model_parameters(model, args.load_model_parameters_from_file)#this sets model parameters
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


for epoch in range(args.num_epochs):
    local_train_loss = 0
    model.train()
    for batch_train, batch_augmenting_edges, set_cover_data in zip(train_loader,augmenting_edges_loader, coverage_list):
        batch_train = batch_train.to(device)
        batch_augmenting_edges = batch_augmenting_edges.to(device)
        optimizer.zero_grad()
        
        if args.loss_type =="supervised":
            loss = supervised_loss( model(batch_train), batch_augmenting_edges.edge_index, batch_train.y)
        else:
            loss = unsupervised_erdos_loss( model(batch_train), batch_augmenting_edges.edge_index, \
                                           batch_augmenting_edges.edge_attr, set_cover_data, verbose=True )
        
        loss.backward()
        optimizer.step()
        local_train_loss += loss.item()/len(train_list)

        if epoch%200==5:#ovo mi daje probsove za samo jedan graf
            probs = augmenting_edges_probabilities(model, batch_train.to_data_list()[0])
            print(" vjerojatnosti bridova: ")
            print(probs.round(decimals=2))
            print(batch_train.to_data_list()[0].y)
    
    #logging and validation
    if epoch % 1 == 0:
        print(f'Epoch {epoch + 1}')
        print(f'Average Train Loss: {local_train_loss}')
        """
        give me loss on validation set
        and compare ML solution with optimal solution
        """
        model.eval()
        with torch.no_grad():
            perfect_ratio = 0
            val_loss = 0
            local_optimality_gap = 0
            for val_graph, set_cover_data in zip(validation_list, coverage_list):
                val_graph = val_graph.to(device)
                if args.loss_type == "supervised":
                    loss = supervised_loss( model(val_graph), val_graph.edge_index[:, 2*(val_graph.x.size(0)-1):], val_graph.y)
                else:
                    loss = unsupervised_erdos_loss( model(val_graph), val_graph.edge_index[:, 2*(val_graph.x.size(0)-1):],\
                                                    val_graph.edge_attr[2*(val_graph.x.size(0)-1):], set_cover_data)
                val_loss += loss.item() / len( validation_list )
                
                probs = augmenting_edges_probabilities(model, val_graph)
                example_gap = cover_work(set_cover_data, val_graph.x.size(0), val_graph.edge_attr, probs.tolist()) / len(validation_list)
                local_optimality_gap += example_gap
                if example_gap==0:
                    perfect_ratio += 1/len(validation_list)

            """
            if lowest_optimality_gap > local_optimality_gap:
                lowest_optimality_gap = local_optimality_gap
                save_model(model, lowest_optimality_gap)
                #print me new solutions over validaton set
                for val_graph, set_cover_data in zip(validation_list, coverage_list):#validation_list
                    probs = augmenting_edges_probabilities(model, val_graph)
                    cover_work(set_cover_data, val_graph.x.size(0), val_graph.edge_attr, probs.tolist(), verbose=True)   
            """

            if lowest_optimality_gap > local_train_loss:
                lowest_optimality_gap = local_train_loss
                save_model(model, lowest_optimality_gap)
                #print me new solutions over validaton set
                for val_graph, set_cover_data in zip(validation_list, coverage_list):#validation_list
                    probs = augmenting_edges_probabilities(model, val_graph)
                    cover_work(set_cover_data, val_graph.x.size(0), val_graph.edge_attr, probs.tolist(), verbose=True)      

            if args.wandb_log:
                wandb.log({
                    "Train Loss": local_train_loss,
                    "Optimality gap": local_optimality_gap,
                    "Validation Loss": val_loss,
                    "Perfect ratio": perfect_ratio
                })
            print(f'Average Validation Loss: { val_loss }')
            

