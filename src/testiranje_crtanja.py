from data_utils import *
from model_logic import *
import argparse
from models import *
import random

parser = argparse.ArgumentParser()
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--load_model_parameters_from_file", type=bool, default=True)
parser.add_argument("--model_type", type=str, default="sage")
parser.add_argument("--embedding_type", type=str, default="zeros")
parser.add_argument("--reproducible", type=bool, default=True)
args = parser.parse_args()

reproducibility_settings(args.reproducible)

input_dim = args.num_features
hidden_dim = 128
output_dim = 64
model = GNNModel( args.model_type, input_dim, hidden_dim, output_dim, num_layers=5)
a=initialize_model_parameters(model, args.load_model_parameters_from_file)
print("optimalitygap " + str(a))
model.eval()#train()#very important that model is in evaluation mode to prevent dropouts and such when inferring


validation_list = parse_graph_file('validation.txt', args.num_features, args.embedding_type)
agumenting_edges_list = create_augmenting_edges_list(validation_list)
coverage_list = parse_coverage_file("coverage.txt")

#for i in range(len(validation_list)):
 #   rand_val_index=i
while True:
    rand_val_index = random.randrange(len(validation_list))
    
    Graph = validation_list[rand_val_index]
    Aug_Graph = agumenting_edges_list[rand_val_index]

    probs = augmenting_edges_probabilities(model, Graph)
    cover_work(coverage_list[rand_val_index], Graph.x.size(0), Graph.edge_attr, probs.tolist(), verbose=True)
    draw_custom_edges(Graph.edge_index, num_tree_edges=Graph.x.size(0)-1, red_widths=probs.tolist())

