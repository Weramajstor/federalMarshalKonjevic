import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader 
from torch_geometric.nn import Node2Vec
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



def create_graph_embeddings(num_vertices, num_artificial_features, edge_index, embedding_type):

    if embedding_type == "rand":
        return torch.rand((num_vertices, num_artificial_features), dtype=torch.float)
    
    if embedding_type == "ones":
        return torch.ones((num_vertices, num_artificial_features), dtype=torch.float)
    
    if embedding_type == "zeros":
        return torch.zeros((num_vertices, num_artificial_features), dtype=torch.float)
    
    if embedding_type == "node2vec":

        num_epochs = 30
        learning_rate = 0.03

        model = Node2Vec(
            edge_index,
            embedding_dim=num_artificial_features,  # Set the embedding dimension
            walk_length=10,                         # Set the walk length
            context_size=3,                         # Set the context size
            walks_per_node=10,                      # Set the number of walks per node
            sparse=True,                            # Use sparse matrices for efficiency
            p=0.25,                                 # Set the return parameter (local exploration)
            q=4.0                                   # Set the inout parameter (global exploration)
        )

        optimizer = optim.SparseAdam(list(model.parameters()), lr=learning_rate)

        def train_node2vec():
            model.train()
            for epoch in range(0,num_epochs):
                total_loss = 0
                for pos_rw, neg_rw in model.loader():
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                #print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(model.loader())}')

        train_node2vec()

        # After training, set the model to evaluation mode and return the learned embeddings
        model.eval()
        node_embeddings = model.embedding.weight.data.clone()

        #print(node_embeddings)
        
        return model.embedding.weight.data.clone()

    # Default fallback
    return torch.rand((num_vertices, num_artificial_features), dtype=torch.float)


"""arhitektura fajla embedingova grafova
t dim
n1
emb_1_1 emb_1_2 ... emb_1_dim
...
emb_n1_1 emb_n1_2 ... emb_n1_dim
n2
emb_1_1 emb_1_2 ... emb_1_dim
...
emb_n2_1 emb_n2_2 ... emb_n2_dim
nt
emb_1_1 emb_1_2 ... emb_1_dim
...
emb_nt_1 emb_nt_2 ... emb_nt_dim
"""
def add_artificial_features(embedding_type, graph_filename, graph_list, num_artificial_features):
    embedding_filename = graph_filename + "_embed.txt"

    if embedding_type == "node2vec" and os.path.exists(embedding_filename):
        # Load precomputed embeddings from the file
        print(f"Loading embeddings from {embedding_filename}")
        
        with open(embedding_filename, 'r') as f:
            # First line: t and dim
            t, dim = map(int, f.readline().split())
            
            # Initialize a list to hold embeddings
            embeddings = []

            for _ in range(t):
                # Read number of nodes for the current graph
                n = int(f.readline())
                
                # Read the embeddings for this graph
                graph_embeddings = []
                for _ in range(n):
                    graph_embeddings.append(list(map(float, f.readline().split())))
                
                embeddings.append(torch.tensor(graph_embeddings))

            # Assign the embeddings to each graph
            for idx, graph in enumerate(graph_list):
                graph.x = torch.cat((graph.x, embeddings[idx]), dim=1)

    else:
        # Generate new embeddings for each graph
        cnt=0
        for graph in graph_list:
            cnt+=1
            if cnt%50==0:
                print("embeda se " + str(cnt) + "-i graf")
            new_embeddings = create_graph_embeddings(graph.x.shape[0], num_artificial_features, graph.edge_index, embedding_type)
            graph.x = torch.cat((graph.x, new_embeddings), dim=1)

        if embedding_type == "node2vec" and not os.path.exists(embedding_filename):
            print(f"Saving embeddings to {embedding_filename}")

            with open(embedding_filename, 'w') as f:
                # First, write t (number of graphs) and dim (number of embedding dimensions)
                f.write(f"{len(graph_list)} {num_artificial_features}\n")

                # Save the embeddings for each graph
                for graph in graph_list:
                    n = graph.x.shape[0]  # number of nodes in the current graph
                    f.write(f"{n}\n")  # Write the number of nodes in the graph
                    
                    # Write the node embeddings
                    for i in range(n):
                        f.write(" ".join(map(str, graph.x[i, -num_artificial_features:].tolist())) + "\n")  # Write last `num_features` as the embedding
        



"""
ARHITEKTURA FAJLA koji definira graf:
(e je broj usmjerenih edgeva, svaki T-T par predstavlja dva usmjerena brida kao npr 3-17 , 17-3 )
VAZNO: napominjem ponovo da svaki JEDAN element ovdje u svakom retku se odnosi na dva smjera istog brida
i to mora tak biti zbog efikasne arhitekture machine learninga nad grafovima
(tezine i chosenost su samo za augmenting edgeove)
n e
9 58
ei1 (ei kao edge index)
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
ei2
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
weights
0 0 0 0 0 0 0 0 w w w w w w w w w w w w w w w w w w w w w
chosen(prvih n-1 je u pocetnom stablu odabrano)
0 0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 1 0 0
feature1 feature2 , feature1=broj susjeda, feature2=broj susjeda susjedova
number_11 number_12
number_21 number_22
...
number_n1 number_n2
"""
def parse_graph_file(filename, num_features, embedding_type):

    num_manual_features = 2

    graph_list = []  # List to store each graph's Data object
    with open(filename, 'r') as file:
        lines = file.readlines()
    line_idx = 0  # Initialize the line index

    while line_idx < len(lines):
        # Parse number of nodes and edges
        num_vertices, e = map(int, lines[line_idx].split())
        line_idx += 1

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
        hand_features = []
        for i in range(num_vertices):  # Read n lines of node features
            features = list(map(int, lines[line_idx].split()))
            assert len(features) == num_manual_features  # Ensure correct feature dimension
            hand_features.append(features)
            line_idx += 1
        hand_features = torch.tensor(hand_features, dtype=torch.float)

        loaded_graph = Data(x=hand_features, edge_index=edge_index, edge_attr=edge_weights, y=y)
        graph_list.append(loaded_graph)

        #artifical_features = create_artificial_featues(num_vertices, num_features - num_manual_features, edge_index, args)
        #print(artifical_features)
        #all_features = torch.cat((hand_features, artifical_features), dim=1)
        
    add_artificial_features(embedding_type, filename, graph_list, num_features-num_manual_features)

    return graph_list  # Return the list of Data objects



"""
arhitektura cover filea:
t(broj grafova) puta bude sljedece:
broj_augmenting_edgeova scip_solution_value
edgeovi u stablu pokriveni aue-om 1
edgeovi u stablu pokriveni aue-om 2
edgeovi u stablu pokriveni aue-om 3
.....
edgeovi u stablu pokriveni aue-om broj_augmenting_edgeova
"""
def parse_coverage_file(file_path):
    with open(file_path, 'r') as file:
        elements=[]
        while True:
            
            # read aug_edges_size, solution_value
            line = file.readline().strip()
            if not line:  # End of file
                break
            
            aug_edges_size, solution_value = map(int, line.split())
            
            # Read the next n lines
            lines = [[aug_edges_size, solution_value]]
            for _ in range(aug_edges_size):
                line = file.readline().strip()
                if line:  # If line is not empty
                    numbers = list(map(int, line.split()))
                    lines.append(numbers)

            elements.append(lines)
            
    return elements


def create_augmenting_edges_list(data_list):
    """
    Creates and returns data list without tree edges
    """
    support_list=[]
    for data in data_list:
        n=data.x.size(0)
        # sve ostane isto osim edge indexova
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
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    print(f"Model loaded from {filename}")
    return loss


def cover_work(set_cover_data, n, all_weights , probs_double, verbose=False ):
    """
    Sorts edges descending by probability, choses the ones that cover at least one uncovered 
    edge in tree, while ignores the ones that bring nothing new. Result is printed in the end.

    Args:
        cover_data (list): It tells us which tree edges are covered by each augmenting edge
        n (int): The number of vertices.
        weights_with_tree (list): its called redundant because it has weights of edges already in tree
        probs_double (list): A list of probabilities, with every second element used for sorting.

    Returns:
        (double) optimality gap of the given solution
    """

    agumenting_weights = all_weights[(n-1)*2:]
    
    probs=[]
    for i in range(0,len(probs_double),2):#mogo bi ovo deduplicirat
        probs.append( [probs_double[i],i//2] )#kroz dva jer uzimam svaki drugi pa normaliziram indekse
    
    probs.sort()
    probs.reverse()

    aug_edges_size=set_cover_data[0][0]
    real_solution=set_cover_data[0][1]

    covered=[False]*(n-1)

    ML_solution=0
    
    for prob_index in probs:#prob_index[0] nije bitan jer bitno sam koja je najveca vjer ne kakva je
        ind=prob_index[1]
        w=int(agumenting_weights[ind])
        ok=False
        for el in set_cover_data[ind+1]:#+1 jer prvi data je aug_edges i real_solution
            #print(str(el) + " " + str(n))
            if not covered[el]:
                ok=True
                covered[el]=True
        if ok:
            #print(covered)
            ML_solution+=w

    if verbose:
        print( "rjesenja " + str(real_solution) + " " + str(ML_solution) )
    return (ML_solution/real_solution-1)



def initialize_model_parameters(model, load_model_parameters_from_file):
    import os
    file_path = 'model.pth'
    starting_loss=1e9
    if load_model_parameters_from_file==True:

        if os.path.exists(file_path):
            starting_loss=load_model(model)
            print("Loss of loaded model: " + str(starting_loss))
        else:
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
    return starting_loss


def reproducibility_settings(reproducible):
    import random
    if reproducible:
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using CUDA



def draw_custom_edges(edge_index, num_tree_edges, red_widths):
    red_widths = red_widths[::2]

    data = Data(edge_index=edge_index, num_nodes=num_tree_edges + 1)
    G = to_networkx(data, to_undirected=True)

    # Deduplicate edges: treat (u, v) and (v, u) as the same
    seen_edges = set()
    edges = []
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        edge = tuple(sorted((u, v)))
        if edge not in seen_edges:
            seen_edges.add(edge)
            edges.append(edge)

    assert len(red_widths) == len(edges) - num_tree_edges, "red_widths length mismatch"

    # Create edge labels only for red edges
    red_edges = edges[num_tree_edges:]  # the red ones
    red_labels = {
        edge: f"{width:.2f}" for edge, width in zip(red_edges, red_widths)
    }

    for i in range(len(red_widths)):
        red_widths[i] = red_widths[i] * (red_widths[i]/0.55) ** 4

    edge_colors = ['black'] * num_tree_edges + ['red'] * (len(edges) - num_tree_edges)
    edge_widths = [3] * num_tree_edges + red_widths

    pos = nx.spring_layout(G)
    nx.draw(G, pos,
            edgelist=edges,
            edge_color=edge_colors,
            width=edge_widths,
            node_color='skyblue',
            node_size=900,
            with_labels=True)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=red_labels, font_color='red')
    plt.show()