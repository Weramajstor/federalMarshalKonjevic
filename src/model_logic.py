import torch.nn.functional as F
import torch

def edge_logits(node_features, augmenting_edge_index):
    """
    output_features ima dimenzije output_dim koju zadajemo za svaki cvor
    gives logits of edges given by model
    je li ono sto ide u sigmoidu na kraju logit? moram istrazit malo stvari
    """

    """
    vraca parove pripadnih cvorova edgeova koji nisu u stablu te ih kasnije
    element-wise multiplicira
    """
    node_a_features = node_features[augmenting_edge_index[0]]#isto kao i node_a_features = output_features[edge_index[0][2*(n-1):]]
    node_b_features = node_features[augmenting_edge_index[1]]

    """
    ovo daje logite svih non chosen edgeova
    zelimo poboljsat samo non chosen edgeova zato jer je to lakse nego da
    ucimo i chosen edgeove
    """
    logits = (node_a_features * node_b_features).sum(dim=1)

    #return torch.sigmoid(logits)
    return logits
global ok
ok=0
def supervised_loss(node_features, augmenting_edge_index, target_values):
    logits=edge_logits(node_features, augmenting_edge_index)
    #ovo je mozda numericki stabilnije nego da guram .probs odmah
    #kak su isti na pocetku ne kuzim?
    """
    global ok
    ok+=1
    if ok>2000:
        print("usporedbe")
        print(logits)
        print(torch.round( torch.sigmoid(logits)*100 )/100)
        print(target_values.float())
        print(augmenting_edge_index)
        exit(0)
    """
    return F.binary_cross_entropy_with_logits(logits, target_values.float(), reduction="sum")


def augmenting_edges_probabilities(model, data):
    augmenting_ei = data.edge_index[:, 2*(data.x.size(0)-1):]#znaci bez onih vec u stablu
    logiti = edge_logits(model(data), augmenting_ei)
    probs = torch.sigmoid( logiti )
    return probs

"""
arhitektura cover instance:
broj_augmenting_edgeova scip_solution_value
edgeovi u stablu pokriveni aue-om 1
edgeovi u stablu pokriveni aue-om 2
edgeovi u stablu pokriveni aue-om 3
.....
edgeovi u stablu pokriveni aue-om broj_augmenting_edgeova
"""
def bridge_number_expectancy(probs, set_cover_data, node_number):
    not_found_probs = torch.ones(node_number - 1, dtype=torch.float32, device=probs.device)
    
    for i in range(1, len(set_cover_data)):
        for tree_edge in set_cover_data[i]:
            not_found_probs[tree_edge] *= (1 - probs[i - 1])
    
    loss = torch.sum(not_found_probs)
    return loss

def solution_value_expectancy( probs, augmenting_edge_weights ):
    deduplicated_aug_edge_weights = augmenting_edge_weights[::2]#cause its 1d
    return (probs * deduplicated_aug_edge_weights).sum()#sum itself

#i think this only supports batch_size 1, currently
def unsupervised_erdos_loss(node_features, augmenting_edge_index, augmenting_edge_weights, set_cover_data, verbose=False):
    deduplicated_aug_ei = augmenting_edge_index[:, ::2]
    logits = edge_logits(node_features, deduplicated_aug_ei)
    probs = torch.sigmoid(torch.clamp(logits, -10, 10))

    if verbose:
        print("E(sol_val)")
        print(solution_value_expectancy(probs, augmenting_edge_weights))
        print("E(bridge_number)")
        print(bridge_number_expectancy(probs, set_cover_data, node_features.shape[0]))

    return solution_value_expectancy(probs, augmenting_edge_weights) + bridge_number_expectancy(probs, set_cover_data, node_features.shape[0])
