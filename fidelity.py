# License removed for repository anonymization
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP


def subgraph(model, node_idx, x, edge_index, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    flow = 'source_to_target'
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=num_nodes, flow=flow)

    x = x[subset]
    for key, item in kwargs:
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, kwargs


def fidelity(model,  # is a must
             node_idx,  # is a must
             full_feature_matrix,  # must
             edge_index=None,  # the whole, so data.edge_index
             node_mask=None,  # at least one of these three node, feature, edge
             feature_mask=None,
             edge_mask=None,
             samples=100,
             random_seed=12345,
             device="cpu",
             validity=False,
             ):
    """
    Distortion/Fidelity (for Node Classification)
    :param model: GNN model which is explained
    :param node_idx: The node which is explained
    :param full_feature_matrix: The feature matrix from the Graph (X)
    :param edge_index: All edges
    :param node_mask: Is a (binary) tensor with 1/0 for each node in the computational graph
    => 1 means the features of this node will be fixed
    => 0 means the features of this node will be pertubed/randomized
    if not available torch.ones((1, num_computation_graph_nodes))
    :param feature_mask: Is a (binary) tensor with 1/0 for each feature
    => 1 means this features is fixed for all nodes with 1
    => 0 means this feature is randomized for all nodes
    if not available torch.ones((1, number_of_features))
    :param edge_mask:
    :param samples:
    :param random_seed:
    :param device:
    :param validity:
    :return:
    """
    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node_idx, full_feature_matrix, edge_index)

    # get predicted label
    # log_logits = model(x=computation_graph_feature_matrix,
    #                    edge_index=computation_graph_edge_index)
    log_logits = model(computation_graph_feature_matrix,
                       computation_graph_edge_index)
    predicted_labels = log_logits.argmax(dim=-1)

    predicted_label = predicted_labels[mapping]

    # fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask= torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:
        # all nodes selected
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)

    # print("feature_mask", feature_mask.shape)
    # print("node_mask", node_mask.shape)


    # set edge mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = edge_mask

    (num_nodes, num_features) = full_feature_matrix.size()

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix

    mask = node_mask.T.matmul(feature_mask)

    if validity:
        samples = 1
        full_feature_matrix = torch.zeros_like(full_feature_matrix)

    correct = 0.0

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i, :, :])

        randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

        # log_logits = model(x=randomized_features, edge_index=computation_graph_edge_index)
        log_logits = model(randomized_features, computation_graph_edge_index)
        distorted_labels = log_logits.argmax(dim=-1)

        if distorted_labels[mapping] == predicted_label:
            correct += 1

    # reset mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    return correct / samples
