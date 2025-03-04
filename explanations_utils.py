
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import from_networkx
from pathlib import Path
import numpy as np
import os
from torch_geometric.data import Data
from read_bitcoin import *
import networkx as nx
SYN1_PATH = "data/syn1.npz"
SYN2_PATH = "data/syn2.npz"
from read_credit import read_credit_dataset

def load_dataset(data_set, working_directory=None):
    if working_directory is None:
        working_directory = Path(".").resolve()
    if data_set == "Cora":
        dataset = Planetoid(root=working_directory.joinpath('tmp/Cora'), name='Cora')
        data = dataset[0]
        results_path = "cora"
        
    # if data_set == "CoraPrivate":
    #     dataset = Planetoid(root=working_directory.joinpath('tmp/private_dataset/private_dataset1'), name='')
    #     data = dataset[0]
    #     results_path = "coraprivate"    
        
    elif data_set == "CoraPrivate":
        # Path to the private dataset folder
        private_dataset_dir = working_directory.joinpath("private_dataset/private_dataset1")
        
        # Load the .pt file
        graph_data = torch.load(os.path.join(private_dataset_dir, "graph_data.pt"))
        
        # Extract node features, labels, and masks
        node_features = graph_data["node_features"]
        node_labels = graph_data["node_labels"]
        train_mask = graph_data["train_mask"].squeeze()
        val_mask = graph_data["val_mask"].squeeze()
        test_mask = graph_data["test_mask"].squeeze()
        
        # Ensure node_labels is a 1D tensor
        if node_labels.dim() > 1:
            node_labels = node_labels.argmax(dim=1)  # Convert one-hot to class indices
          
        # Create a dummy edge_index if not provided
        num_nodes = node_features.size(0)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge_index
           
        # Create a PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,  
            y=node_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
      
        
        # Create a Dataset namedtuple
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], torch.max(data.y).item() + 1)
     
        
        results_path = "coraprivate"
        
        
    elif data_set == "CiteSeer":
        dataset = Planetoid(root=working_directory.joinpath('tmp/CiteSeer'), name='CiteSeer')
        data = dataset[0]
        results_path = "citeseer"
    elif data_set == "PubMed":
        dataset = Planetoid(root=working_directory.joinpath('tmp/PubMed'), name='PubMed')
        data = dataset[0]
        results_path = "pubmed"
    elif data_set == "AmazonC":
        dataset = Amazon(root=working_directory.joinpath("tmp/AmazonComputers"), name="Computers")
        data = dataset[0]
        results_path = "amazon_computers"
        # add train/test split
        # select 20 nodes from each class
        data.train_mask, data.val_mask, data.test_mask = create_train_val_test_mask(data)
    elif data_set=="Chameleon":
        data = read_chameleon_dataset("./data/Chameleon/")
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)

        results_path = "Chameleon"
    elif data_set=="Credit":
        data = read_credit_dataset(path="./data/Credit/", label_number=15000)
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)
        results_path = "Credit"

    elif data_set == "Credit_mini":
        data = read_credit_dataset_mini(path="./data/Credit/", label_number=15000)
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)
        results_path = "Credit_mini"



    elif data_set[:4] == "syn2":
        try:
            save_data = np.load(working_directory.joinpath(SYN2_PATH))
        except FileNotFoundError:
            save_data = create_syn(data_set)

        transformed_data = {}
        for name in save_data:
            transformed_data[name] = torch.tensor(save_data[name])
        data = torch_geometric.data.Data.from_dict(transformed_data)

        results_path = data_set
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(10, max(data.y.numpy()) + 1)

    elif data_set == "syn1":
        try:
            save_data = np.load(working_directory.joinpath(SYN1_PATH))
        except FileNotFoundError:
            save_data = create_syn(data_set)

        transformed_data = {}
        for name in save_data:
            transformed_data[name] = torch.tensor(save_data[name])
        data = torch_geometric.data.Data.from_dict(transformed_data)

        results_path = "syn1"
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(1, max(data.y.numpy()) + 1)

    elif data_set =='cora_ml':
        graph = load_dataset_text(data_set)
        A = graph['A']
        y = torch.tensor(graph['z'])

        x = np.load('w2v_embeddings.npy',allow_pickle=True)
        x = torch.tensor(x,dtype=torch.float)
        Acoo = A.tocoo()

        Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                      torch.LongTensor(Acoo.data.astype(np.int32)))

        edge_index = Apt._indices()

        data = Data(x=x,edge_index=edge_index,y=y)
        print(data.y)
        data.train_mask, data.val_mask, data.test_mask = create_train_val_test_mask(data)
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)
        results_path = "cora_ml"


    elif data_set=="Bitcoin_alpha":
        g, labels, name = read_bitcoinalpha()
        A = nx.adjacency_matrix(g).todense()

        data = from_networkx(g)
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.RATING

        data.y = np.array(labels)
        data.y = torch.from_numpy(data.y)
        num_nodes = A.shape[0]
        train_ratio = 0.8
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)

        train_mask = np.full_like(data.y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(data.y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        data.train_mask, data.test_mask = train_mask, test_mask
        from collections import namedtuple
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y).item() + 1)

        results_path = "Bitcoin_alpha"


    else:
        raise ValueError("Dataset " + data_set + "not implemented")

    return dataset, data, results_path


import numpy as np
import scipy.sparse as sp

# Feteches private edges for a given node from private dataset
def get_private_edges(node_id, private_data):
    """
    Fetches private edges for a given node from the privatized dataset.

    Args:
        node_id (int): Node ID.
        private_data (dict): Node-wise private data dictionary.

    Returns:
        Tensor: Edges connected to the node.
    """
    if node_id not in private_data:
        raise KeyError(f"Node {node_id} not found in private dataset.")
    return private_data[node_id]

def load_dataset_text(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    # np_load_old = np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    with np.load(file_name,allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def create_syn(dataset_name="syn2"):
    import generate_gnnexplainer_dataset as gn
    if dataset_name == "syn2":
        g, labels, name = gn.gen_syn2()
    elif dataset_name == "syn1":
        g, labels, name = gn.gen_syn1()
    else:
        raise NotImplementedError("Dataset not known")

    data = from_networkx(g)

    edge_index = data.edge_index.numpy()
    x = data.x.numpy().astype(np.float32)
    y = np.array(labels)

    train_ratio = 0.8

    num_nodes = x.shape[0]
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_mask = np.full_like(y, False, dtype=bool)
    train_mask[idx[:num_train]] = True
    test_mask = np.full_like(y, False, dtype=bool)
    test_mask[idx[num_train:]] = True

    save_data = {"edge_index": edge_index,
                 "x": x,
                 "y": y,
                 "train_mask": train_mask,
                 "test_mask": test_mask,
                 "num_nodes": g.number_of_nodes()
                 }

    if dataset_name == "syn2":
        np.savez_compressed(SYN2_PATH, **save_data)
    elif dataset_name == "syn1":
        np.savez_compressed(SYN1_PATH, **save_data)
    return save_data


# a slight adoption of the method of Planetoid
def create_train_val_test_mask(data, num_train_per_class=20, num_classes=None, num_val=500, num_test=1000, ):
    import numpy as np
    # fix seed for selecting train_mask
    # rng = np.random.default_rng(seed=42 * 20200909)
    rng = np.random.RandomState(seed=42 * 20200909)

    if num_classes is None:
        num_classes = torch.max(data.y)

    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero().view(-1)
        idx = idx[rng.permutation(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero().view(-1)
    remaining = remaining[rng.permutation(remaining.size(0))]

    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[remaining[:num_val]] = True

    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask


class GCNNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)
        elif edge_index.dim() == 2 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCNNet_bitcoin(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNet_bitcoin, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index,edge_attr):
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)
        elif edge_index.dim() == 2 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)
        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,edge_attr)

        return F.log_softmax(x, dim=1)


class GCN_syn2(torch.nn.Module):
    # only for syn2
    def __init__(self, dataset):
        super(GCN_syn2, self).__init__()
        hidden_dim = 20
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.lin_pred = Linear(3 * hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_all = [x]
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = self.lin_pred(x)

        return F.log_softmax(x, dim=1)


class GCNNetDeep(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNetDeep, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 16)
        self.conv4 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    # based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP10Net(torch.nn.Module):
    def __init__(self, dataset):
        super(APPNP10Net, self).__init__()
        # default values from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py
        self.dropout = 0.5
        self.hidden = 64
        self.K = 10
        self.alpha = 0.1
        self.lin1 = Linear(dataset.num_features, self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = APPNP(self.K, self.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP2Net(torch.nn.Module):
    def __init__(self, dataset):
        super(APPNP2Net, self).__init__()
        # default values from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py
        self.dropout = 0.5
        self.hidden = 64
        self.K = 2  # adjusted to two layers
        self.alpha = 0.1
        self.lin1 = Linear(dataset.num_features, self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = APPNP(self.K, self.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GINConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINConvNet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)
        #
        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)
        #
        # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv5 = GINConv(nn5)
        # self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        # x = F.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# For Cora

def load_model(path, model):
    print("TEST Path: " + path)
    saved_state_dict = torch.load(path)
    print(f"SAVED MODEL: {saved_state_dict.keys()=}")
    print(f"DEFINED MODEL: {model.state_dict().keys()}")

    # # Rename keys to match your model's layer names
    new_state_dict = {}
    for key, value in saved_state_dict.items():
        print(f"{key=}")
        # Check for conv1 and conv2 keys and rename them
        if key == "conv1.weight":
            new_key = "conv1.lin.weight"
            new_value = value.T
        elif key == "conv2.weight":
            new_key = "conv2.lin.weight"
            new_value = value.T
        elif key == "conv1.lin.weight":
            new_key = "conv1.lin.weight"
            new_value = value
            if new_value.shape != model.conv1.lin.weight.shape:
                new_value = new_value[:, :model.conv1.lin.weight.shape[1]]  
        else:
            new_key = key  # Keep other keys unchanged
            new_value = value
        print(f"{new_key=}")
        new_state_dict[new_key] = new_value
        
        print(f"{new_state_dict.keys()=}")
    for key, value in new_state_dict.items():
        print(f"{key}: {value.shape}")
    
    
    # new_state_dict = {}
    # for key, value in saved_state_dict.items():
    #     print(f"{key=}")
    #     # Check for conv1 and conv2 keys and rename them
    #     if key == "conv1.lin.weight":
    #         new_key = "conv1.lin.weight"
    #         new_value = value
    #         print(f"{new_value.shape=}")
    #         if new_value.shape != model.conv1.lin.weight.shape:
    #             new_value = new_value[:, :model.conv1.lin.weight.shape[1]]
    #     elif key == "conv2.weight":
    #         new_key = "conv2.lin.weight"
    #         new_value = value
    #     else:
    #         new_key = key  # Keep other keys unchanged
    #         new_value = value
    #     print(f"{new_key=}")
    #     new_state_dict[new_key] = new_value
    #     print(f"{new_state_dict.keys()=}")
    # for key, value in new_state_dict.items():
    #     print(f"{key}: {value.shape}")
    
    print(f"MODIFIED SAVED DICT: {new_state_dict.keys()}")

    # print(f"{model=}")
    # print(f"{dir(model.conv1)=}")
    # print(f"{torch.load(path)=}")
    if not torch.cuda.is_available():  ## NOTE Not run as long as CUDA is available
        # model.load_state_dict(torch.load(path, map_location="cpu"))
        model.load_state_dict(new_state_dict, map_location="cpu")
    else:
        # model.load_state_dict(torch.load(path), strict=False)
        model.load_state_dict(new_state_dict)
    model.eval()
   


def save_model(model, path):
    torch.save(model.state_dict(), path)

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, clip=None, loss_function="nll_loss",
                epoch_save_path=None, no_output=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    accuracies = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if loss_function == "nll_loss":
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        elif loss_function == "cross_entropy":
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], size_average=True)
        else:
            raise Exception()
        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        loss.backward()
        optimizer.step()

        if epoch_save_path is not None:
            # circumvent .pt ending
            save_model(model, epoch_save_path[:-3] + "_epoch_" + str(epoch) + epoch_save_path[-3:])
            accuracies.append(retrieve_accuracy(model, data, value=True))
            print('Accuracy: {:.4f}'.format(accuracies[-1]), "Epoch", epoch)
        else:
            if epoch % 25 == 0 and not no_output:
                print(retrieve_accuracy(model, data))

    model.eval()

    return accuracies




def retrieve_accuracy(model, data, test_mask=None, value=False):
    _, pred = model(data.x, data.edge_index).max(dim=1)
    if test_mask is None:
        test_mask = data.test_mask
    correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    if value:
        return acc
    else:
        return 'Accuracy: {:.4f}'.format(acc)
    
    
    
    


