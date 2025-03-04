import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

# print(torch.__version__)
# exit()
from torch_geometric.nn import MessagePassing
import os
import torch.nn.functional as F
import torch_geometric.utils as ut
from pathlib import Path
from torch_geometric.utils import to_dense_adj,k_hop_subgraph
def save_model(model, path):
    torch.save(model.state_dict(), path)
from torch_geometric.utils import to_dense_adj,k_hop_subgraph
from graph_lime import *
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from explanations_utils import load_dataset,load_model,train_model
from explanations_utils import *
from grad_explainer import *
import collections
from fidelity import *
from gnn_explainer import *
from zorro import *

import argparse
from scipy.stats import entropy


def arg_parse():
    parser = argparse.ArgumentParser(description="arguments for generating explanations")
    parser.add_argument("--dataset",type=str,dest="dataset",help="input dataset")
    parser.add_argument("--explainer",type=str,dest="explainer",help="explainer to be used")
    parser.add_argument("--model",type=str,dest="model",help="GNN model")
    parser.add_argument("--save_exp",action="store_true",default=False,help="save explanation")
    parser.add_argument("--start",type=int, dest="start",help="start node")
    parser.add_argument("--end",type=int, dest="end",help="end node")
    parser.set_defaults(dataset="Cora",
                        explainer="Grad",
                        model="GCN")

    return parser.parse_args()

args = arg_parse()

#working_directory = Path("./tmp_ds/private_dataset/private_dataset1").resolve()
working_directory = Path("./tmp_ds").resolve()

if args.explainer=="Grad":
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
data_set = args.dataset
print(f"{data_set=} | {working_directory=}")
dataset, data, results_path = load_dataset(data_set, working_directory=working_directory)
data.to(device)
print(f"{dataset=} | {results_path=}")

model_dict = {"GCN":GCNNet,"GAT":GATNet,"GIN":GINConvNet,"APPNP":APPNP2Net}
explainer_dict ={"Grad":grad_node_explanation,"GradInput":gradinput_node_explanation,"GraphLime":GLIME,"GNNExplainer":GNNExplainer}


## Defining the GNN model
model = model_dict[args.model](dataset)
model.to(device)


### path to the saved model 
save_dir = 'saved_models'
model_directory = args.model
filename = os.path.join(save_dir, args.model)
saved_model = os.path.join(filename,args.dataset)
saved_model_dir = saved_model+"_.pth.tar"

# Check if the file exists
if not os.path.exists(saved_model_dir):
    print(f"Model file not found: {saved_model_dir}. Training and saving the model.")
    train_model(model, data)
    torch.save(model.state_dict(), saved_model_dir)
else:
    ### load the trained GNN model
    load_model(saved_model_dir, model)





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

num_nodes, num_features = data.x.size()



### Defining the path to save explanations
exp_save_dir = 'Saved_Explanations'
model_directory = args.model
explainer = args.explainer

exp_filename = os.path.join(exp_save_dir, args.explainer)
exp_filename = os.path.join(exp_filename, model_directory)
exp_dir = os.path.join(exp_filename,data_set)
print(exp_dir,"explanation")

print(f"============ WILL GENERATE EXPLANATIONS FOR {num_nodes} nodes ============")
for node in range(num_nodes): 
    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node, data.x, data.edge_index)
    # computation_graph_edge_index.to(device)
    # computation_graph_feature_matrix.to(device)

    # computation_data = Data(x=computation_graph_feature_matrix,
    #                         edge_index=computation_graph_edge_index).to(device)
    # computation_data.to(device)
    
    computation_graph_edge_index = computation_graph_edge_index.to(device)
    computation_graph_feature_matrix = computation_graph_feature_matrix.to(device)

    computation_data = Data(x=computation_graph_feature_matrix,
                            edge_index=computation_graph_edge_index).to(device)
    computation_data = computation_data.to(device)
    if args.explainer=="Grad":
        #node = torch.tensor(node).to(device)
        feature_mask, node_mask = grad_node_explanation(model,mapping,computation_graph_feature_matrix,computation_graph_edge_index)
        feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)

    if args.explainer=="GradInput":
        feature_mask, node_mask = gradinput_node_explanation(model,mapping,computation_graph_feature_matrix,computation_graph_edge_index)
        feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)
        
    if args.explainer=="GraphLime":
        lime = GLIME(model,computation_data.x,mapping,computation_data.edge_index,2,rho=0.1,device=device)##0.32
        feature_mask= lime.explain(x=data.x)
        feature_mask = feature_mask.reshape(1,-1).to(torch.float32).to(device)
        
    if args.explainer=="GNNExplainer":
        gnn_explainer = GNNExplainer(model,log=False)
        # feature_mask,edge_mask = gnn_explainer.explain_node(node_idx=mapping,x=computation_graph_feature_matrix,edge_index=computation_graph_edge_index)
        # edge_mask  =torch.nn.Parameter(edge_mask)
        # feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)
        
        # Ensure edge_index is 2-dimensional
        if computation_graph_edge_index.dim() == 1:
            computation_graph_edge_index = computation_graph_edge_index.unsqueeze(0)
        # Ensure edge_weight is handled correctly
        if 'edge_weight' in kwargs and kwargs['edge_weight'].dim() == 2:
            kwargs['edge_weight'] = kwargs['edge_weight'].squeeze(1)    
        feature_mask, edge_mask = gnn_explainer.explain_node(node_idx=mapping, x=computation_graph_feature_matrix, edge_index=computation_graph_edge_index)
        edge_mask = torch.nn.Parameter(edge_mask)
        feature_mask = feature_mask.reshape(1, -1)
        

    if args.explainer=="hard_zorro":
        hard_zorro = SISDistortionGraphExplainer(model,device)
        explanation = hard_zorro.explain_node(node,data.x,data.edge_index,tau=0.03,recursion_depth=1)
        selected_nodes, feature_mask, executed_selections = explanation[0]
        feature_mask = torch.from_numpy(feature_mask).to(device)
        
    elif args.explainer=="soft_zorro":
        soft_zorro= GradientBasedDistortionGraphExplainer(model,device)
        explanation = soft_zorro.explain_node(node,data.x,data.edge_index)
        node_masks= explanation[0]
        feature_mask = explanation[1].reshape(1,-1)
        feature_mask = torch.from_numpy(feature_mask).to(device)

    file_ext = ".pt"
    file_path = Path(exp_dir)
    file_name = "feature_masks_node=" + str(node) + file_ext
    file_full_path = file_path / file_name
    file_path_edge = exp_dir+"/edge_masks_node="+str(node) + file_ext
    if args.save_exp:
        if not file_path.exists():
            file_path.mkdir(parents=True, exist_ok=True)
        torch.save(feature_mask, file_full_path)
        print(f"SAVED to {file_full_path.absolute()}")
        #torch.save(edge_mask, file_path_edge)
