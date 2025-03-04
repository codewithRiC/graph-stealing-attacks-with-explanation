from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, APPNP, GINConv, global_add_pool
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx
import numpy as np

print("All imports successful!")