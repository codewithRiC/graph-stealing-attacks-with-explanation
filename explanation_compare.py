import torch
import os
import numpy as np

def load_feature_mask(file_path, target_size=1433):
    # Load the feature mask from the file
    feature_mask = torch.load(file_path)
    # Pad the feature mask to the target size
    if feature_mask.size(0) < target_size:
        padding = target_size - feature_mask.size(0)
        feature_mask = torch.nn.functional.pad(feature_mask, (0, padding))
    return feature_mask

def cosine_similarity(tensor1, tensor2):
    # Compute the cosine similarity between two tensors
    dot_product = torch.sum(tensor1 * tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    return dot_product / (norm1 * norm2)

def compare_explanations(exp1_dir, exp2_dir, num_nodes, target_size=1433):
    similarities = []
    for node in range(num_nodes):
        exp1_path = os.path.join(exp1_dir, f'feature_masks_node={node}.pt')
        exp2_path = os.path.join(exp2_dir, f'feature_masks_node={node}.pt')
        
        if os.path.exists(exp1_path) and os.path.exists(exp2_path):
            exp1_mask = load_feature_mask(exp1_path, target_size)
            exp2_mask = load_feature_mask(exp2_path, target_size)
            sim = cosine_similarity(exp1_mask, exp2_mask)
            similarities.append(sim.item())
        else:
            print(f"Feature mask for node {node} not found in one of the directories.")
    
    return np.mean(similarities)

# Directories containing the saved explanations
exp1_dir = 'Saved_Explanations/GraphLime/GCN/Cora'
exp2_dir = 'Saved_Explanations/GraphLime/GCN/Coraprivate'

# Number of nodes (adjust this based on your dataset)
num_nodes = 2708

# Compare the explanations and compute the similarity accuracy
similarity_accuracy = compare_explanations(exp1_dir, exp2_dir, num_nodes)
print(f'Similarity Accuracy: {similarity_accuracy:.4f}')