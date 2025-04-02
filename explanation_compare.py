from sklearn.metrics import roc_auc_score
import os
import torch
import numpy as np

def load_feature_mask(file_path):
    # Load the feature mask
    mask = torch.load(file_path)
    return mask.flatten().cpu().numpy()

def compare_explanations_auc(exp1_dir, exp2_dir, num_nodes):
    y_true = []  # Ground truth labels (from exp1_dir)
    y_score = []  # Predicted scores (from exp2_dir)

    for node in range(num_nodes):
        exp1_path = os.path.join(exp1_dir, f'feature_masks_node={node}.pt')
        exp2_path = os.path.join(exp2_dir, f'feature_masks_node={node}.pt')
        
        if os.path.exists(exp1_path) and os.path.exists(exp2_path):
            exp1_mask = load_feature_mask(exp1_path)
            exp2_mask = load_feature_mask(exp2_path)
            
            # Binarize y_true (exp1_mask) if it contains continuous values
            exp1_mask_binary = (exp1_mask > 0).astype(int)  # Convert to binary (0 or 1)

            # Append the masks for AUC-ROC computation
            y_true.extend(exp1_mask_binary)  # Ground truth (binary values)
            y_score.extend(exp2_mask)  # Predicted scores (continuous values)
        else:
            print(f"Feature mask for node {node} not found in one of the directories.")
    
    # Compute AUC-ROC
    if len(y_true) > 0 and len(y_score) > 0:
        auc_roc = roc_auc_score(y_true, y_score)
        return auc_roc
    else:
        raise ValueError("No valid feature masks found for comparison.")

# Directories containing the saved explanations
exp1_dir = 'Saved_Explanations/Grad/GCN/Pubmed_private'
exp2_dir = 'Saved_Explanations/Grad/GCN/PubMedPrivate'

# Number of nodes (adjust this based on your dataset)
num_nodes = 19717

# Compare the explanations and compute the AUC-ROC
try:
    auc_roc_score = compare_explanations_auc(exp1_dir, exp2_dir, num_nodes)
    print(f'AUC-ROC Score: {auc_roc_score:.4f}')
except ValueError as e:
    print(f"Error: {e}")