import torch
import pandas as pd

# Iterate through files
for i in range(1):
    file_path = f"./Saved_Explanations/GraphLime/GCN/Cora/feature_masks_node={i}.pt"
    # file_path = f".../tmp/Cora/Cora/processed/data.pt"
    # file_path = f"./tmp_ds/private_dataset/private_dataset1/graph_data.pt"

    try:
        # Load the .pt file
        data = torch.load(file_path)

        # If data is a dictionary, convert it to a table
        if isinstance(data, dict):
            print(f"Node {i} - Keys in the .pt file:")
            for key, value in data.items():
                if torch.is_tensor(value):  # Convert tensor data to a table
                    # Move the tensor to CPU if it's on CUDA
                    if value.is_cuda:
                        value = value.cpu()

                    # Convert tensor to a DataFrame
                    df = pd.DataFrame(value.numpy())  
                    print(f"\nKey: {key}")
                    print(df.to_string(index=False))  # Print the entire DataFrame
                else:
                    print(f"Key: {key}, Value: {value}")  # For non-tensor data
        else:
            # For non-dict data, print it directly
            print(f"Node {i} - Content of the .pt file:")
            if torch.is_tensor(data):
                # Move the tensor to CPU if it's on CUDA
                if data.is_cuda:
                    data = data.cpu()

                # Convert tensor to a DataFrame
                df = pd.DataFrame(data.numpy())  
                print(df)
            else:
                print(data)
                exit()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
