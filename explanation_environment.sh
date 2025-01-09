set -e


#conda env create python=3.8 --file explanation_environment.yml  # --force

#conda init

#conda activate privacy
TORCH=1.9.1
CUDA=cu111
python -m pip install --force torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#python -m pip install --force torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install --force torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install --force torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install --force torch-geometric --no-cache-dir
python -m pip install --force graphlime
