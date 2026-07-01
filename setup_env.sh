#!/usr/bin/env bash
# Reproducible environment for the InfraScan 3D walk-through viewer (reconstruction side).
#
# Stack: depth-supervised 3D Gaussian Splatting authored on `gsplat`.
# Key choice: we install gsplat's PREBUILT CUDA wheel (no nvcc / no local compile),
# which pins the whole toolchain — Python 3.10 + torch 2.1.2 + CUDA 11.8.
#
# Requires: `uv` (https://docs.astral.sh/uv/) and an NVIDIA GPU with a CUDA 11.8+ driver.
# Usage: bash setup_env.sh   then   source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

# 1. Python 3.10 venv (the prebuilt gsplat wheel is cp310-only for this torch/cuda combo).
uv venv --python 3.10 .venv
source .venv/bin/activate

# 2. PyTorch built for CUDA 11.8.
uv pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu118

# 3. Data / IO libs. numpy pinned <2 because torch 2.1.2 predates the numpy 2 ABI.
uv pip install "numpy<2" plyfile open3d opencv-python-headless imageio tqdm pillow

# 4. gsplat — deps from PyPI first, then the PREBUILT wheel with --no-deps from the
#    gsplat wheel index (that index hosts only gsplat, so deps must come from PyPI).
uv pip install jaxtyping rich typing_extensions ninja "setuptools<70"
uv pip install gsplat --no-deps --index-url https://docs.gsplat.studio/whl/pt21cu118

# 5. Smoke test: forward + backward rasterization on the GPU.
python - <<'PY'
import torch, gsplat
N=64; d='cuda'
q=torch.randn(N,4,device=d); q/=q.norm(dim=-1,keepdim=True)
vm=torch.eye(4,device=d)[None]; vm[0,2,3]=5
K=torch.tensor([[100.,0,32],[0,100.,32],[0,0,1]],device=d)[None]
m=torch.randn(N,3,device=d,requires_grad=True)
out,_,_=gsplat.rasterization(m,q,torch.rand(N,3,device=d)*.1,torch.rand(N,device=d),
                             torch.rand(N,3,device=d),vm,K,64,64)
out.sum().backward()
print(f"OK gsplat {gsplat.__version__} fwd+bwd on {torch.cuda.get_device_name(0)}")
PY
echo "Environment ready. Activate with:  source .venv/bin/activate"
