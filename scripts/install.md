- To prepare the encironment for running Fancy123, follow the following steps.
- We use Python 3.10.14 and torch 2.1.0+cu118. 
Other versions may also work, but we recommand torch2.1.0. 
- Our environment is mostly based on [InstantMesh](https://github.com/TencentARC/InstantMesh) and [Unique3D](https://github.com/AiuniAI/Unique3D/blob/main/Installation.md). If any problem occurs, it can be helpful to check their repos.
- Feel free to leave an issue if you encounter any problem.



# Create conda environment
```bash
conda create --name fancy123 python=3.10
conda activate fancy123
```

# Install basic stuff
```bash

pip install ninja # Ensure Ninja is installed


# Install PyTorch and xformers
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118 # will install xformers-0.0.22.post7+cu118 

pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html # Install kaolin

# For Linux users: Install Triton 
# pip install triton # already installed by the previous commands

pip install -r requirements.txt # Install other requirements


pip install huggingface-hub==0.23.4

```


# Install other stuff:
```bash
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp310-cp310-linux_x86_64.whl # torch sparse for 3D deformation

pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp310-cp310-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp310-cp310-linux_x86_64.whl

pip install onnxruntime-gpu==1.18.1  --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```
# Install TensorRT for Fast_SR
- We use Unique3D's fast_sr to increase resolution of multiview images. To accelerate this module, it's recommended to you install TensorRT, otherwise it would use CPU and would be slow.

- If you find it hard to install ommxruntime or tensorrt, you can:
    - simply skip it, it would be slow but can still work
    - Set fast_sr to False. The resolution would be lower.

```bash
pip install https://pypi.nvidia.com/tensorrt-cu11/tensorrt-cu11-10.1.0.tar.gz

# you also need to:
export LD_LIBRARY_PATH=your_conda_path/envs/fancy123/lib/python3.10/site-packages/tensorrt_libs/:${LD_LIBRARY_PATH}
# e.g. export LD_LIBRARY_PATH=/root/.conda/envs/fancy123/lib/python3.10/site-packages/tensorrt_libs/:${LD_LIBRARY_PATH}

```

# Trouble Shooting
- ImportError: cannot import name ‘packaging‘ from ‘pkg_resources‘:
    - ```python -m pip install setuptools==69.5.1```

- [E:onnxruntime:Default, provider_bridge_ort.cc:2167 TryGetProviderInfo_TensorRT] /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1778 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_tensorrt.so with error: libcublas.so.12: cannot open shared object file: No such file or directory

    - `libcublas.so.12` is with CUDA12. It seems that the most up-to-date version of onnxruntime-gpu requires cuda12, but here we use cuda11. Try specify the exact version for onnxruntime-gpu as follows:
        ```bash
        pip install onnxruntime-gpu==1.18.1  --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
        ```




# Download Models
For InstantMesh models, the python scripts will download them automatically. Alternatively, you can manually download the `instant-mesh-large` reconstruction model variant from the
 [model card](https://huggingface.co/TencentARC/InstantMesh).


For Unique3D's fast_sr, download the weights from [huggingface spaces](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt) or [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/319762ec478d46c8bdf7/), and extract it to unique3d/ckpt/realesrgan-x4.onnx.






