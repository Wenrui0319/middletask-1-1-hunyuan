## 部署与安装指南

### 1. 环境准备

在开始之前，请确保您的系统已安装以下基础软件：

*   **Git**: 用于克隆项目仓库。
*   **Python**: 推荐使用 `3.8` 或更高版本。
*   **pip**: Python 包管理器 (通常随 Python 一起安装)。
*   **wget**: 用于从命令行下载文件。
*   **(可选但强烈推荐) NVIDIA GPU**: 并已正确安装相应的 [CUDA](https://developer.nvidia.com/cuda-downloads) 和 [cuDNN](https://developer.nvidia.com/cudnn) 驱动，以便进行GPU加速。
*   **C/C++ 编译器**: 用于编译自定义的 CUDA 核心。在基于 Debian/Ubuntu 的系统上，可以通过 `sudo apt install build-essential` 安装。

### 2. 克隆项目仓库

首先，我们将克隆主项目仓库、创建虚拟环境并安装核心依赖。

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2

pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```
### 3. 组件专项安装

完成主项目安装后，需要对 `Segment Anything` 和 `Hunyuan3D` 两个核心组件进行专项安装。

#### A. 安装 Segment Anything (SAM)

此步骤将安装 SAM 库并下载预训练的模型权重。

```bash
# 1. 从官方GitHub仓库安装SAM库
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. 创建模型存储目录 (如果尚不存在)
mkdir -p models

# 3. 下载预训练的SAM模型权重
# ViT-H (推荐, 效果最好)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
# ViT-L (次之)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O models/sam_vit_l_0b3195.pth
# ViT-B (最小)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```

#### B.通过 ComfyUI 部署 Qwen-Image-Edit 和 Qwen-Image-InPainting 模型
安装ComfyUI软件：[ComfyUI Linux Installation Tutorial | ComfyUI Wiki](https://comfyui-wiki.com/en/install/install-comfyui/install-comfyui-on-linux)
```bash
# 1. 从官方GitHub仓库安装SAM库
pip install git+https://github.com/facebookresearch/segment-anything.git

# 2. 创建模型存储目录 (如果尚不存在)
mkdir -p models

# 3. 下载预训练的SAM模型权重
# ViT-H (推荐, 效果最好)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
# ViT-L (次之)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O models/sam_vit_l_0b3195.pth
# ViT-B (最小)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```