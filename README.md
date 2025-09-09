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

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git

```
### 3. 后端模型安装
创建虚拟环境`Hunyuan`部署`Hunyuan 3D-2`和`SAM`模型；虚拟环境`Qwen`部署基于`Qwen-Image-Edit`模型的ComfyUI全局重绘和局部重绘工作流
#### A.安装 Hunyuan 3D-2 模型
```bash
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

#### B. 安装 Segment Anything (SAM)

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

#### C.通过 ComfyUI 部署 Qwen-Image-Edit 和 Qwen-Image-InPainting 工作流
安装ComfyUI软件：[ComfyUI Linux Installation Tutorial | ComfyUI Wiki](https://comfyui-wiki.com/en/install/install-comfyui/install-comfyui-on-linux)
```bash
comfy launch -- --enable-cors#启动comfyUI
#通过http://localhost:8081访问
```
在templates->image中找到Qwen-Image-Edit，下载并在指定目录放好模型文件，搭建工作流。或者直接导入工作流文件`image_qwen_image_edit.json`。
导入工作流文件`Qwen+Image+Inapint模型局部重绘V1.json`，下载并放置好模型文件。

#### D.通过 Gemini CLI 获取 Gemini 2.5 Pro API

#### E.获取 Nano Banana 图像编辑模型的API
将API key 填入 `gemini_image_edit_app.py`

### 4.运行
```bash
#进入Qwen虚拟环境启动Qwen-Image-Edit和Qwen-Image-InPainting工作流
comfy launch -- --enable-cors
```
```bash
#进入Hunyuan虚拟环境启动前端主程序、Hunyuan 3D后端、SAM后端
python app.py --sam_device cuda:2 --device cuda:1
```
通过`http://localhost:4000`访问主页面。