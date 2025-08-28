```bash
# install deps
pip install -r requirements.txt
pip install -e .

# download sam model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O models/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
# install sam deps
pip install git+https://github.com/facebookresearch/segment-anything.git

# install hunyuan3d deps
# sudo apt install gcc+-11 g++-11
cd hy3dgen/texgen/custom_rasterizer
CC=gcc-11 CXX=g++-11 python3 setup.py install

cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```
