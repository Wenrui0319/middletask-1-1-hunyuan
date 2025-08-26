# hunyuan_logic.py

import os
import random
import shutil
import time
from glob import glob
from pathlib import Path
import uuid  # <-- FIX 1: Import uuid
import gradio as gr
import torch
import trimesh
from hy3dgen.shapegen.utils import logger
from hy3dgen.shapegen.pipelines import export_to_trimesh
args = None
SAVE_DIR = None
CURRENT_DIR = None
MV_MODE = False
TURBO_MODE = False
HTML_HEIGHT = 650
HTML_WIDTH = 500
MAX_SEED = int(1e7)

texgen_worker = None
t2i_worker = None
rmbg_worker = None
i23d_worker = None
floater_remove_worker = None
degenerate_face_remove_worker = None
face_reduce_worker = None
HAS_TEXTUREGEN = False
HAS_T2I = False

def initialize_hunyuan_models(cli_args):
    """Loads all Hunyuan models and sets up global variables for this module."""
    global args, SAVE_DIR, CURRENT_DIR, MV_MODE, TURBO_MODE, HTML_HEIGHT
    global texgen_worker, t2i_worker, rmbg_worker, i23d_worker
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker
    global HAS_TEXTUREGEN, HAS_T2I
    
    args = cli_args
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder
    HTML_HEIGHT = 690 if MV_MODE else 650

    if not args.disable_tex:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            if args.low_vram_mode: texgen_worker.enable_model_cpu_offload()
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e, "Failed to load texture generator.")
            HAS_TEXTUREGEN = False

    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=args.device)
        HAS_T2I = True

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, subfolder=args.subfolder, use_safetensors=True, device=args.device
    )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile: i23d_worker.compile()
    
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list

def get_example_mv_list():
    print('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
    for mv_dir in os.listdir(root):
        view_list = []
        for view in ['front', 'back', 'left', 'right']:
            path = os.path.join(root, mv_dir, f'{view}.png')
            if os.path.exists(path):
                view_list.append(path)
            else:
                view_list.append(None)
        mv_list.append(view_list)
    return mv_list

def gen_save_folder(max_size=200):
    os.makedirs(SAVE_DIR, exist_ok=True)
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
    return new_folder

def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()
    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)
    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')
    return f"<div style='height: {height}; width: 100%;'>{iframe_tag}</div>"

def _gen_shape(
    caption=None, image=None, mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
    steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, check_box_rembg=False, num_chunks=200000,
    randomize_seed: bool = False
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Please provide at least one view image.")
        image = {}
        if mv_image_front: image['front'] = mv_image_front
        if mv_image_back: image['back'] = mv_image_back
        if mv_image_left: image['left'] = mv_image_left
        if mv_image_right: image['right'] = mv_image_right
    seed = int(randomize_seed_fn(seed, randomize_seed))
    octree_resolution = int(octree_resolution)
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {
        'model': {'shapegen': f'{args.model_path}/{args.subfolder}', 'texgen': f'{args.texgen_model_path}'},
        'params': {'caption': caption, 'steps': steps, 'guidance_scale': guidance_scale, 'seed': seed,
                   'octree_resolution': octree_resolution, 'check_box_rembg': check_box_rembg, 'num_chunks': num_chunks}
    }
    time_meta = {}
    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time
    if MV_MODE:
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = rmbg_worker(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = rmbg_worker(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time
    start_time = time.time()
    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    outputs = i23d_worker(
        image=image, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
        octree_resolution=octree_resolution, num_chunks=num_chunks, output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start
    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]
    stats['time'] = time_meta
    main_image = image if not MV_MODE else image.get('front')
    return mesh, main_image, save_folder, stats, seed

def generation_all(
    caption=None, image=None, mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
    steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, check_box_rembg=False, num_chunks=200000,
    randomize_seed: bool = False
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption, image, mv_image_front=mv_image_front, mv_image_back=mv_image_back, mv_image_left=mv_image_left,
        mv_image_right=mv_image_right, steps=steps, guidance_scale=guidance_scale, seed=seed,
        octree_resolution=octree_resolution, check_box_rembg=check_box_rembg, num_chunks=num_chunks,
        randomize_seed=randomize_seed
    )
    path = export_mesh(mesh, save_folder, textured=False)
    tmp_time = time.time()
    mesh = face_reduce_worker(mesh)
    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time
    tmp_time = time.time()
    textured_mesh = texgen_worker(mesh, image)
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0
    textured_mesh.metadata['extras'] = stats
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return gr.update(value=path), gr.update(value=path_textured), model_viewer_html_textured, stats, seed

def shape_generation(
    caption=None, image=None, mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
    steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, check_box_rembg=False, num_chunks=200000,
    randomize_seed: bool = False, progress=gr.Progress()
):
    progress(0.3, "生成mesh中...")
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption, image, mv_image_front=mv_image_front, mv_image_back=mv_image_back, mv_image_left=mv_image_left,
        mv_image_right=mv_image_right, steps=steps, guidance_scale=guidance_scale, seed=seed,
        octree_resolution=octree_resolution, check_box_rembg=check_box_rembg, num_chunks=num_chunks,
        randomize_seed=randomize_seed
    )
    progress(0.7, "传输mesh结果中...")
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return gr.update(value=path), model_viewer_html, stats, seed