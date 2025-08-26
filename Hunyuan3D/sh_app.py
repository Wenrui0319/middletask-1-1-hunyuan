# ==============================================================================
# SECTION 1: IMPORTS & INITIAL CONFIGURATION
# ==============================================================================
import os
import cv2
import torch
import logging
import numpy as np
from typing import Tuple, Dict, Any, List
from skimage import color
import warnings
import random
import shutil
import time
from glob import glob
from pathlib import Path
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import gradio as gr

# --- Import Segment Anything (SAM) specific libraries ---
try:
    from segment_anything import (
        SamAutomaticMaskGenerator,
        build_sam_vit_b,
        build_sam_vit_h,
        build_sam_vit_l,
        SamPredictor,
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("Segment Anything library not found. The 'SAM Segmentation' tab will be disabled.")

# --- Import Hunyuan specific libraries ---
from hy3dgen.shapegen.utils import logger as hy_logger

# --- Basic Configuration ---
MAX_SEED = int(1e7)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Gradio Temporary Directory Setup ---
try:
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['GRADIO_TEMP_DIR'] = temp_dir
except (OSError, PermissionError) as e:
    warnings.warn(f"无法创建或设置 Gradio 临时目录: {e}。将使用默认设置。")
    temp_dir = None

# ==============================================================================
# SECTION 2: BACKEND FUNCTIONS FOR SAM SEGMENTATION
# ==============================================================================
if SAM_AVAILABLE:
    def create_color_mask(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
        if not annotations: return image
        sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
        if not sorted_anns: return image
        h, w, _ = image.shape
        mask_img = np.zeros((h, w), dtype=np.uint16)
        for i, ann in enumerate(sorted_anns):
            mask_img[ann["segmentation"]] = i + 1
        color_mask = color.label2rgb(mask_img, image, bg_label=0, kind='overlay')
        return (color_mask * 255).astype(np.uint8)

    def cut_out_object(original_image: np.ndarray, mask: np.ndarray):
        if original_image is None or mask is None:
            gr.Warning("请先生成一个蒙版。")
            return None
        rgba_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
        rgba_image[:, :, 3] = mask * 255
        return rgba_image

    @torch.no_grad()
    def generate_everything(predictor: SamPredictor, original_image: np.ndarray, progress=gr.Progress()) -> Tuple[np.ndarray, List[np.ndarray]]:
        if original_image is None: raise gr.Error("请先上传一张图片。")
        if predictor is None: raise gr.Error("模型未加载。")
        progress(0.1, desc="正在运行全图自动分割...")
        generator = SamAutomaticMaskGenerator(model=predictor.model)
        annotations = generator.generate(original_image)
        progress(0.6, desc="正在创建彩色蒙版和抠图...")
        color_mask = create_color_mask(original_image, annotations)
        cutouts = []
        for ann in sorted(annotations, key=(lambda x: x['area']), reverse=True):
            cutouts.append(cut_out_object(original_image, ann['segmentation']))
        progress(1, desc="完成！")
        return color_mask, cutouts

    def load_model(name: str, device: str, progress=gr.Progress()) -> SamPredictor:
        if not name:
            gr.Warning("没有可加载的模型。")
            return None
        progress(0, desc=f"正在加载模型: {name}...")
        checkpoint_path = os.path.join("models", name)
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"模型文件未找到: {checkpoint_path}")

        model_builder = {"vit_b": build_sam_vit_b, "vit_h": build_sam_vit_h, "vit_l": build_sam_vit_l}
        model_type = next((t for t in model_builder if t in name.lower()), None)
        if model_type is None: raise ValueError(f"无效的模型文件名: {name}。必须包含 'vit_b', 'vit_l', 或 'vit_h'。")

        try:
            model = model_builder[model_type](checkpoint=checkpoint_path)
            model.to(device)
            logger.info(f"模型已加载: {name}")
            progress(1, desc=f"模型 {name} 加载成功！")
            return SamPredictor(model)
        except Exception as e:
            raise gr.Error(f"加载模型失败: {e}")

    def set_image_for_predictor(predictor: SamPredictor, image: np.ndarray):
        if predictor is None:
            gr.Warning("模型尚未加载。")
            return None, None, None, [], None, None, None
        if image is not None:
            predictor.set_image(image)
            gr.Info("图片已设置。")
        return image, image, None, [], None, predictor, None # Reset box_start_state

    def visualize_prompts(image, history, current_mask=None, box_start_point=None):
        if image is None: return None
        vis_image = image.copy()
        if current_mask is not None:
            blue_mask = np.zeros_like(vis_image, dtype=np.uint8)
            blue_mask[current_mask] = [65, 105, 225]
            vis_image = cv2.addWeighted(vis_image, 1.0, blue_mask, 0.5, 0)
        
        # 可视化已完成的提示
        for prompt in history:
            if prompt['type'] == 'box':
                box = prompt['data']
                pt1 = (int(box[0][0]), int(box[0][1]))
                pt2 = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 2)
            elif prompt['type'] == 'point':
                point, label = prompt['data']
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(vis_image, (int(point[0]), int(point[1])), 8, color, -1)
        
        # 如果正在画框，可视化第一个点
        if box_start_point is not None:
            pt = (int(box_start_point[0]), int(box_start_point[1]))
            cv2.circle(vis_image, pt, 8, (255, 255, 0), -1) # 用黄色表示起始点

        return vis_image
    
    # ==============================================================================
    # START: FINAL ROBUST SOLUTION
    # This function implements a manual, two-click box drawing logic that only
    # relies on the click event, which is confirmed to be working.
    # ==============================================================================
    @torch.no_grad()
    def interactive_predict(
        predictor: SamPredictor,
        original_image: np.ndarray,
        history: List,
        mode: str,
        box_start_state: List, # 用于存储画框的第一个点
        evt: gr.SelectData
    ) -> Tuple[np.ndarray, np.ndarray, List, SamPredictor, List]:
        if predictor is None or original_image is None or not predictor.is_image_set:
            raise gr.Error("请先上传图片并确保模型已加载！")

        if mode == "添加点 (Add Point)":
            # 添加点模式逻辑
            new_prompt = {'type': 'point', 'data': (evt.index, 1)}
            history.append(new_prompt)
            box_start_state = None # 如果在画框时切换到点模式，则取消画框

        elif mode == "画框 (Draw Box)":
            if box_start_state is None:
                # 这是画框的第一次点击
                gr.Info("已选择第一个点，请点击对角点以完成画框。")
                box_start_state = evt.index
                # 只更新UI，不进行预测
                vis_image = visualize_prompts(original_image, history, None, box_start_state)
                return vis_image, None, history, predictor, box_start_state
            else:
                # 这是画框的第二次点击
                box_end_point = evt.index
                # 一个框会覆盖所有之前的提示
                history = [{'type': 'box', 'data': [box_start_state, box_end_point]}]
                box_start_state = None # 重置画框状态
                gr.Info("画框完成，正在分割...")

        # --- 统一的预测逻辑 ---
        point_coords, point_labels, box_coords = [], [], None
        for prompt in history:
            if prompt['type'] == 'point':
                point_coords.append(prompt['data'][0])
                point_labels.append(prompt['data'][1])
            elif prompt['type'] == 'box':
                box_coords = np.array(prompt['data']).flatten()

        point_coords_np = np.array(point_coords) if point_coords else None
        point_labels_np = np.array(point_labels) if point_labels else None

        masks, scores, _ = predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            box=box_coords,
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores)]
        vis_image = visualize_prompts(original_image, history, best_mask, box_start_state)
        
        return vis_image, best_mask, history, predictor, box_start_state
    # ==============================================================================
    # END: FINAL ROBUST SOLUTION
    # ==============================================================================

    def single_cutout(image, mask):
        if image is None or mask is None:
            gr.Warning("请先通过点选或框选生成一个蒙版。")
            return None
        cutout = cut_out_object(image, mask)
        return [cutout] if cutout is not None else None

    def reset_all(predictor, original_image):
        if predictor is not None and original_image is not None:
            predictor.set_image(original_image)
            gr.Info("已重置。")
        return original_image, original_image, None, [], None, predictor, None # Reset box_start_state

# ==============================================================================
# SECTION 3: BACKEND FUNCTIONS FOR HUNYUAN 3D
# ==============================================================================
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
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')
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
    hy_logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
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
    hy_logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time
    tmp_time = time.time()
    textured_mesh = texgen_worker(mesh, image)
    hy_logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
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
    randomize_seed: bool = False
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption, image, mv_image_front=mv_image_front, mv_image_back=mv_image_back, mv_image_left=mv_image_left,
        mv_image_right=mv_image_right, steps=steps, guidance_scale=guidance_scale, seed=seed,
        octree_resolution=octree_resolution, check_box_rembg=check_box_rembg, num_chunks=num_chunks,
        randomize_seed=randomize_seed
    )
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return gr.update(value=path), model_viewer_html, stats, seed

# ==============================================================================
# SECTION 4: GRADIO UI APPLICATION BUILDER
# ==============================================================================
def build_combined_app():
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) { max-width: 1480px; }
    .mv-image button .wrap { font-size: 10px; }
    .mv-image .icon-wrap { width: 20px; }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Hunyuan & SAM", analytics_enabled=False, css=custom_css) as demo:
        gr.Markdown("# Interactive Demo: Hunyuan 3D & Segment Anything")
        
        with gr.Tabs():
            # ------------------- TAB 1: Hunyuan 3D -------------------
            with gr.TabItem("Hunyuan"):
                title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
                if MV_MODE: title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
                if 'mini' in args.subfolder: title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
                if TURBO_MODE: title = title.replace(':', '-Turbo: Fast ')

                gr.HTML(f"""
                <div style="font-size: 1.5em; font-weight: bold; text-align: center; margin-bottom: 5px">{title}</div>
                <div align="center">
                  <a href="https://github.com/tencent/Hunyuan3D-2">Github</a> &ensp; 
                  <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
                  <a href="https://3d.hunyuan.tencent.com">Hunyuan3D Studio</a> &ensp;
                  <a href="#">Technical Report</a> &ensp;
                  <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Pretrained Models</a> &ensp;
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                            with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                                image_hy = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                            with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
                                caption_hy = gr.Textbox(label='Text Prompt', placeholder='A 3D model of a cute cat...')
                            with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                                with gr.Row():
                                    mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                                    mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                                with gr.Row():
                                    mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                                    mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                            btn_all = gr.Button(value='Gen Textured Shape', variant='primary', visible=HAS_TEXTUREGEN, min_width=100)
                        with gr.Group():
                            file_out = gr.File(label="File", visible=False)
                            file_out2 = gr.File(label="File", visible=False)
                        with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                            with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                                gen_mode = gr.Radio(label='Generation Mode', choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                                decode_mode = gr.Radio(label='Decoding Mode', choices=['Low', 'Standard', 'High'], value='Standard')
                            with gr.Tab('Advanced Options', id='tab_advanced_options'):
                                with gr.Row():
                                    check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=1234)
                                with gr.Row():
                                    num_steps = gr.Slider(maximum=100, minimum=1, value=5 if 'turbo' in args.subfolder else 30, step=1, label='Inference Steps')
                                    octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                                with gr.Row():
                                    cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                                    num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000, label='Number of Chunks')
                            with gr.Tab("Export", id='tab_export'):
                                with gr.Row():
                                    file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS, value='glb')
                                    reduce_face = gr.Checkbox(label='Simplify Mesh', value=False)
                                    export_texture = gr.Checkbox(label='Include Texture', value=False, visible=False)
                                target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000, label='Target Face Number')
                                with gr.Row():
                                    confirm_export = gr.Button(value="Transform")
                                    file_export = gr.DownloadButton(label="Download", variant='primary', interactive=False)
                    with gr.Column(scale=6):
                        with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                            with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                                html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                            with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                                html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                            with gr.Tab('Mesh Statistic', id='stats_panel'):
                                stats = gr.Json({}, label='Mesh Stats')
                    with gr.Column(scale=3 if MV_MODE else 2):
                        with gr.Tabs(selected='tab_img_gallery') as gallery:
                            with gr.Tab('Image to 3D Gallery', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                                gr.Examples(examples=example_is, inputs=[image_hy], label=None, examples_per_page=18)
                            with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                                gr.Examples(examples=example_ts, inputs=[caption_hy], label=None, examples_per_page=18)
                            with gr.Tab('MultiView to 3D Gallery', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                                gr.Examples(examples=example_mvs, inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right], label=None, examples_per_page=6)
                
                # Hunyuan Event Handlers
                tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
                if HAS_T2I:
                    tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)
                
                btn.click(
                    shape_generation,
                    inputs=[caption_hy, image_hy, mv_image_front, mv_image_back, mv_image_left, mv_image_right, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed],
                    outputs=[file_out, html_gen_mesh, stats, seed]
                ).then(
                    lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)),
                    outputs=[export_texture, reduce_face, confirm_export, file_export],
                ).then(lambda: gr.update(selected='gen_mesh_panel'), outputs=[tabs_output])

                btn_all.click(
                    generation_all,
                    inputs=[caption_hy, image_hy, mv_image_front, mv_image_back, mv_image_left, mv_image_right, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed],
                    outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
                ).then(
                    lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)),
                    outputs=[export_texture, reduce_face, confirm_export, file_export],
                ).then(lambda: gr.update(selected='gen_mesh_panel'), outputs=[tabs_output])

                gen_mode.change(lambda v: gr.update(value=5 if v == 'Turbo' else 10 if v == 'Fast' else 30), inputs=[gen_mode], outputs=[num_steps])
                decode_mode.change(lambda v: gr.update(value=196 if v == 'Low' else 256 if v == 'Standard' else 384), inputs=[decode_mode], outputs=[octree_resolution])

                def on_export_click(file_out_path, file_out2_path, file_type_val, reduce_face_val, export_texture_val, target_face_num_val):
                    if file_out_path is None: raise gr.Error('Please generate a mesh first.')
                    if export_texture_val:
                        mesh = trimesh.load(file_out2_path.name)
                        save_folder = gen_save_folder()
                        path = export_mesh(mesh, save_folder, textured=True, type=file_type_val)
                        _ = export_mesh(mesh, gen_save_folder(), textured=True)
                        model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
                    else:
                        mesh = trimesh.load(file_out_path.name)
                        mesh = floater_remove_worker(mesh)
                        mesh = degenerate_face_remove_worker(mesh)
                        if reduce_face_val: mesh = face_reduce_worker(mesh, target_face_num_val)
                        save_folder = gen_save_folder()
                        path = export_mesh(mesh, save_folder, textured=False, type=file_type_val)
                        _ = export_mesh(mesh, gen_save_folder(), textured=False)
                        model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)
                    return model_viewer_html, gr.update(value=path, interactive=True)

                confirm_export.click(
                    lambda: gr.update(selected='export_mesh_panel'), outputs=[tabs_output]
                ).then(on_export_click, inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num], outputs=[html_export_mesh, file_export])

            # ------------------- TAB 2: SAM Segmentation -------------------
            with gr.TabItem("SAM Segmentation", visible=SAM_AVAILABLE):
                # SAM Global States
                predictor_state = gr.State(None)
                original_image_state = gr.State(None)
                mask_state = gr.State(None)
                history_state = gr.State([])
                box_start_state = gr.State(None) # State to hold the first click of a box

                gr.Markdown("## Segment Anything 交互式演示")
                with gr.Row():
                    # Setup SAM model loading
                    sam_model_dir = "models"
                    if not os.path.exists(sam_model_dir): os.makedirs(sam_model_dir)
                    available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
                    default_sam_model = available_sam_models[0] if available_sam_models else None
                    sam_device = "cuda" if torch.cuda.is_available() else "cpu"

                    selected_model = gr.Dropdown(choices=available_sam_models, label="模型 (Model)", value=default_sam_model)
                    load_model_btn = gr.Button("加载模型 (Load Model)")
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=250):
                        with gr.Group():
                            source_image = gr.Image(label="上传图片 (Upload Image)", type="numpy")
                        with gr.Group():
                            mode_radio = gr.Radio(choices=["添加点 (Add Point)", "画框 (Draw Box)"], value="添加点 (Add Point)", label="工具 (Tool)")
                            cut_everything_btn = gr.Button("分割所有物体 (Segment Everything)")
                            cut_out_btn = gr.Button("抠图 (Cut out object)")
                            reset_btn = gr.Button("重置 (Reset)")
                    with gr.Column(scale=3):
                        interactive_display = gr.Image(
                            label="点击来添加提示",
                            type="numpy",
                            interactive=True,
                            height=500,
                        )
                        cutout_gallery = gr.Gallery(label="抠图结果 (Cutout Results)", preview=True, object_fit="contain", height="auto")

                # --- SAM Event Handlers ---
                demo.load(fn=load_model, inputs=[selected_model, gr.State(sam_device)], outputs=[predictor_state])
                load_model_btn.click(fn=load_model, inputs=[selected_model, gr.State(sam_device)], outputs=[predictor_state])
                
                upload_and_reset_outputs = [interactive_display, original_image_state, mask_state, history_state, cutout_gallery, predictor_state, box_start_state]
                
                source_image.upload(fn=set_image_for_predictor, inputs=[predictor_state, source_image], outputs=upload_and_reset_outputs)
                
                interactive_display.select(
                    fn=interactive_predict,
                    inputs=[predictor_state, original_image_state, history_state, mode_radio, box_start_state],
                    outputs=[interactive_display, mask_state, history_state, predictor_state, box_start_state]
                )
                
                cut_everything_btn.click(fn=generate_everything, inputs=[predictor_state, original_image_state], outputs=[interactive_display, cutout_gallery])
                cut_out_btn.click(fn=single_cutout, inputs=[original_image_state, mask_state], outputs=[cutout_gallery])
                reset_btn.click(fn=reset_all, inputs=[predictor_state, original_image_state], outputs=upload_and_reset_outputs)

        # Footer for warnings
        gr.HTML(f"""<div align="center">Activated Model - Shape Generation ({args.model_path}/{args.subfolder}) ; Texture Generation ({'Hunyuan3D-2' if HAS_TEXTUREGEN else 'Unavailable'})</div>""")
        if not HAS_TEXTUREGEN:
            gr.HTML("""<div style="margin-top: 5px;" align="center"><b>Warning: </b>Texture synthesis is disable due to missing requirements.</div>""")
        if not args.enable_t23d:
            gr.HTML("""<div style="margin-top: 5px;" align="center"><b>Warning: </b>Text to 3D is disable. To activate it, please run `python app.py --enable_t23d`.</div>""")
            
    return demo

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    args = parser.parse_args()

    # --- Global Variables & Constants Setup (from Hunyuan script) ---
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder
    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome!</p><p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()
    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    # --- Load Hunyuan Models/Workers ---
    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            if args.low_vram_mode: texgen_worker.enable_model_cpu_offload()
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e, "Failed to load texture generator.")
            HAS_TEXTUREGEN = False

    HAS_T2I = False
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=args.device)
        HAS_T2I = True

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh
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

    # --- FastAPI App Setup (from Hunyuan script) ---
    app = FastAPI()
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    if os.path.exists('./assets/env_maps'):
        shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)
    
    if args.low_vram_mode:
        torch.cuda.empty_cache()

    # --- Build and Mount the Combined Gradio App ---
    demo = build_combined_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    
    # --- Run the App ---
    uvicorn.run(app, host=args.host, port=args.port, workers=1)