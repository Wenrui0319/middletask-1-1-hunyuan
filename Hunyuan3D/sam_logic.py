# sam_logic.py

import gradio as gr
import numpy as np
import cv2
import torch
from skimage import color
from typing import Tuple, Dict, Any, List, Optional # 确保 List, Dict, Any 都已导入
import os
import tempfile
import time
from PIL import Image

sam_predictor_global = None

def initialize_sam(args):
    global sam_predictor_global
    if SAM_AVAILABLE:
        print("\n--- [SAM] 准备预加载SAM模型...")
        sam_model_dir = "models"
        if not os.path.exists(sam_model_dir):
            os.makedirs(sam_model_dir)
        available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
        if available_sam_models:
            default_sam_model = 'sam_vit_b_01ec64.pth' or available_sam_models[0]
            print(f"    > 找到默认SAM模型: {default_sam_model}")
            sam_predictor_global = load_sam_model(default_sam_model, args.device)
        else:
            print("    > 警告: 在 'models' 文件夹中未找到SAM模型。SAM功能将不可用，直到手动选择模型。")

try:
    from segment_anything import (
        SamAutomaticMaskGenerator,
        build_sam_vit_b,
        build_sam_vit_h,
        build_sam_vit_l,
        SamPredictor,
    )
    from segment_anything.utils.transforms import ResizeLongestSide
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def save_rgba_to_temp_png(rgba_array: np.ndarray) -> str:
    """将一个RGBA NumPy数组保存到一个临时的PNG文件，并返回其路径。"""
    if rgba_array is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
        
        pil_image = Image.fromarray(rgba_array, 'RGBA')
        pil_image.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"Error saving temp png: {e}")
        return None

def save_rgba_to_data_png(rgba_array: np.ndarray, filename: str) -> Optional[str]:
    """将一个RGBA NumPy数组保存到 'data/sam' 文件夹，并返回其路径。"""
    if rgba_array is None or not filename:
        return None

    data_sam_dir = os.path.abspath(os.path.join("data", "sam"))
    os.makedirs(data_sam_dir, exist_ok=True) 
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Filename contains invalid path characters.")
    if not filename.lower().endswith(".png"):
        filename += ".png"

    full_path = os.path.join(data_sam_dir, filename)
    
    if not os.path.abspath(full_path).startswith(data_sam_dir):
        raise ValueError("Attempted to save file outside of 'data/sam' directory.")

    try:
        pil_image = Image.fromarray(rgba_array, 'RGBA')
        pil_image.save(full_path)
        return full_path
    except Exception as e:
        gr.Error(f"保存图片至data/sam文件夹失败")
        return None

def handle_save_cutouts(image_list: List[Tuple[str, Any]]) -> str:
    if not image_list:
        gr.Warning("没有抠图结果可保存。")
        return None

    saved_count = 0
    for item in image_list:
        temp_file_path = item[0]
        try:
            img = Image.open(temp_file_path).convert("RGBA")
            rgba_array = np.array(img)
            
            original_basename = os.path.basename(temp_file_path)
            timestamp = int(time.time() * 1000)
            new_filename = f"cutout_{timestamp}.png"

            full_path = save_rgba_to_data_png(rgba_array, new_filename)
            if full_path:
                saved_count += 1
        except Exception as e:
            gr.Error(f"保存文件失败: {os.path.basename(temp_file_path)}. 错误信息: {e}")

    if saved_count > 0:
        gr.Info(f"成功保存 {saved_count} 个抠图到 data/sam 文件夹。")
        return None
    else:
        gr.Info(f"没有新的抠图被保存。")
        return None

def create_color_mask(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    if not annotations: return image
    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    if not sorted_anns: return image
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:
        image = image[..., :3]
    
    h, w, _ = image.shape
    mask_img = np.zeros((h, w), dtype=np.uint16)
    for i, ann in enumerate(sorted_anns):
        mask_img[ann["segmentation"]] = i + 1
    color_mask = color.label2rgb(mask_img, image, bg_label=0, kind='overlay')
    return (color_mask * 255).astype(np.uint8)

def cut_out_object(original_image: np.ndarray, mask: np.ndarray):
    """
    根据蒙版抠出物体，并将透明区域的RGB值清零。
    """
    if original_image is None or mask is None:
        gr.Warning("请先生成一个蒙版。")
        return None
    
    rgba_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = mask * 255
    
    transparent_pixels = rgba_image[:, :, 3] == 0
    rgba_image[transparent_pixels, :3] = 0
    
    return rgba_image

@torch.no_grad()
def generate_everything(predictor: SamPredictor, original_image: np.ndarray, progress=gr.Progress()) -> Tuple[np.ndarray, List[str]]:
    if original_image is None: raise gr.Error("请先上传一张图片。")
    if predictor is None: raise gr.Error("模型未加载。")
    progress(0.1, desc="正在运行全图自动分割...")
    generator = SamAutomaticMaskGenerator(model=predictor.model)
    annotations = generator.generate(original_image)
    progress(0.6, desc="正在创建彩色蒙版和抠图...")
    color_mask = create_color_mask(original_image, annotations)
    
    cutout_paths = []
    for ann in sorted(annotations, key=(lambda x: x['area']), reverse=True):
        rgba_array = cut_out_object(original_image, ann['segmentation'])
        if rgba_array is not None:
            temp_file_path = save_rgba_to_temp_png(rgba_array)
            if temp_file_path:
                cutout_paths.append(temp_file_path)
    progress(1, desc="完成！")
    return color_mask, cutout_paths

def load_sam_model(name: str, device: str, progress=gr.Progress()) -> SamPredictor:
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
        gr.Info(f"SAM模型已加载: {name}")
        progress(1, desc=f"SAM模型 {name} 加载成功！")
        return SamPredictor(model)
    except Exception as e:
        raise gr.Error(f"加载SAM模型失败: {e}")

def set_image_for_predictor(predictor: SamPredictor, image_np_in):
    if image_np_in is None:
        # When clearing, we just need to clear the states.
        return None, None, [], None, predictor, None

    # The incoming array might be RGBA from the image component, we need RGB for the predictor.
    if image_np_in.shape[2] == 4:
        image_rgb_np = image_np_in[..., :3]
    else:
        image_rgb_np = image_np_in

    if predictor is None:
        gr.Warning("SAM模型尚未加载。")
        # The first return value corresponds to sam_original_image_state
        return image_rgb_np, None, [], None, None, None
        
    try:
        predictor.set_image(image_rgb_np)
        gr.Info("图片已为SAM设置。")
        # Update states, but don't return a value for the interactive_display itself.
        return image_rgb_np, None, [], None, predictor, None
    except Exception as e:
        gr.Error(f"为SAM设置图片时出错: {e}")
        return image_rgb_np, None, [], None, predictor, None

def clear_sam_panel():
    return None, None, None, [], None, None

def visualize_prompts(image, history, current_mask=None, box_start_point=None):
    if image is None: return None
    vis_image = image.copy()
    if current_mask is not None:
        blue_mask = np.zeros_like(vis_image, dtype=np.uint8)
        blue_mask[current_mask] = [65, 105, 225]
        vis_image = cv2.addWeighted(vis_image, 1.0, blue_mask, 0.5, 0)
    
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
    
    if box_start_point is not None:
        pt = (int(box_start_point[0]), int(box_start_point[1]))
        cv2.circle(vis_image, pt, 8, (255, 255, 0), -1) 

    return vis_image

@torch.no_grad()
def interactive_predict(
    predictor: SamPredictor,
    original_image: np.ndarray,
    history: List,
    mode: str,
    box_start_state: List,
    evt: gr.SelectData
) -> Tuple[np.ndarray, np.ndarray, List, SamPredictor, List]:
    if predictor is None or original_image is None or not predictor.is_image_set:
        raise gr.Error("请先上传图片并确保SAM模型已加载！")

    if mode == "添加点 (Add Point)":
        new_prompt = {'type': 'point', 'data': (evt.index, 1)}
        history.append(new_prompt)
        box_start_state = None

    elif mode == "画框 (Draw Box)":
        if box_start_state is None:
            gr.Info("已选择第一个点，请点击对角点以完成画框。")
            box_start_state = evt.index
            vis_image = visualize_prompts(original_image, history, None, box_start_state)
            return vis_image, None, history, predictor, box_start_state
        else:
            box_end_point = evt.index
            history = [{'type': 'box', 'data': [box_start_state, box_end_point]}]
            box_start_state = None
            gr.Info("画框完成，正在分割...")

    point_coords, point_labels, box_coords = [], [], None
    for prompt in history:
        if prompt['type'] == 'point':
            point_coords.append(prompt['data'][0])
            point_labels.append(prompt['data'][1])
        elif prompt['type'] == 'box':
            box_coords = np.array(prompt['data'])

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

def single_cutout(image, mask):
    if image is None or mask is None:
        gr.Warning("请先通过点选或框选生成一个蒙版。")
        return None
    
    rgba_array = cut_out_object(image, mask)
    if rgba_array is None:
        return None
    
    temp_file_path = save_rgba_to_temp_png(rgba_array)
    
    return [temp_file_path] if temp_file_path else None

def reset_all_sam(predictor, original_image):
    if predictor is not None and original_image is not None:
        predictor.set_image(original_image)
        gr.Info("已重置。")
    return original_image, original_image, None, [], None, predictor, None

def create_sam_ui(sam_predictor_global):
    sam_predictor_state = gr.State(sam_predictor_global)
    sam_original_image_state = gr.State(None)
    sam_mask_state = gr.State(None)
    sam_history_state = gr.State([])
    sam_box_start_state = gr.State(None)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=250):
            sam_model_dir = "models"
            if not os.path.exists(sam_model_dir): os.makedirs(sam_model_dir)
            available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
            default_sam_model = available_sam_models[0] if available_sam_models else None
            
            with gr.Group():
                selected_model = gr.Dropdown(choices=available_sam_models, label="切换SAM模型 (可选)", value=default_sam_model)
            
            with gr.Group():
                mode_radio = gr.Radio(choices=["添加点 (Add Point)", "画框 (Draw Box)"], value="添加点 (Add Point)", label="工具 (Tool)")
                cut_everything_btn = gr.Button("分割所有物体")
                cut_out_btn = gr.Button("抠图")
                reset_btn = gr.Button("重置提示")
        
        with gr.Column(scale=5):
            interactive_display = gr.Image(label="交互式显示 (请先在左侧Image Prompt上传图片)", type="numpy", interactive=True, height=400, elem_id="sam_input_image")
            cutout_gallery = gr.Gallery(label="抠图结果", preview=True, object_fit="contain", height="auto")
            save_to_data_btn = gr.Button("保存抠图到数据文件夹")
    
    selected_model.change(fn=load_sam_model, inputs=[selected_model, gr.State('cuda')], outputs=[sam_predictor_state])
    
    interactive_display.select(
        fn=interactive_predict,
        inputs=[sam_predictor_state, sam_original_image_state, sam_history_state, mode_radio, sam_box_start_state],
        outputs=[interactive_display, sam_mask_state, sam_history_state, sam_predictor_state, sam_box_start_state]
    )
    
    cut_everything_btn.click(fn=generate_everything, inputs=[sam_predictor_state, sam_original_image_state], outputs=[interactive_display, cutout_gallery])
    cut_out_btn.click(fn=single_cutout, inputs=[sam_original_image_state, sam_mask_state], outputs=[cutout_gallery])
    reset_btn.click(fn=reset_all_sam, inputs=[sam_predictor_state, sam_original_image_state], outputs=[interactive_display, sam_original_image_state, sam_mask_state, sam_history_state, cutout_gallery, sam_predictor_state, sam_box_start_state])
    
    save_to_data_btn.click(
        fn=handle_save_cutouts,
        inputs=[cutout_gallery],
        outputs=[]
    )
    
    # Define outputs for setting a new image. Note that this list does NOT include
    # the interactive_display itself, to prevent an infinite update loop.
    # The interactive_display is updated by dispatch_image, and this .change event
    # is only for updating the internal states of the SAM model.
    sam_set_image_outputs = [sam_original_image_state, sam_mask_state, sam_history_state, cutout_gallery, sam_predictor_state, sam_box_start_state]
    
    interactive_display.change(
        fn=set_image_for_predictor,
        inputs=[sam_predictor_state, interactive_display],
        outputs=sam_set_image_outputs,
        show_progress="none"
    )

    return interactive_display
