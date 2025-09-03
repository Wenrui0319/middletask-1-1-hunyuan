# sam_logic.py

import gradio as gr
import numpy as np
import cv2
import torch
from skimage import color
from typing import Tuple, Dict, Any, List, Optional
import os
import tempfile
import time
from PIL import Image

sam_predictor_global = None

def check_sam_model_on_load():
    if sam_predictor_global is not None:
        gr.Info("SAM 模型已成功预加载，随时可用！")
    else:
        gr.Warning("警告：SAM 模型预加载失败。当您上传图片到SAM时，将自动尝试加载。")

def initialize_sam(args):
    global sam_predictor_global
    if SAM_AVAILABLE:
        print("\n--- [SAM] 准备预加载SAM模型...")
        sam_model_dir = "models"
        if not os.path.exists(sam_model_dir): os.makedirs(sam_model_dir)
        available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
        if available_sam_models:
            default_sam_model = 'sam_vit_h_4b8939.pth' if 'sam_vit_h_4b8939.pth' in available_sam_models else available_sam_models[0]
            try:
                print(f"    > 尝试预加载默认SAM模型: {default_sam_model} 到设备 {args.device}")
                sam_predictor_global = load_sam_model(default_sam_model, args.device)
            except Exception as e:
                print(f"    > SAM 模型预加载失败: {e}")
                sam_predictor_global = None
        else:
            print("    > 警告: 在 'models' 文件夹中未找到SAM模型。")

try:
    from segment_anything import SamAutomaticMaskGenerator, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

def save_rgba_to_temp_png(rgba_array: np.ndarray) -> str:
    if rgba_array is None: return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
        Image.fromarray(rgba_array, 'RGBA').save(temp_path)
        return temp_path
    except Exception as e:
        print(f"Error saving temp png: {e}")
        return None

def save_rgba_to_data_png(rgba_array: np.ndarray, filename: str) -> Optional[str]:
    if rgba_array is None or not filename: return None
    data_sam_dir = os.path.abspath(os.path.join("data", "sam"))
    os.makedirs(data_sam_dir, exist_ok=True) 
    if ".." in filename or "/" in filename or "\\" in filename: raise ValueError("Filename contains invalid path characters.")
    if not filename.lower().endswith(".png"): filename += ".png"
    full_path = os.path.join(data_sam_dir, filename)
    if not os.path.abspath(full_path).startswith(data_sam_dir): raise ValueError("Attempted to save file outside of 'data/sam' directory.")
    try:
        Image.fromarray(rgba_array, 'RGBA').save(full_path)
        return full_path
    except Exception as e:
        gr.Error(f"保存图片至data/sam文件夹失败")
        return None

def handle_save_cutouts(image_list: List[Tuple[str, Any]]) -> str:
    if not image_list:
        gr.Warning("没有抠图结果可保存。")
        return gr.FileExplorer()
    saved_count = 0
    for item in image_list:
        temp_file_path = item[0]
        try:
            img = Image.open(temp_file_path).convert("RGBA")
            rgba_array = np.array(img)
            timestamp = int(time.time() * 1000)
            new_filename = f"cutout_{timestamp}_{saved_count}.png"
            if save_rgba_to_data_png(rgba_array, new_filename): saved_count += 1
        except Exception as e:
            gr.Error(f"保存文件失败: {os.path.basename(temp_file_path)}. 错误信息: {e}")
    if saved_count > 0: gr.Info(f"成功保存 {saved_count} 个抠图到 data/sam 文件夹。")
    else: gr.Info(f"没有新的抠图被保存。")
    return gr.FileExplorer(key=str(time.time()))

def create_color_mask(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    if not annotations: return image
    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    if not sorted_anns: return image
    if image.ndim == 2: image = color.gray2rgb(image)
    elif image.shape[2] == 4: image = image[..., :3]
    h, w, _ = image.shape
    mask_img = np.zeros((h, w), dtype=np.uint16)
    for i, ann in enumerate(sorted_anns): mask_img[ann["segmentation"]] = i + 1
    color_mask = color.label2rgb(mask_img, image, bg_label=0, kind='overlay')
    return (color_mask * 255).astype(np.uint8)

def cut_out_object(original_image: np.ndarray, mask: np.ndarray):
    if original_image is None or mask is None:
        gr.Warning("请先生成一个蒙版。")
        return None
    rgba_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = mask * 255
    rgba_image[rgba_image[:, :, 3] == 0, :3] = 0
    return rgba_image

@torch.no_grad()
def generate_everything(predictor: SamPredictor, original_image: np.ndarray, progress=gr.Progress()) -> Tuple:
    if original_image is None: return "请先上传图片", None, None
    if predictor is None: return "模型未加载，无法分割", original_image, None
    progress(0.1, desc="正在运行全图自动分割...")
    generator = SamAutomaticMaskGenerator(model=predictor.model)
    annotations = generator.generate(original_image)
    progress(0.6, desc="正在创建彩色蒙版和抠图...")
    color_mask = create_color_mask(original_image, annotations)
    cutout_paths = [save_rgba_to_temp_png(cut_out_object(original_image, ann['segmentation'])) for ann in sorted(annotations, key=(lambda x: x['area']), reverse=True)]
    progress(1, desc="完成！")
    return "分割完成！", color_mask, [p for p in cutout_paths if p]

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
        gr.Info(f"SAM模型已加载到 {device}: {name}")
        progress(1, desc=f"SAM模型 {name} 加载成功！")
        return SamPredictor(model)
    except Exception as e:
        raise gr.Error(f"加载SAM模型失败: {e}")

def set_image_for_predictor(predictor, image_np_in):
    if image_np_in is None: return "请上传图片", None, None, [], None, None, None
    image_rgb_np = image_np_in[..., :3] if image_np_in.shape[2] == 4 else image_np_in
    if predictor is None: return "SAM模型尚未加载", image_rgb_np, None, [], None, None, None
    try:
        predictor.set_image(image_rgb_np)
        return "图片已加载，请开始交互", image_rgb_np, image_rgb_np, None, [], predictor, None
    except Exception as e:
        gr.Error(f"为SAM设置图片时出错: {e}")
        return f"错误: {e}", image_rgb_np, image_rgb_np, None, [], predictor, None

def visualize_prompts(image, history, current_mask=None, box_start_point=None):
    if image is None: return None
    vis_image = image.copy()
    if current_mask is not None:
        blue_mask = np.zeros_like(vis_image, dtype=np.uint8)
        blue_mask[current_mask] = [65, 105, 255]
        vis_image = cv2.addWeighted(vis_image, 1.0, blue_mask, 0.5, 0)
    for prompt in history:
        if prompt['type'] == 'box':
            p1_coords, p2_coords = prompt['data']
            pt1 = (int(p1_coords[0]), int(p1_coords[1]))
            pt2 = (int(p2_coords[0]), int(p2_coords[1]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 2)
        elif prompt['type'] == 'point':
            point, label = prompt['data']
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(vis_image, (int(point[0]), int(point[1])), 8, color, -1)
    if box_start_point is not None:
        cv2.circle(vis_image, (int(box_start_point[0]), int(box_start_point[1])), 8, (255, 255, 0), -1) 
    return vis_image

@torch.no_grad()
def interactive_predict(predictor: SamPredictor, original_image: np.ndarray, history: List, mode: str, box_start_state: List, evt: gr.SelectData) -> Tuple:
    if predictor is None or original_image is None or not predictor.is_image_set:
        return "请先上传图片并加载模型", visualize_prompts(original_image, history, None, box_start_state), None, history, predictor, box_start_state
    
    status_msg = "处理中..."
    if mode == "添加点 (Add Point)":
        history.append({'type': 'point', 'data': (evt.index, 1)})
        box_start_state = None
        status_msg = "添加点成功, 正在分割..."
    elif mode == "画框 (Draw Box)":
        if box_start_state is None:
            status_msg = "已选择第一个点，请点击对角点以完成画框。"
            box_start_state = evt.index
            return status_msg, visualize_prompts(original_image, history, None, box_start_state), None, history, predictor, box_start_state
        else:
            history = [{'type': 'box', 'data': [box_start_state, evt.index]}]
            box_start_state = None
            status_msg = "画框完成，正在分割..."

    point_coords, point_labels, box_coords = [], [], None
    for p in history:
        if p['type'] == 'point': point_coords.append(p['data'][0]); point_labels.append(p['data'][1])
        elif p['type'] == 'box': box_coords = np.array(p['data'])
    
    masks, scores, _ = predictor.predict(point_coords=np.array(point_coords) if point_coords else None, point_labels=np.array(point_labels) if point_labels else None, box=box_coords, multimask_output=True)
    best_mask = masks[np.argmax(scores)]
    vis_image = visualize_prompts(original_image, history, best_mask, box_start_state)
    return "分割完成！", vis_image, best_mask, history, predictor, box_start_state

def single_cutout(image, mask):
    if image is None or mask is None:
        return None, "无蒙版可抠图"
    rgba_array = cut_out_object(image, mask)
    return [save_rgba_to_temp_png(rgba_array)] if rgba_array is not None else None, "抠图完成"

def reset_all_sam(original_image):
    if original_image is not None: gr.Info("提示已重置。")
    return "提示已重置", original_image, None, [], None, None

def create_sam_ui(sam_predictor_global, file_explorer, device):
    sam_predictor_state = gr.State(sam_predictor_global)
    sam_original_image_state = gr.State(None)
    sam_mask_state = gr.State(None)
    sam_history_state = gr.State([])
    sam_box_start_state = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group():
                # --- START: 核心修改 ---
                # 这个组件现在可以接收文件路径（来自“编辑”）或Numpy数组（来自直接上传）
                input_image = gr.Image(label="上传并预览图片", type="numpy", sources=["upload", "clipboard"], height=250, elem_id="sam_input_image_upload")
                # --- END: 核心修改 ---
            status_text = gr.Textbox(label="状态 (Status)", value="就绪 (Ready)", interactive=False)
            with gr.Group():
                sam_model_dir = "models"
                if not os.path.exists(sam_model_dir): os.makedirs(sam_model_dir)
                available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
                default_sam_model = available_sam_models[0] if available_sam_models else None
                selected_model = gr.Dropdown(choices=available_sam_models, label="切换SAM模型 (可选)", value=default_sam_model)
            with gr.Group():
                mode_radio = gr.Radio(choices=["添加点 (Add Point)", "画框 (Draw Box)"], value="添加点 (Add Point)", label="工具 (Tool)")
            with gr.Group():
                cut_everything_btn = gr.Button("分割所有物体")
                cut_out_btn = gr.Button("抠图")
                reset_btn = gr.Button("重置提示")
        with gr.Column(scale=2):
            interactive_display = gr.Image(label="工作区 (在此点击图片进行分割)", type="numpy", interactive=True, height=700, elem_id="sam_interactive_display")
            with gr.Group():
                cutout_gallery = gr.Gallery(label="抠图结果", preview=True, object_fit="contain", height="auto")
                save_to_data_btn = gr.Button("保存至工作区")
    
    def load_model_if_needed_and_set_image(image_data, predictor, selected_model_name, device_name):
        # --- START: 核心修改 ---
        # image_data 现在可能是 Numpy 数组或文件路径字符串
        if image_data is None:
            return "请上传图片", None, None, None, [], None, None, None
        
        # 如果传入的是文件路径，先用PIL加载成Numpy数组
        image_np = None
        if isinstance(image_data, str):
            try:
                image_np = np.array(Image.open(image_data))
            except Exception as e:
                gr.Error(f"从路径加载图片失败: {e}")
                return f"错误: 加载图片失败", None, None, None, [], None, None, None
        else: # 否则，它就是Numpy数组
            image_np = image_data
        # --- END: 核心修改 ---

        current_predictor = predictor
        
        if current_predictor is None:
            try:
                gr.Info(f"SAM模型未加载，正在自动加载 {selected_model_name}...")
                current_predictor = load_sam_model(selected_model_name, device_name)
            except Exception as e:
                return f"错误: 模型加载失败!", None, None, None, [], None, None, None

        status, original_img, interactive_img, mask, history, final_predictor, box_start = set_image_for_predictor(current_predictor, image_np)
        return status, original_img, interactive_img, mask, history, final_predictor, box_start, gr.update(value=None)

    outputs_for_new_image = [status_text, sam_original_image_state, interactive_display, sam_mask_state, sam_history_state, sam_predictor_state, sam_box_start_state, cutout_gallery]
    input_image.change(
        fn=load_model_if_needed_and_set_image,
        inputs=[input_image, sam_predictor_state, selected_model, gr.State(device)],
        outputs=outputs_for_new_image
    )

    selected_model.change(fn=load_sam_model, inputs=[selected_model, gr.State(device)], outputs=[sam_predictor_state])
    
    interactive_predict_outputs = [status_text, interactive_display, sam_mask_state, sam_history_state, sam_predictor_state, sam_box_start_state]
    interactive_display.select(fn=interactive_predict, inputs=[sam_predictor_state, sam_original_image_state, sam_history_state, mode_radio, sam_box_start_state], outputs=interactive_predict_outputs)
    
    generate_outputs = [status_text, interactive_display, cutout_gallery]
    cut_everything_btn.click(fn=generate_everything, inputs=[sam_predictor_state, sam_original_image_state], outputs=generate_outputs)
    
    cut_out_outputs = [cutout_gallery, status_text]
    cut_out_btn.click(fn=single_cutout, inputs=[sam_original_image_state, sam_mask_state], outputs=cut_out_outputs)
    
    reset_outputs = [status_text, interactive_display, sam_mask_state, sam_history_state, cutout_gallery, sam_box_start_state]
    reset_btn.click(fn=reset_all_sam, inputs=[sam_original_image_state], outputs=reset_outputs)
    
    save_to_data_btn.click(fn=handle_save_cutouts, inputs=[cutout_gallery], outputs=[file_explorer])
    
    return input_image