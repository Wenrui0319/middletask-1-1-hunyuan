import os
import cv2
import torch
import time
import logging
import numpy as np
from typing import Tuple, Dict, Any, List, Union
from pathlib import Path
from skimage import color
from segment_anything import (
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    SamPredictor,
)
from PIL import Image
import gradio as gr
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# --- 配置与初始化 ---
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
os.makedirs(temp_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = temp_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#加载混元模型，初次需要下载模型，建议调试时先注释，因为加载需要较久时间，在点击生成按钮前不会报错
model_path = 'tencent/Hunyuan3D-2'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
print("Init hunyuan finished")

# --- 后端函数 (这部分逻辑已验证，无需修改) ---

def create_color_mask(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    if not annotations: return image
    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    if not sorted_anns: return image
    h, w = sorted_anns[0]["segmentation"].shape
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
    model_type = next((t for t in model_builder if t in name), None)
    if model_type is None: raise ValueError(f"无效的模型文件名: {name}")

    model = model_builder[model_type](checkpoint=checkpoint_path)
    model.to(device)
    logger.info(f"模型已加载: {name}")
    progress(1, desc=f"模型 {name} 加载成功！")
    return SamPredictor(model)

def set_image_for_predictor(predictor: SamPredictor, image: np.ndarray):
    """当上传新图片时，为 predictor 设置它，并返回更新后的 predictor 和其他重置状态。"""
    if predictor is None:
        gr.Warning("模型尚未加载。")
        return None, None, None, [], None, None
    if image is not None:
        predictor.set_image(image)
        gr.Info("图片已设置。")
    return image, image, None, [], None, predictor

def visualize_prompts(image, history, current_mask=None):
    vis_image = image.copy()
    if current_mask is not None:
        blue_mask = np.zeros_like(vis_image, dtype=np.uint8)
        blue_mask[current_mask] = [65, 105, 225]
        vis_image = cv2.addWeighted(vis_image, 1.0, blue_mask, 0.5, 0)
    for prompt in history:
        if prompt['type'] == 'box':
            box = prompt['data']
            cv2.rectangle(vis_image, box[0], box[1], (0, 255, 0), 2)
        elif prompt['type'] == 'point':
            point, label = prompt['data']
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(vis_image, point, 8, color, -1)
    return vis_image

@torch.no_grad()
def interactive_predict(
    predictor: SamPredictor, 
    original_image: np.ndarray,
    history: List,
    mode: str,
    evt: gr.SelectData 
) -> Tuple[np.ndarray, np.ndarray, List, SamPredictor]:
    if predictor is None or original_image is None or not predictor.is_image_set:
        raise gr.Error("请先上传图片并确保模型已加载！")

    if mode == "Box":
        history = [{'type': 'box', 'data': [evt.index[0], evt.index[1]]}]
    elif mode == "Hover & Click":
        new_prompt = {'type': 'point', 'data': (evt.index, 1)}
        history.append(new_prompt)

    point_coords, point_labels, box_coords = [], [], None
    for prompt in history:
        if prompt['type'] == 'point':
            point_coords.append(prompt['data'][0])
            point_labels.append(prompt['data'][1])
        elif prompt['type'] == 'box':
            box_coords = np.array(prompt['data'])

    point_coords = np.array(point_coords) if point_coords else None
    point_labels = np.array(point_labels) if point_labels else None

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_coords,
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]
    vis_image = visualize_prompts(original_image, history, best_mask)
    return vis_image, best_mask, history, predictor



#获得gallery中当前预览的图片
def get_gallery_image(evt: gr.SelectData):
    
    if evt is None:
        return None
        
    value = evt.value
    if isinstance(value, np.ndarray):
        image = value
        print(f"成功获取图片！类型是 NumPy 数组。形状是: {image.shape}")
    elif isinstance(value, dict) and 'image' in value and 'path' in value['image']:
        image_path = value['image']['path']
        print(f"从路径加载图片: {image_path}")
        try:
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            print(f"从路径加载并转换为 NumPy 数组成功！形状是: {image.shape}")
        except Exception as e:
            print(f"ERROR: 从路径加载图片失败: {e}")
            image = None
    else:
        print(f"ERROR: 无法处理 evt.value 的类型: {type(value)}")
        image = None
    
    return image

def get_mesh_from_mask(masked_image: np.ndarray, progress=gr.Progress()):
    if masked_image is None:
        progress(0.5, desc="未提供待生成的图片")
        return None, "未提供待生成的图片"

    pil_image = Image.fromarray(masked_image)
    progress(0.5, desc="正在生成 3D 网格...")
    mesh = pipeline_shapegen(image=pil_image)[0]
    timestamp = int(time.time())
    output_filename = f"output_{timestamp}.glb"
    output_path = os.path.join(temp_dir, output_filename)
    progress(0.9, desc="正在导出网格文件...")
    mesh.export(output_path)
    progress(1.0, desc="生成成功")
    return output_path, "生成成功"

# --- UI 布局 ---
model_dir = "models"
if not os.path.exists(model_dir): os.makedirs(model_dir)
available_models = [x for x in os.listdir(model_dir) if x.endswith(".pth")]
default_model = available_models[0] if available_models else None
device = "cuda" if torch.cuda.is_available() else "cpu"

with gr.Blocks() as application:
    gr.Markdown("# Segment Anything 交互式演示")
    
    predictor_state = gr.State(None)
    original_image_state = gr.State(None)
    mask_state = gr.State(None)
    history_state = gr.State([])

    with gr.Row():
        selected_model = gr.Dropdown(choices=available_models, label="模型", value=default_model)
        load_model_btn = gr.Button("加载模型")
    
    with gr.Tab("交互式分割"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=250):
                with gr.Group():
                    input_image_upload = gr.Image(label="上传图片", type="numpy")
                with gr.Group():
                    mode_radio = gr.Radio(choices=["Hover & Click", "Box"], value="Hover & Click", label="工具")
                    cut_everything_btn = gr.Button("分割所有物体 (Cut Everything)")
                    cut_out_btn = gr.Button("抠图 (Cut out object)")
                    reset_btn = gr.Button("重置 (Reset)")

            with gr.Column(scale=3):
                interactive_display = gr.Image(label="点击或画框来添加提示", type="numpy", interactive=True, height=500)
                cutout_gallery = gr.Gallery(label="抠图结果", preview=True, object_fit="contain", height="auto")

        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 3D 网格生成")
                gr.Markdown("此功能将基于当前的分割蒙版生成一个简单的3D模型。")
                generate_mesh_btn = gr.Button("生成 3D 模型")
                mesh_status_label = gr.Label(label="状态")
                image_to_generate_3d = gr.Image(label="待生成图片", type="numpy")
            with gr.Column(scale=3):
                model_3d_display = gr.Model3D(label="3D 模型预览", clear_color=[255, 255, 255, 0], interactive=True)
    # --- 事件监听逻辑 ---
    
    application.load(fn=load_model, inputs=[selected_model, gr.State(device)], outputs=[predictor_state])
    load_model_btn.click(fn=load_model, inputs=[selected_model, gr.State(device)], outputs=[predictor_state])
    
    upload_outputs = [interactive_display, original_image_state, mask_state, history_state, cutout_gallery, predictor_state]
    input_image_upload.upload(
        fn=set_image_for_predictor, 
        inputs=[predictor_state, input_image_upload], 
        outputs=upload_outputs
    )

    interactive_display.select(
        fn=interactive_predict,
        inputs=[predictor_state, original_image_state, history_state, mode_radio],
        outputs=[interactive_display, mask_state, history_state, predictor_state]
    )
    
    cut_everything_btn.click(
        fn=generate_everything,
        inputs=[predictor_state, original_image_state],
        outputs=[interactive_display, cutout_gallery]
    )

    def single_cutout(image, mask):
        if image is None or mask is None:
            gr.Warning("请先通过点选或框选生成一个蒙版。")
            return None
        return [cut_out_object(image, mask)]

    cut_out_btn.click(
        fn=single_cutout,
        inputs=[original_image_state, mask_state],
        outputs=[cutout_gallery]
    )
    
    # 最终修正点 1: 重构 reset_all 函数
    def reset_all(predictor, original_image):
        """
        一个真正能重置所有状态的函数。
        """
        if predictor is not None and original_image is not None:
            # 重新在 predictor 中设置图片，这是最关键的一步
            predictor.set_image(original_image)
            gr.Info("已重置。")
        
        # 返回所有需要被重置的 UI 组件和状态
        # (原始图片, 原始图片, 无蒙版, 空历史, 空抠图区, 更新后的 predictor)
        return original_image, original_image, None, [], None, predictor

    # 最终修正点 2: 更新 reset_btn 的事件
    reset_btn.click(
        fn=reset_all,
        inputs=[predictor_state, original_image_state],
        outputs=upload_outputs # 使用和上传时完全相同的 outputs 列表
    )
    #当点击时，获得选择的图片
    cutout_gallery.select(
        fn=get_gallery_image,
        inputs=None,
        outputs=[image_to_generate_3d]
    )
    #调用混元生成mesh
    generate_mesh_btn.click(
        fn=get_mesh_from_mask,
        inputs=[image_to_generate_3d],
        outputs=[model_3d_display, mesh_status_label]
    )

application.launch(server_name="0.0.0.0", server_port=7860)