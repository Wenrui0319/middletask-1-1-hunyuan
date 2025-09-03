import gradio as gr
import json
import uuid
import asyncio
import websockets
import requests
from PIL import Image, ImageOps
import io
import random
import numpy as np
import os
from urllib.parse import urlparse, urljoin

# --- 1. ComfyUI Server Configuration ---
COMFYUI_URL = "http://localhost:8188/"
COMFYUI_CLIENT_ID = str(uuid.uuid4())
COMFYUI_PROMPT_FILE = "Qwen+Image+Inapint模型局部重绘Desperate.json"

# --- 2. Backend Logic ---

async def queue_prompt_and_track_progress(prompt_workflow, client_id, server_address):
    prompt_url = get_server_address(server_address)
    ws_url = f"ws://{urlparse(server_address).netloc}/ws?clientId={client_id}"
    req = requests.post(prompt_url, json={'prompt': prompt_workflow, 'client_id': client_id})
    req.raise_for_status()
    prompt_id = req.json()['prompt_id']

    async with websockets.connect(ws_url) as websocket:
        while True:
            try:
                out = await websocket.recv()
                message = json.loads(out)
                if message['type'] == 'status':
                    data = message['data']['status']
                    yield f"队列: {data.get('exec_info', {}).get('queue_remaining', 0)}", None, None, gr.update(interactive=False), gr.update(interactive=False)
                elif message['type'] == 'execution_start':
                    if message['data']['prompt_id'] == prompt_id:
                        yield "开始执行...", None, None, gr.update(interactive=False), gr.update(interactive=False)
                elif message['type'] == 'executing':
                    if message['data'].get('node') is None and message['data']['prompt_id'] == prompt_id:
                        break
                elif message['type'] == 'progress':
                    data = message['data']
                    yield f"节点 {data.get('node', 'N/A')}: {data['value']}/{data['max']}", None, None, gr.update(interactive=False), gr.update(interactive=False)
                elif message['type'] == 'execution_error':
                    yield f"错误: {message['data']['exception_message']}", None, None, gr.update(interactive=True), gr.update(interactive=True)
                    return
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                yield f"错误: {e}", None, None, gr.update(interactive=True), gr.update(interactive=True)
                break

    history_url = urljoin(server_address, f"history/{prompt_id}")
    history_resp = requests.get(history_url)
    history_resp.raise_for_status()
    history = history_resp.json().get(prompt_id, {})
    outputs = history.get('outputs', {})

    def fetch_image(node_id):
        node_output = outputs.get(node_id, {})
        if 'images' in node_output and node_output['images']:
            image_data = node_output['images'][0]
            image_url = urljoin(server_address, f"view?filename={image_data['filename']}&subfolder={image_data.get('subfolder', '')}&type={image_data['type']}")
            response = requests.get(image_url)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
        return None

    result_image = fetch_image('60')
    comparison_image = fetch_image('76')
    yield "完成!", result_image, comparison_image, gr.update(interactive=True), gr.update(interactive=True)

def get_server_address(server_address):
    return urljoin(server_address, "prompt")

def upload_image(image: Image.Image, server_address: str, filename_prefix: str):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    files = {'image': (f"{filename_prefix}_{uuid.uuid4()}.png", buffer, 'image/png')}
    data = {'overwrite': 'true', 'type': 'input'}
    upload_url = urljoin(server_address, "upload/image")
    response = requests.post(upload_url, files=files, data=data)
    response.raise_for_status()
    return response.json()['name']



async def run_generation(prompt, neg_prompt, input_img, sampler, scheduler, steps, cfg, denoise, shift, strength):
    """
    通用生成函数
    """
    try:
        yield "处理中...", None, None, gr.update(interactive=False), gr.update(interactive=False)
        if not input_img or 'background' not in input_img:
            raise ValueError("请上传图片")
        
        seed = random.randint(0, 2**32 - 1)

        base_image = Image.fromarray(input_img['background'])

        # Inpainting: 使用用户绘制的蒙版
        if 'layers' not in input_img or not input_img['layers']:
                raise ValueError("请在图像上绘制蒙版以进行 Inpainting")
        mask_rgba_layer = Image.fromarray(input_img['layers'][0])
        mask_image = mask_rgba_layer.getchannel('A')
        if not np.any(np.array(mask_image)):
            raise gr.Error("蒙版是空的！请在图像上绘制需要修复的区域。")

        yield "上传图像...", None, None, gr.update(interactive=False), gr.update(interactive=False)
        image_filename = upload_image(base_image, COMFYUI_URL, "base_image")

        yield "上传蒙版...", None, None, gr.update(interactive=False), gr.update(interactive=False)
        mask_filename = upload_image(mask_image, COMFYUI_URL, "mask_image")

        with open(COMFYUI_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_workflow = json.load(f)

        # 更新工作流参数
        prompt_workflow["3"]["inputs"]["seed"] = seed
        prompt_workflow["3"]["inputs"]["steps"] = steps
        prompt_workflow["3"]["inputs"]["cfg"] = cfg
        prompt_workflow["3"]["inputs"]["sampler_name"] = sampler
        prompt_workflow["3"]["inputs"]["scheduler"] = scheduler
        prompt_workflow["3"]["inputs"]["denoise"] = denoise
        prompt_workflow["79"]["inputs"]["text"] = prompt
        prompt_workflow["7"]["inputs"]["text"] = neg_prompt
        prompt_workflow["66"]["inputs"]["shift"] = shift
        prompt_workflow["69"]["inputs"]["strength"] = strength
        prompt_workflow["71"]["inputs"]["image"] = image_filename
        prompt_workflow["110"]["inputs"]["image"] = mask_filename
        prompt_workflow["110"]["inputs"]["channel"] = "red"

        async for status, res_img, comp_img, btn_state, _ in queue_prompt_and_track_progress(prompt_workflow, COMFYUI_CLIENT_ID, COMFYUI_URL):
            yield status, res_img, comp_img, btn_state, btn_state

    except Exception as e:
        yield f"错误: {str(e)}", None, None, gr.update(interactive=True), gr.update(interactive=True)



import time

def save_image_to_workspace(image, subfolder, prefix):
    if image is None:
        gr.Warning("没有图像可保存。")
        return gr.FileExplorer(key=str(time.time()))

    try:
        save_dir = os.path.join("data", subfolder)
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        
        save_path = os.path.join(save_dir, filename)
        if not os.path.abspath(save_path).startswith(os.path.abspath(save_dir)):
            raise ValueError("检测到不安全的保存路径。")
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image.save(save_path)
        
        gr.Info(f"图像已保存至 {save_path}")
        return gr.FileExplorer(key=str(time.time()))
        
    except Exception as e:
        gr.Error(f"保存图像失败: {e}")
        return gr.FileExplorer(key=str(time.time()))

def process_and_save_masked_image(image_dict):
    if image_dict is None or 'background' not in image_dict or image_dict['background'] is None:
        gr.Warning("没有图像可处理。")
        return gr.FileExplorer(key=str(time.time()))
        
    base_image_pil = Image.fromarray(image_dict['background'])
    
    output_image = base_image_pil.convert("RGBA")
    alpha = Image.new('L', output_image.size, 0)
    output_image.putalpha(alpha)

    if 'layers' in image_dict and image_dict['layers']:
        mask_layer = Image.fromarray(image_dict['layers'][0])
        if mask_layer.mode == 'RGBA':
            mask = mask_layer.getchannel('A')
            if np.any(np.array(mask)):
                output_image.paste(base_image_pil, (0,0), mask=mask)
                output_image.putalpha(mask)

    if not np.any(np.array(output_image.getchannel('A'))):
        output_image = output_image.convert("RGB")

    return save_image_to_workspace(output_image, "qwen_inpainting", "masked")

def upload_masked_image(file_obj):
    if file_obj is None:
        return gr.update()
        
    try:
        uploaded_image = Image.open(file_obj.name)
    except Exception as e:
        raise gr.Error(f"无法打开图像文件: {e}")

    # 检查图像是否有Alpha通道
    if uploaded_image.mode == 'RGBA':
        background_img = uploaded_image.convert("RGB")
        mask_np = np.array(uploaded_image.getchannel('A'))
        
        # 使用固定的红色来可视化遮罩
        color = np.array([255, 0, 0], dtype=np.uint8)
        layer_np = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        layer_np[mask_np > 0, :3] = color
        layer_np[..., 3] = mask_np
        
        layers = [layer_np]
    else:
        # 如果是RGB图像，则没有遮罩
        background_img = uploaded_image.convert("RGB")
        # 创建一个空的图层
        layer_np = np.zeros((background_img.height, background_img.width, 4), dtype=np.uint8)
        layers = [layer_np]

    return {
        "background": np.array(background_img),
        "layers": layers,
        "composite": None
    }


# --- 3. Gradio UI Layout ---
def create_qwen_inpainting_ui(file_explorer):
    with gr.Blocks(theme=gr.themes.Base(), analytics_enabled=False, css=".info-icon {display: flex; align-items: center; justify-content: center;}") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                with gr.Group():
                    prompt = gr.Textbox(label="正面提示词 (Positive Prompt)", lines=2, value="A beautiful girl")
                    neg_prompt = gr.Textbox(label="负面提示词 (Negative Prompt)", lines=2, value="(bad anatomy, deformed, distorted, disfigured:1.6), blurry, low quality")
                
                gr.Markdown("### 参数设置")
                with gr.Tabs():
                    with gr.Tab("基础参数"):
                        with gr.Group():
                            with gr.Row(variant='compact', equal_height=True):
                                denoise_slider = gr.Slider(label="Denoise", minimum=0.0, maximum=1.0, step=0.05, value=0.85, scale=9)
                                gr.Markdown("[ⓘ](### '控制重绘区域与原图的差异程度。\nInpainting建议: 0.7-0.9 | 图像编辑建议: 0.9-1.0')", elem_classes=["info-icon"])
                            with gr.Row(variant='compact', equal_height=True):
                                steps_slider = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=10, scale=9)
                                gr.Markdown("[ⓘ](### '生成步数，步数越高细节越多，但速度越慢。建议值: 10左右')", elem_classes=["info-icon"])
                            with gr.Row(variant='compact', equal_height=True):
                                cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.1, value=4.5, scale=9)
                                gr.Markdown("[ⓘ](### '提示词相关性，值越高越遵循提示词，但可能过曝。建议值: 3.0-8.0')", elem_classes=["info-icon"])
                    with gr.Tab("高级参数"):
                        with gr.Group():
                            with gr.Row(variant='compact', equal_height=True):
                                sampler_dropdown = gr.Dropdown(label="Sampler", choices=["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"], value="euler", scale=9)
                                gr.Markdown("[ⓘ](### '采样器，影响图像生成风格和速度。')", elem_classes=["info-icon"])
                            with gr.Row(variant='compact', equal_height=True):
                                scheduler_dropdown = gr.Dropdown(label="Scheduler", choices=["normal", "karras", "simple"], value="simple", scale=9)
                                gr.Markdown("[ⓘ](### '调度器，与采样器配合使用，影响细节和收敛速度。')", elem_classes=["info-icon"])
                            with gr.Row(variant='compact', equal_height=True):
                                shift_slider = gr.Slider(label="Shift (AuraFlow)", minimum=1.0, maximum=5.0, step=0.1, value=3.1, scale=9)
                                gr.Markdown("[ⓘ](### 'AuraFlow特定参数，控制生成光环效果的强度。')", elem_classes=["info-icon"])
                            with gr.Row(variant='compact', equal_height=True):
                                strength_slider = gr.Slider(label="Strength (ControlNet)", minimum=0.0, maximum=1.0, step=0.05, value=1.0, scale=9)
                                gr.Markdown("[ⓘ](### 'ControlNet强度，控制模型遵循边缘、深度等条件的程度。')", elem_classes=["info-icon"])
                with gr.Row():
                    inpaint_btn = gr.Button("局部重绘 (Inpainting)", variant="primary", size="lg", scale=1)
    
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.TabItem("重绘区域 (Inpaint Area)"):
                        input_img = gr.ImageEditor(label="上传并编辑图像 (Upload and Edit Image)", interactive=True, height=700, type="numpy", elem_id="qwen_inpainting_input_image")
                        with gr.Row():
                            save_mask_btn = gr.Button("保存遮罩图像至工作区")
                            upload_mask_btn = gr.UploadButton("上传含遮罩图像", file_types=["image"], visible=False)
                    with gr.TabItem("生成图 (Result)"):
                        output_img = gr.Image(label="生成结果 (Generated Result)", interactive=False, height=700, type="pil")
                        save_result_btn = gr.Button("保存局部重绘图像至工作区")
                    with gr.TabItem("对比图 (Comparison)"):
                        comparison_img = gr.Image(label="拼接对比图 (Stitched Comparison)", interactive=False, height=700)
    
        all_inputs = [prompt, neg_prompt, input_img, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, denoise_slider, shift_slider, strength_slider]
        all_outputs = [status_text, output_img, comparison_img, inpaint_btn, inpaint_btn]
    
        inpaint_btn.click(fn=run_generation, inputs=all_inputs, outputs=all_outputs)
    
        # Handlers for mask import/export and saving results
        upload_mask_btn.upload(
            fn=upload_masked_image,
            inputs=[upload_mask_btn],
            outputs=[input_img],
            queue=False
        )

        save_mask_btn.click(
            fn=process_and_save_masked_image,
            inputs=[input_img],
            outputs=[file_explorer]
        )
        
        save_result_btn.click(
            fn=lambda img: save_image_to_workspace(img, "qwen_inpainting", "inpainted"),
            inputs=[output_img],
            outputs=[file_explorer]
        )
    return input_img