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
COMFYUI_PROMPT_FILE = "image_qwen_image_edit_api.json"

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
                    yield f"队列: {data.get('exec_info', {}).get('queue_remaining', 0)}", None, None, gr.update(interactive=False)
                elif message['type'] == 'execution_start':
                    if message['data']['prompt_id'] == prompt_id:
                        yield "开始执行...", None, None, gr.update(interactive=False)
                elif message['type'] == 'executing':
                    if message['data'].get('node') is None and message['data']['prompt_id'] == prompt_id:
                        break
                elif message['type'] == 'progress':
                    data = message['data']
                    yield f"节点 {data.get('node', 'N/A')}: {data['value']}/{data['max']}", None, None, gr.update(interactive=False)
                elif message['type'] == 'execution_error':
                    yield f"错误: {message['data']['exception_message']}", None, None, gr.update(interactive=True)
                    return
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                yield f"错误: {e}", None, None, gr.update(interactive=True)
                return

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
    comparison_image = fetch_image('102')
    yield "完成!", result_image, comparison_image, gr.update(interactive=True)

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



async def run_generation(prompt, neg_prompt, input_img, sampler, scheduler, steps, cfg, denoise, shift):
    """
    通用生成函数，用于图像编辑。
    """
    try:
        yield "处理中...", None, None, gr.update(interactive=False)
        if input_img is None:
            raise ValueError("请上传图片")

        seed = random.randint(0, 2**32 - 1)
        base_image = Image.fromarray(input_img)

        yield "上传图像...", None, None, gr.update(interactive=False)
        image_filename = upload_image(base_image, COMFYUI_URL, "base_image")

        with open(COMFYUI_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_workflow = json.load(f)

        # 更新工作流参数
        prompt_workflow["3"]["inputs"]["seed"] = seed
        prompt_workflow["3"]["inputs"]["steps"] = steps
        prompt_workflow["3"]["inputs"]["cfg"] = cfg
        prompt_workflow["3"]["inputs"]["sampler_name"] = sampler
        prompt_workflow["3"]["inputs"]["scheduler"] = scheduler
        prompt_workflow["3"]["inputs"]["denoise"] = denoise
        prompt_workflow["76"]["inputs"]["prompt"] = prompt
        prompt_workflow["77"]["inputs"]["prompt"] = neg_prompt
        prompt_workflow["66"]["inputs"]["shift"] = shift
        prompt_workflow["78"]["inputs"]["image"] = image_filename
        
        if "110" in prompt_workflow:
            del prompt_workflow["110"]
        
        prompt_workflow["3"]["inputs"]["latent_image"] = ["88", 0]

        async for status, res_img, comp_img, btn_state in queue_prompt_and_track_progress(prompt_workflow, COMFYUI_CLIENT_ID, COMFYUI_URL):
            yield status, res_img, comp_img, btn_state

    except Exception as e:
        yield f"错误: {str(e)}", None, None, gr.update(interactive=True)



# --- 3. Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Base(), title="Qwen Image Edit with ComfyUI", analytics_enabled=False, css=".info-icon {display: flex; align-items: center; justify-content: center;}") as demo:
    gr.Markdown("## Qwen Image Edit with ComfyUI")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
            with gr.Group():
                prompt = gr.Textbox(label="编辑指令 (Edit Instruction)", lines=2, value="Make the sky look like a van gogh painting")
                neg_prompt = gr.Textbox(label="负面提示词 (Negative Prompt)", lines=2, value="text, watermark, bad quality")
            
            gr.Markdown("### 图像编辑参数")
            with gr.Row(equal_height=True):
                with gr.Group():
                    with gr.Row(variant='compact', equal_height=True):
                        denoise_slider = gr.Slider(label="Denoise", minimum=0.0, maximum=1.0, step=0.05, value=1.0, scale=9)
                        gr.Markdown("[ⓘ](### '控制图像变化的程度。1.0 表示完全重新生成。')", elem_classes=["info-icon"])
                    with gr.Row(variant='compact', equal_height=True):
                        steps_slider = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=10, scale=9)
                        gr.Markdown("[ⓘ](### '生成步数，步数越高细节越多，但速度越慢。建议值: 10左右')", elem_classes=["info-icon"])
                    with gr.Row(variant='compact', equal_height=True):
                        shift_slider = gr.Slider(label="Shift (AuraFlow)", minimum=1.0, maximum=5.0, step=0.1, value=3.0, scale=9)
                        gr.Markdown("[ⓘ](### 'AuraFlow特定参数，控制生成光环效果的强度。')", elem_classes=["info-icon"])

                with gr.Group():
                    with gr.Row(variant='compact', equal_height=True):
                        cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.1, value=2.5, scale=9)
                        gr.Markdown("[ⓘ](### '提示词相关性，值越高越遵循提示词，但可能过曝。建议值: 2.5-7.0')", elem_classes=["info-icon"])
                    with gr.Row(variant='compact', equal_height=True):
                        scheduler_dropdown = gr.Dropdown(label="Scheduler", choices=["normal", "karras", "simple"], value="simple", scale=9)
                        gr.Markdown("[ⓘ](### '调度器，与采样器配合使用，影响细节和收敛速度。')", elem_classes=["info-icon"])
                    with gr.Row(variant='compact', equal_height=True):
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"], value="euler", scale=9)
                        gr.Markdown("[ⓘ](### '采样器，影响图像生成风格和速度。')", elem_classes=["info-icon"])
            with gr.Row():
                edit_btn = gr.Button("图像编辑 (Image Edit)", variant="primary", size="lg", scale=1)

        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.TabItem("图像编辑区 (Image Edit Area)"):
                    input_img = gr.Image(label="上传或拖拽图像 (Upload or Drag Image)", interactive=True, height=700, type="numpy")
                with gr.TabItem("生成结果 (Result)"):
                    output_img = gr.Image(label="生成结果 (Generated Result)", interactive=False, height=700)
                with gr.TabItem("对比图 (Comparison)"):
                    comparison_img = gr.Image(label="拼接对比图 (Stitched Comparison)", interactive=False, height=700)

    all_inputs = [prompt, neg_prompt, input_img, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, denoise_slider, shift_slider]
    all_outputs = [status_text, output_img, comparison_img, edit_btn]

    edit_btn.click(fn=run_generation, inputs=all_inputs, outputs=all_outputs)

if __name__ == "__main__":
    if not os.path.exists("temp_outputs"):
        os.makedirs("temp_outputs")
    demo.queue().launch()
