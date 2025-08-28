import gradio as gr
import json
import random
import uuid
import asyncio
import websockets
import requests
from PIL import Image
import io

# --- 1. ComfyUI API Wrapper ---

def get_server_address(server_address_in_ui):
    return server_address_in_ui.strip('/')

async def upload_image(server_address, image_pil, overwrite=True):
    if not image_pil: return None
    url = f"{get_server_address(server_address)}/upload/image"
    byte_io = io.BytesIO()
    image_pil.save(byte_io, format="PNG")
    files = {'image': ('image.png', byte_io.getvalue(), 'image/png')}
    data = {'overwrite': str(overwrite).lower()}
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error uploading image: {e}")
        return None

async def queue_prompt(server_address, prompt_workflow, client_id):
    url = f"{get_server_address(server_address)}/prompt"
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error queueing prompt: {e}")
        return None

async def get_history(server_address, prompt_id):
    url = f"{get_server_address(server_address)}/history/{prompt_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting history: {e}")
        return None

# --- 2. Main Logic ---

async def run_generation(
    server_address, input_image, positive_prompt, negative_prompt, 
    seed, sampler_name, scheduler, steps, cfg, denoise
):
    if not server_address or not server_address.strip():
        raise gr.Error("ComfyUI Server Address is required.")
    if input_image is None:
        raise gr.Error("Input image is required.")

    image_pil = Image.fromarray(input_image)
    client_id = str(uuid.uuid4())
    
    yield "Initializing...", None, gr.Button(interactive=False)

    try:
        yield "Uploading...", None, gr.Button(interactive=False)
        uploaded_image_data = await upload_image(server_address, image_pil)
        if not uploaded_image_data:
            raise gr.Error("Failed to upload image.")
        
        with open("image_qwen_image_edit.json", 'r') as f:
            workflow = json.load(f)
        
        workflow["78"]["inputs"]["image"] = uploaded_image_data['name']
        workflow["76"]["inputs"]["prompt"] = positive_prompt
        workflow["77"]["inputs"]["prompt"] = negative_prompt
        
        ksampler = workflow["3"]["inputs"]
        ksampler["seed"] = int(seed)
        ksampler["steps"] = int(steps)
        ksampler["cfg"] = float(cfg)
        ksampler["sampler_name"] = sampler_name
        ksampler["scheduler"] = scheduler
        ksampler["denoise"] = float(denoise)

        yield "Queueing...", None, gr.Button(interactive=False)
        queued_data = await queue_prompt(server_address, workflow, client_id)
        if not queued_data:
            raise gr.Error("Failed to queue prompt.")
        
        prompt_id = queued_data['prompt_id']
        ws_url = f"ws://{get_server_address(server_address).replace('http://', '').replace('https://', '')}/ws?clientId={client_id}"
        
        async with websockets.connect(ws_url) as ws:
            while True:
                msg = json.loads(await ws.recv())
                if msg['type'] == 'status':
                    q = msg['data']['status']['exec_info']['queue_remaining']
                    yield f"In queue: {q} left", None, gr.Button(interactive=False)
                elif msg['type'] == 'progress':
                    p = (msg['data']['value'] / msg['data']['max']) * 100
                    yield f"Processing... {p:.1f}%", None, gr.Button(interactive=False)
                elif msg['type'] == 'executed' and msg['data']['prompt_id'] == prompt_id:
                    yield "Fetching...", None, gr.Button(interactive=False)
                    history = await get_history(server_address, prompt_id)
                    filename = history[prompt_id]['outputs'].get('60', {}).get('images', [{}])[0].get('filename')
                    if filename:
                        image_url = f"{get_server_address(server_address)}/view?filename={filename}"
                        response = requests.get(image_url)
                        response.raise_for_status()
                        final_image = Image.open(io.BytesIO(response.content))
                        yield "Done!", final_image, gr.Button("Start Generation", interactive=True)
                    else:
                        raise gr.Error("Output image not found.")
                    break
                elif msg['type'] == 'execution_error':
                    raise gr.Error(f"Execution Error: {msg['data'].get('exception_message', 'Unknown')}")

    except Exception as e:
        yield f"Error: {e}", None, gr.Button("Start Generation", interactive=True)

# --- 3. Gradio UI (Final Precise Layout v7) ---

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=".gradio-container { max-width: 1400px !important; }") as demo:
        gr.Markdown("## ComfyUI Qwen Image Editor")
        
        with gr.Row():
            # --- Left Main Column ---
            with gr.Column(scale=1):
                server_address_input = gr.Textbox(label="ComfyUI Server Address", value="http://127.0.0.1:8188/")
                with gr.Row():
                    input_image_widget = gr.Image(label="Source Image", type="numpy", height=300, scale=2)
                    with gr.Column(scale=1):
                        with gr.Row():
                            seed_widget = gr.Number(label="Seed", value=0, precision=0, scale=4)
                            def randomize_seed(): return random.randint(0, 1e15)
                            random_seed_btn = gr.Button("🎲", scale=1, min_width=10, elem_id="random-seed")
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"], value="euler")
                        scheduler_dropdown = gr.Dropdown(label="Scheduler", choices=["normal", "karras", "simple"], value="simple")
                # Reduced lines for height alignment
                positive_prompt_widget = gr.Textbox(label="Positive Prompt (Edit Instruction)", lines=2, value="A girl in a beautiful dress")
                negative_prompt_widget = gr.Textbox(label="Negative Prompt", lines=2, value="text, watermark, bad quality")

            # --- Right Main Column ---
            with gr.Column(scale=1):
                status_text_widget = gr.Textbox(label="Status", value="Ready", interactive=False)
                result_image_widget = gr.Image(label="Generated Result", height=450, interactive=False)
                with gr.Accordion("Fine-tuning Parameters", open=True):
                    steps_slider = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=20)
                    cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.1, value=2.5)
                    denoise_slider = gr.Slider(label="Denoise", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
        
        generate_btn = gr.Button("Start Generation", variant="primary", size="lg")

        all_inputs = [
            server_address_input, input_image_widget, positive_prompt_widget, negative_prompt_widget,
            seed_widget, sampler_dropdown, scheduler_dropdown, 
            steps_slider, cfg_slider, denoise_slider
        ]
        all_outputs = [status_text_widget, result_image_widget, generate_btn]
        
        random_seed_btn.click(fn=randomize_seed, outputs=[seed_widget])
        generate_btn.click(fn=run_generation, inputs=all_inputs, outputs=all_outputs)
    
    return demo

if __name__ == "__main__":
    import os
    os.environ['GRADIO_ALLOW_FLAGGING'] = 'never'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    
    app = build_ui()
    app.queue().launch()
