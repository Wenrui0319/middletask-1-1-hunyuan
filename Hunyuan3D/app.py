# app.py

import os
import torch
import shutil
import gradio as gr
import uvicorn
import time
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import trimesh

import hunyuan_logic
import sam_logic
from sam_logic import SAM_AVAILABLE
import file_operations
import qwen_edit_logic
import qwen_inpainting_logic
import gemini_gradio_app

try:
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['GRADIO_TEMP_DIR'] = temp_dir
    print(f"Gradio temporary directory set to: {temp_dir}")
except Exception as e:
    print(f"Warning: Could not create or set local Gradio temp directory: {e}")

os.makedirs("data/sam", exist_ok=True)

def build_app(args):
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if hunyuan_logic.MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in hunyuan_logic.args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
    if hunyuan_logic.TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False, css=".gradio-container { max-width: unset !important; padding-left: 20px; padding-right: 20px; }") as demo:
        # 主界面标题
        gr.HTML("""
        <div style="text-align: center;">
            <h1 style="font-size: 2em; margin: 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                Selective 3D Generation Engine
            </h1>
            <h2 style="font-size: 1em; margin: 10px 0 0 0; font-weight: 300; opacity: 0.9;">
                Middle Task 1 - Team 1 
            </h2>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("File Explorer"):
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        caption = gr.Textbox(visible=False)
                        mv_image_front = gr.Image(visible=False)
                        mv_image_back = gr.Image(visible=False)
                        mv_image_left = gr.Image(visible=False)
                        mv_image_right = gr.Image(visible=False)
                        with gr.Group():
                            file_out = gr.File(label="File", visible=False)
                            file_out2 = gr.File(label="File", visible=False)
                        with gr.Group():
                            file_explorer = gr.FileExplorer(root_dir="data/", file_count="multiple", label="Select Input File", show_label=True, height=600)
                            with gr.Row():
                                delete_btn = gr.Button("Delete")
                                upload_btn = gr.Button("Edit")
                                local_upload_btn = gr.UploadButton("Upload", file_count="multiple")
                                download_btn = gr.Button("Download")
                            selected_files = gr.State([])
                            temp_download = gr.File(label="Download", visible=False)
            with gr.Column(scale=9):
                active_tab_state = gr.State("SAM")
                with gr.Tabs() as tabs_output:
                    with gr.Tab('SAM', visible=SAM_AVAILABLE, id="SAM"):
                        sam_input_image = sam_logic.create_sam_ui(
                            sam_predictor_global=sam_logic.sam_predictor_global,
                            file_explorer=file_explorer,
                            device=args.sam_device
                        )
                    with gr.Tab('Qwen Edit', id="Qwen Edit"):
                        qwen_edit_input_image = qwen_edit_logic.create_qwen_edit_ui(file_explorer)
                    with gr.Tab('Qwen Inpainting', id="Qwen Inpainting"):
                        qwen_inpainting_input_image = qwen_inpainting_logic.create_qwen_inpainting_ui(file_explorer)
                    with gr.Tab('Gemini Chat', id="Gemini Chat"):
                        gemini_uploaded_files_state, gemini_text_input = gemini_gradio_app.create_gemini_chat_ui()
                    with gr.Tab('Hunyuan3D', id="Hunyuan3D"):
                        hunyuan_input_image = hunyuan_logic.create_hunyuan_ui(hunyuan_logic.SUPPORTED_FORMATS, hunyuan_logic.HTML_OUTPUT_PLACEHOLDER, tabs_output, caption, mv_image_front, mv_image_back, mv_image_left, mv_image_right, file_out, file_out2, file_explorer)
        
        def on_tab_select(evt: gr.SelectData):
            return evt.value
        tabs_output.select(fn=on_tab_select, inputs=None, outputs=[active_tab_state])
        
        file_explorer.change(fn=file_operations.handle_file_selection, inputs=[file_explorer], outputs=[selected_files])
        file_explorer.change(fn=file_operations.preview_image, inputs=[file_explorer], outputs=[image])
        delete_btn.click(fn=file_operations.delete_selected_files, inputs=[selected_files], outputs=[file_explorer])
        upload_btn.click(fn=file_operations.dispatch_image, inputs=[selected_files, active_tab_state], outputs=[sam_input_image, qwen_edit_input_image, qwen_inpainting_input_image, gemini_uploaded_files_state, gemini_text_input, hunyuan_input_image])
        download_btn.click(
            fn=file_operations.download_selected_files,
            inputs=[selected_files],
            outputs=[temp_download]
        ).then(fn=None, inputs=[temp_download], js="""(file) => { if (file && file.url) { const a = document.createElement('a'); a.href = file.url; a.download = file.orig_name || 'selected_files.zip'; document.body.appendChild(a); a.click(); document.body.removeChild(a); } }""")
        local_upload_btn.upload(fn=file_operations.upload_from_local, inputs=[local_upload_btn], outputs=[file_explorer])
        
        demo.load(fn=sam_logic.check_sam_model_on_load)
    
    return demo

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-0')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device for main models like Hunyuan3D.")
    parser.add_argument('--sam_device', type=str, default=None, help="Device for SAM model. Defaults to --device if not set.")
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    args = parser.parse_args()

    if args.sam_device is None:
        args.sam_device = args.device

    hunyuan_logic.initialize_hunyuan(args)
    
    from argparse import Namespace
    sam_args = Namespace(device=args.sam_device)
    sam_logic.initialize_sam(sam_args)

    hunyuan_logic.args = args
    hunyuan_logic.SAVE_DIR = temp_dir

    app = FastAPI()
    
    demo = build_app(args)
    app = gr.mount_gradio_app(app, demo, path="/")
    
    print(f"\n>>> Gradio 服务即将启动！请在浏览器中打开 http://{args.host}:{args.port}")
    print(f"    - 主模型设备: {args.device}")
    print(f"    - SAM 模型设备: {args.sam_device}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
