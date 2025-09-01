# app.py

import os
import torch
import shutil
import gradio as gr
import uvicorn
import time
from fastapi import FastAPI
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

# 确保data目录结构存在
os.makedirs("data/sam", exist_ok=True)

def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if hunyuan_logic.MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in hunyuan_logic.args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
    if hunyuan_logic.TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')


    
    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False, css=".gradio-container { max-width: unset !important; padding-left: 20px; padding-right: 20px; }") as demo:

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
                                delete_btn = gr.Button("Delete Selected Files")
                                upload_btn = gr.Button("Upload Selected File")
                                download_btn = gr.Button("Download Selected Files")
        
                            selected_files = gr.State([])
                            temp_download = gr.File(label="Download", visible=False)

            with gr.Column(scale=9):
                with gr.Tabs() as tabs_output:
                    with gr.Tab('SAM', visible=SAM_AVAILABLE):
                        sam_logic.create_sam_ui(sam_logic.sam_predictor_global, image)
                    with gr.Tab('Hunyuan3D'):
                        geneting_image = hunyuan_logic.create_hunyuan_ui(hunyuan_logic.SUPPORTED_FORMATS, hunyuan_logic.HTML_OUTPUT_PLACEHOLDER, tabs_output, image, caption, mv_image_front, mv_image_back, mv_image_left, mv_image_right, file_out, file_out2)
                    
                    with gr.Tab('Qwen Edit'):
                        qwen_edit_logic.create_qwen_edit_ui()

                    with gr.Tab('Qwen Inpainting'):
                        qwen_inpainting_logic.create_qwen_inpainting_ui()

                    with gr.Tab('Gemini Chat'):
                        gemini_gradio_app.create_gemini_chat_ui()
                        


        gr.HTML(f"""
        <div align="center">
        Activated Model - Shape Generation ({hunyuan_logic.args.model_path}/{hunyuan_logic.args.subfolder}) ; Texture Generation ({'Hunyuan3D-2' if hunyuan_logic.HAS_TEXTUREGEN else 'Unavailable'})
        </div>
        """)
        if not hunyuan_logic.HAS_TEXTUREGEN:
            gr.HTML("""
            <div style="margin-top: 5px;"  align="center">
                <b>Warning: </b>
                Texture synthesis is disable due to missing requirements,
                 please install requirements following <a href="https://github.com/Tencent/Hunyuan3D-2?tab=readme-ov-file#install-requirements">README.md</a>to activate it.
            </div>
            """)
        if not hunyuan_logic.args.enable_t23d:
            gr.HTML("""
            <div style="margin-top: 5px;"  align="center">
                <b>Warning: </b>
                Text to 3D is disable. To activate it, please run `python app.py --enable_t23d`.
            </div>
            """)


        # File operation event handlers
        file_explorer.change(
            fn=file_operations.handle_file_selection,
            inputs=[file_explorer],
            outputs=[selected_files]
        )
        
        delete_btn.click(
            fn=file_operations.delete_selected_files,
            inputs=[selected_files],
            outputs=[file_explorer]
        ).then(
            fn=lambda: gr.update(root_dir="data/"),
            outputs=[file_explorer]
        )

        upload_btn.click(
            fn=file_operations.upload_selected_image,
            inputs=[selected_files],
            outputs=[image]
        )
        
        download_btn.click(
            fn=file_operations.download_selected_files,
            inputs=[selected_files],
            outputs=[temp_download]
        ).then(
            fn=None,
            inputs=[temp_download],
            js="""
            (file) => {
                if (file && file.url) {
                    // Create a temporary link and trigger download
                    const a = document.createElement('a');
                    a.href = file.url;
                    a.download = file.orig_name || 'selected_files.zip';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            }
            """
        )
    
    return demo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-0')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    args = parser.parse_args()

    # hunyuan_logic.initialize_hunyuan(args)
    sam_logic.initialize_sam(args)
    # For frontend testing
    hunyuan_logic.args = args
    hunyuan_logic.SAVE_DIR = temp_dir

    app = FastAPI()
    static_dir = Path(hunyuan_logic.SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    if os.path.exists('./assets/env_maps'):
        shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)
    
    if args.low_vram_mode:
        torch.cuda.empty_cache()

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    
    print(f"\n>>> Gradio 服务即将启动！请在浏览器中打开 http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
