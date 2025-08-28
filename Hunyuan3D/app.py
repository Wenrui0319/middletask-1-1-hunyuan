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


    
    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False) as demo:

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt'):
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=hunyuan_logic.HAS_T2I and not hunyuan_logic.MV_MODE):
                        caption = gr.Textbox(label='Text Prompt', placeholder='HunyuanDiT will be used to generate image.', info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=hunyuan_logic.MV_MODE):
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140, min_width=100, elem_classes='mv-image')

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Group():
                    file_explorer = gr.FileExplorer(root_dir="data/", file_count="multiple", label="Select Input File", show_label=True)
                    with gr.Row():
                        delete_btn = gr.Button("Delete Selected Files")
                        upload_btn = gr.Button("Upload Selected File")
                        download_btn = gr.Button("Download Selected Files")

                    selected_files = gr.State([])
                    temp_download = gr.File(label="Download", visible=False)

            with gr.Column(scale=9):
                with gr.Tabs() as tabs_output:
                    with gr.Tab('Hunyuan3D'):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=250):
                                gr.Markdown("### Hunyuan3D Controls")
                                geneting_image = gr.Image(label='待生成图片', type='pil', image_mode='RGBA', height=290)
                                btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                                btn_all = gr.Button(value='Gen Textured Shape', variant='primary', visible=hunyuan_logic.HAS_TEXTUREGEN, min_width=100)
                                with gr.Tabs() as export_tabs:
                                    with gr.Tab("Options", id='tab_options', visible=hunyuan_logic.TURBO_MODE):
                                        gen_mode = gr.Radio(label='Generation Mode', info='Recommendation: Turbo for most cases, Fast for very complex cases, Standard seldom use.', choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                                        decode_mode = gr.Radio(label='Decoding Mode', info='The resolution for exporting mesh from generated vectset', choices=['Low', 'Standard', 'High'], value='Standard')
                                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                                        with gr.Row():
                                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                                        seed = gr.Slider(label="Seed", minimum=0, maximum=hunyuan_logic.MAX_SEED, step=1, value=1234, min_width=100)
                                        with gr.Row():
                                            num_steps = gr.Slider(maximum=100, minimum=1, value=5 if 'turbo' in hunyuan_logic.args.subfolder else 30, step=1, label='Inference Steps')
                                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                                        with gr.Row():
                                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000, label='Number of Chunks', min_width=100)
                                    with gr.Tab("Export", id='tab_export'):
                                        with gr.Row():
                                            file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS, value='glb', min_width=100)
                                            reduce_face = gr.Checkbox(label='Simplify Mesh', value=False, min_width=100)
                                            export_texture = gr.Checkbox(label='Include Texture', value=False, visible=False, min_width=100)
                                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000, label='Target Face Number')
                                        with gr.Row():
                                            confirm_export = gr.Button(value="Transform", min_width=100)
                                            file_export = gr.DownloadButton(label="Download", variant='primary', interactive=False, min_width=100)
                                
                            with gr.Column(scale=5):
                                gr.Markdown("### Hunyuan3D Result")
                                with gr.Tabs():
                                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                                        stats = gr.Json({}, label='Mesh Stats')
                    
                    with gr.Tab('SAM Segmentation', visible=SAM_AVAILABLE) as sam_tab:
                        sam_predictor_state = gr.State(sam_predictor_global)
                        sam_original_image_state = gr.State(None)
                        sam_mask_state = gr.State(None)
                        sam_history_state = gr.State([])
                        sam_box_start_state = gr.State(None)

                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2, min_width=250):
                                gr.Markdown("### SAM Controls")
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
                                interactive_display = gr.Image(label="交互式显示 (请先在左侧Image Prompt上传图片)", type="numpy", interactive=True, height=400)
                                cutout_gallery = gr.Gallery(label="抠图结果", preview=True, object_fit="contain", height="auto")
                        

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

        #hunyuan更新组件列表
        hunyuan_upload_and_reset_outputs = [geneting_image]
        hunyuan_clear_outputs = [geneting_image]

        #sam的更新组件列表
        sam_upload_and_reset_outputs = [interactive_display, sam_original_image_state, sam_mask_state, sam_history_state, cutout_gallery, sam_predictor_state, sam_box_start_state]
        sam_clear_outputs = [interactive_display, sam_original_image_state, sam_mask_state, sam_history_state, cutout_gallery, sam_box_start_state]
        
        #统一控制选择逻辑
        def combine_set_image(sam_predictor_state, image):
            hunyuan_outputs = (image, )
            sam_outputs = sam_logic.set_image_for_predictor(sam_predictor_state, image)
            return hunyuan_outputs + sam_outputs
        image.upload(
            fn=combine_set_image, 
            inputs=[sam_predictor_state, image], 
            outputs= hunyuan_upload_and_reset_outputs + sam_upload_and_reset_outputs
        )

        def combine_clear_image():
            hunyuan_outputs = (None, ) 
            sam_outputs = sam_logic.clear_sam_panel()
            return hunyuan_outputs + sam_outputs
        image.clear(
            fn=combine_clear_image,
            outputs=hunyuan_clear_outputs + sam_clear_outputs
        )
        
        #hunyuan3D生成逻辑
        btn.click(
            hunyuan_logic.shape_generation,
            inputs=[caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed],
            outputs=[file_out, html_gen_mesh, stats, seed]
        )
        
        btn_all.click(
            hunyuan_logic.generation_all,
            inputs=[caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        )

        def on_gen_mode_change(value):
            if value == 'Turbo': return gr.update(value=5)
            elif value == 'Fast': return gr.update(value=10)
            else: return gr.update(value=30)
        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low': return gr.update(value=196)
            elif value == 'Standard': return gr.update(value=256)
            else: return gr.update(value=384)
        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], outputs=[octree_resolution])

        def on_export_click(file_out_val, file_out2_val, file_type_val, reduce_face_val, export_texture_val, target_face_num_val):
            if file_out_val is None: 
                raise gr.Error('Please generate a mesh first.')
            if export_texture_val:
                mesh = trimesh.load(file_out2_val.name)
                save_folder = hunyuan_logic.gen_save_folder()
                path = hunyuan_logic.export_mesh(mesh, save_folder, textured=True, type=file_type_val)
                _ = hunyuan_logic.export_mesh(mesh, hunyuan_logic.gen_save_folder(), textured=True)
                model_viewer_html = hunyuan_logic.build_model_viewer_html(save_folder, height=hunyuan_logic.HTML_HEIGHT, width=hunyuan_logic.HTML_WIDTH, textured=True)
            else:
                mesh = trimesh.load(file_out_val.name)
                mesh = hunyuan_logic.floater_remove_worker(mesh)
                mesh = hunyuan_logic.degenerate_face_remove_worker(mesh)
                if reduce_face_val:
                    mesh = hunyuan_logic.face_reduce_worker(mesh, target_face_num_val)
                save_folder = hunyuan_logic.gen_save_folder()
                path = hunyuan_logic.export_mesh(mesh, save_folder, textured=False, type=file_type_val)
                _ = hunyuan_logic.export_mesh(mesh, hunyuan_logic.gen_save_folder(), textured=False)
                model_viewer_html = hunyuan_logic.build_model_viewer_html(save_folder, height=hunyuan_logic.HTML_HEIGHT, width=hunyuan_logic.HTML_WIDTH, textured=False)
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected=tabs_output.get_tabs()[1]),
            outputs=None
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

        #SAM控制逻辑
        selected_model.change(fn=sam_logic.load_sam_model, inputs=[selected_model, gr.State(args.device)], outputs=[sam_predictor_state])
        
        interactive_display.select(
            fn=sam_logic.interactive_predict,
            inputs=[sam_predictor_state, sam_original_image_state, sam_history_state, mode_radio, sam_box_start_state],
            outputs=[interactive_display, sam_mask_state, sam_history_state, sam_predictor_state, sam_box_start_state]
        )
        
        cut_everything_btn.click(fn=sam_logic.generate_everything, inputs=[sam_predictor_state, sam_original_image_state], outputs=[interactive_display, cutout_gallery])
        cut_out_btn.click(fn=sam_logic.single_cutout, inputs=[sam_original_image_state, sam_mask_state], outputs=[cutout_gallery])
        reset_btn.click(fn=sam_logic.reset_all_sam, inputs=[sam_predictor_state, sam_original_image_state], outputs=sam_upload_and_reset_outputs)
        
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

    hunyuan_logic.initialize_hunyuan_models(args)

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {hunyuan_logic.HTML_HEIGHT}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    sam_predictor_global = None
    if SAM_AVAILABLE:
        print("\n--- [SAM] 准备预加载SAM模型...")
        sam_model_dir = "models"
        if not os.path.exists(sam_model_dir): 
            os.makedirs(sam_model_dir)
        available_sam_models = [x for x in os.listdir(sam_model_dir) if x.endswith(".pth")]
        if available_sam_models:
            default_sam_model = 'sam_vit_b_01ec64.pth' or available_sam_models[0]
            print(f"    > 找到默认SAM模型: {default_sam_model}")
            sam_predictor_global = sam_logic.load_sam_model(default_sam_model, args.device)
        else:
            print("    > 警告: 在 'models' 文件夹中未找到SAM模型。SAM功能将不可用，直到手动选择模型。")

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
