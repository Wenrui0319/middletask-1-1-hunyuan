import gradio as gr
import os
from workflow_manager import WorkflowManager

# 初始化工作流管理器
wm = WorkflowManager(filepath="workflow.json", workspace_dir="workspace")

# --- JavaScript and CSS ---
# 用于前端和后端通信的JS代码
js_script = """
function handleNodeClick(nodeId) {
    // 1. 找到隐藏的 Gradio Textbox 组件。Gradio 会根据 label 生成一个CSS类。
    // 我们需要找到正确的输入元素。通常是父元素 div 的下一个 textarea。
    // 更稳妥的方式是给组件一个 elem_id。
    const hidden_textbox_elem = document.querySelector('#selected_node_id_input textarea');
    
    if (hidden_textbox_elem) {
        // 2. 设置 Textbox 的值
        hidden_textbox_elem.value = nodeId;
        
        // 3. 触发 'input' 和 'change' 事件，确保 Gradio 能监听到变化
        const inputEvent = new Event('input', { bubbles: true });
        const changeEvent = new Event('change', { bubbles: true });
        hidden_textbox_elem.dispatchEvent(inputEvent);
        hidden_textbox_elem.dispatchEvent(changeEvent);
    } else {
        console.error("Could not find the hidden textbox for node selection.");
    }
}
"""

# Mermaid 容器的样式
css_style = """
.mermaid-container {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    background-color: #f9f9f9;
}
"""

# --- Gradio UI 定义 ---
with gr.Blocks(js=js_script, css=css_style) as demo:
    gr.Markdown("## 图像编辑工作流")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 节点预览与操作")
            preview_image = gr.Image(label="节点预览", interactive=False, height=300)
            edit_button = gr.Button("载入此图进行编辑")
            delete_button = gr.Button("删除选中节点及后续")
            
            # 隐藏的组件，用于JS和Python通信
            # 使用 elem_id 来获得一个稳定的 DOM ID
            selected_node_id = gr.Textbox(
                label="Selected Node ID", 
                visible=False, 
                elem_id="selected_node_id_input"
            )

        with gr.Column(scale=2):
            gr.Markdown("#### 工作流历史")
            # 添加一个容器 div 以应用样式
            with gr.Box(elem_classes="mermaid-container"):
                 workflow_tree_html = gr.HTML("请上传一张图片以开始新的工作流。")
            
            upload_button = gr.UploadButton(
                "上传新图像 (创建新工作流)", 
                file_types=["image"]
            )

    # --- 后端逻辑绑定 (将在后续步骤中实现) ---
    
    # --- Backend Logic Binding ---
    
    # 应用加载时，渲染初始的工作流树
    def initial_load():
        return wm.to_mermaid()

    demo.load(fn=initial_load, outputs=workflow_tree_html)

    # 绑定上传按钮的事件
    def on_upload(file_obj):
        if file_obj is None:
            return wm.to_mermaid()
        
        # Gradio 的 UploadButton 返回一个包含临时文件信息的对象
        # .name 是临时文件的路径
        wm.add_root_node(file_obj.name, file_obj.type)
        wm.save()
        return wm.to_mermaid()

    upload_button.upload(fn=on_upload, inputs=upload_button, outputs=workflow_tree_html)

    # 绑定节点选择事件
    def on_select_node(node_id):
        if not node_id:
            return gr.update(value=None)
        
        node = wm.get_node(node_id)
        if node:
            image_path = node['data']['file_path']
            # 确保文件存在
            if os.path.exists(image_path):
                return gr.update(value=image_path)
        
        # 如果找不到节点或文件，则清空预览
        return gr.update(value=None)

    selected_node_id.change(
        fn=on_select_node,
        inputs=selected_node_id,
        outputs=preview_image
    )

    # 绑定“编辑”按钮的事件
    def on_edit_click(current_node_id):
        if not current_node_id:
            gr.Warning("请先在工作流中选择一个节点！")
            return

        node = wm.get_node(current_node_id)
        if not node:
            gr.Warning("找不到所选节点。")
            return

        # 检查节点类型，禁止编辑非叶子节点
        if node['type'] != 'leaf':
            gr.Warning(f"此节点为操作节点 ({node['label']})，其结果不可再次编辑。请选择一个叶子节点。")
            return
        
        # 模拟加载到编辑页
        gr.Info(f"已将节点 {node['label']} ({node['id'][:8]}...) 的图像加载到编辑页面。")
        print(f"EDITING: Loading image from {node['data']['file_path']} for node {node['id']}")

    edit_button.click(
        fn=on_edit_click,
        inputs=[selected_node_id],
        outputs=None # 此操作不直接更新任何组件
    )

    # 添加一个模拟的“保存”按钮来测试 add_child_node
    mock_save_button = gr.Button("模拟保存(添加子节点)")

    def on_mock_save(current_node_id, current_image):
        if not current_node_id or current_image is None:
            # 如果没有选中节点或没有图片，则不执行任何操作
            gr.Warning("请先在工作流中选择一个父节点！")
            return wm.to_mermaid()

        # 模拟一次编辑操作
        operation_name = "Mock Edit"
        params = {"mock_param": "value123"}
        
        # current_image 是一个 numpy 数组，需要先保存为临时文件
        from PIL import Image
        import time
        temp_img_path = f"workspace/temp_{time.time()}.png"
        img = Image.fromarray(current_image)
        img.save(temp_img_path)

        # 添加子节点
        wm.add_child_node(current_node_id, operation_name, temp_img_path, params)
        wm.save()
        
        # 清理临时文件
        os.remove(temp_img_path)

        # 刷新工作流树
        return wm.to_mermaid()

    mock_save_button.click(
        fn=on_mock_save,
        inputs=[selected_node_id, preview_image],
        outputs=workflow_tree_html
    )


if __name__ == "__main__":
    demo.launch()