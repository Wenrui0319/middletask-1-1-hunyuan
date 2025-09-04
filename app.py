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
    
    # 示例：启动时加载现有的工作流
    def initial_load():
        # 这里将在后续实现 to_mermaid 方法
        # return wm.to_mermaid()
        return "请上传一张图片或加载现有工作流。"

    demo.load(fn=lambda: initial_load(), outputs=workflow_tree_html)


if __name__ == "__main__":
    demo.launch()