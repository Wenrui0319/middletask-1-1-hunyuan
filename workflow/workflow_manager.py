# workflow_manager.py

import json
import os
import uuid
from datetime import datetime

class WorkflowManager:
    """
    管理图像编辑工作流的状态，包括节点的创建、删除、查询，
    以及将工作流数据持久化为 JSON 文件。
    """
    def __init__(self, filepath="workflow.json", workspace_dir="workspace"):
        """
        初始化管理器。

        Args:
            filepath (str): 工作流JSON文件的路径。
            workspace_dir (str): 用于存放上传图片、中间结果和遮罩的目录。
        """
        self.filepath = filepath
        self.workspace_dir = workspace_dir
        self.nodes = []
        
        # 确保工作区目录存在
        os.makedirs(os.path.join(self.workspace_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "masks"), exist_ok=True)

        self.load()

    def load(self):
        """
        从 JSON 文件加载节点数据。如果文件不存在，则初始化一个空的工作流。
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", [])
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading workflow file: {e}. Starting with a new workflow.")
                self.nodes = []
        else:
            self.nodes = []

    def save(self):
        """
        将当前节点数据和元数据保存到 JSON 文件。
        """
        data_to_save = {
            "metadata": {
                "version": "1.0",
                "last_saved": datetime.now().isoformat()
            },
            "nodes": self.nodes
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def get_node(self, node_id):
        """
        根据 ID 查找并返回节点对象。

        Args:
            node_id (str): 要查找的节点 ID。

        Returns:
            dict | None: 找到的节点对象，如果未找到则返回 None。
        """
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None

    def add_root_node(self, original_filepath, file_type):
        """
        添加一个新的根节点 (当用户上传新图片时)。
        文件将被复制到工作区内进行管理。

        Args:
            original_filepath (str): 上传文件的临时路径。
            file_type (str): 文件的MIME类型。

        Returns:
            str: 新创建的节点 ID。
        """
        # 为文件创建一个安全的文件名并定义存储路径
        filename = f"{uuid.uuid4().hex}_{os.path.basename(original_filepath)}"
        new_filepath = os.path.join(self.workspace_dir, "uploads", filename)
        
        # 复制文件到工作区
        # Gradio 的 gr.File/gr.Image 的 .name 就是临时路径
        import shutil
        shutil.copy(original_filepath, new_filepath)

        node_id = str(uuid.uuid4())
        new_node = {
            "id": node_id,
            "parent_id": None,
            "type": "leaf",
            "label": file_type.split('/')[-1].upper() if file_type else "FILE",
            "data": {
                "file_path": new_filepath,
                "file_type": file_type,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.nodes.append(new_node)
        return node_id

    def add_child_node(self, parent_id, operation_name, output_image_path, params={}):
        """
        在指定父节点下添加一个新的操作节点。
        如果父节点是 leaf，则将其类型转换为 operation。

        Args:
            parent_id (str): 父节点的 ID。
            operation_name (str): 本次操作的名称，将作为新节点的 label。
            output_image_path (str): 编辑后生成的输出图像的路径。
            params (dict, optional): 本次操作所使用的参数。

        Returns:
            str | None: 新创建的节点 ID，如果父节点不存在则返回 None。
        """
        parent_node = self.get_node(parent_id)
        if not parent_node:
            print(f"Error: Parent node with ID {parent_id} not found.")
            return None

        # 如果父节点是叶子，则将其提升为操作节点
        if parent_node['type'] == 'leaf':
            parent_node['type'] = 'operation'

        # 将输出图片复制到工作区
        filename = f"{uuid.uuid4().hex}_{os.path.basename(output_image_path)}"
        new_filepath = os.path.join(self.workspace_dir, "results", filename)
        import shutil
        shutil.copy(output_image_path, new_filepath)

        node_id = str(uuid.uuid4())
        new_node = {
            "id": node_id,
            "parent_id": parent_id,
            "type": "leaf",  # 新创建的节点默认是叶子，直到它有自己的子节点
            "label": operation_name,
            "data": {
                "file_path": new_filepath,
                "file_type": f"image/{os.path.splitext(new_filepath)[1][1:]}",
                "timestamp": datetime.now().isoformat(),
                "operation_params": params
            }
        }
        self.nodes.append(new_node)
        return node_id

    def delete_subtree(self, node_id):
        """
        删除以指定节点为根的整个子树，并处理父节点的退化。

        Args:
            node_id (str): 要删除的子树的根节点 ID。
        """
        node_to_delete = self.get_node(node_id)
        if not node_to_delete:
            return

        # 1. 找到所有要删除的节点 (本身及其所有后代)
        nodes_to_remove = []
        queue = [node_id]
        visited = {node_id}
        
        while queue:
            current_id = queue.pop(0)
            nodes_to_remove.append(current_id)
            
            # 找到所有子节点
            children = [n['id'] for n in self.nodes if n.get('parent_id') == current_id]
            for child_id in children:
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append(child_id)
        
        # 2. 从文件系统中删除关联的文件
        for nid in nodes_to_remove:
            node = self.get_node(nid)
            if node and 'file_path' in node['data']:
                filepath = node['data']['file_path']
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError as e:
                        print(f"Error deleting file {filepath}: {e}")

        # 3. 从节点列表中移除这些节点
        self.nodes = [n for n in self.nodes if n['id'] not in nodes_to_remove]

        # 4. 处理父节点的退化
        parent_id = node_to_delete.get('parent_id')
        if parent_id:
            parent_node = self.get_node(parent_id)
            if parent_node:
                # 检查父节点是否还有其他子节点
                has_children = any(n.get('parent_id') == parent_id for n in self.nodes)
                if not has_children and parent_node['type'] == 'operation':
                    parent_node['type'] = 'leaf'
                    print(f"Node {parent_id} has been demoted to a leaf node.")

    def to_mermaid(self):
        """
        将当前工作流数据转换为 Mermaid.js 格式的字符串。
        这个字符串将由前端的 JavaScript 渲染。

        Returns:
            str: Mermaid.js 图表定义的字符串。
        """
        if not self.nodes:
            return "<p style='text-align:center;'>工作流为空，请上传一张图片以开始。</p>"

        # 替换在 ID 中可能影响 Mermaid 语法的字符
        def sanitize_id(node_id):
            return "node_" + node_id.replace('-', '_')

        mermaid_string = "graph TD;\n"
        
        # 定义节点
        for node in self.nodes:
            s_id = sanitize_id(node['id'])
            # 移除标签中的引号，避免语法错误
            label = node['label'].replace('"', '')
            mermaid_string += f'    {s_id}["{label}"];\n'
        
        mermaid_string += "\n"

        # 定义连接
        for node in self.nodes:
            if node.get('parent_id'):
                s_parent_id = sanitize_id(node['parent_id'])
                s_id = sanitize_id(node['id'])
                mermaid_string += f'    {s_parent_id} --> {s_id};\n'

        mermaid_string += "\n"

        # 定义样式 (增加了手写字体)
        mermaid_string += "    classDef leafNode fill:#3498db,color:#fff,stroke:#fff,stroke-width:2px,font-family:monospace;\n"
        mermaid_string += "    classDef opNode fill:#2980b9,color:#fff,stroke:#fff,stroke-width:2px,font-family:monospace;\n\n"

        # 应用样式和点击事件
        for node in self.nodes:
            s_id = sanitize_id(node['id'])
            node_class = "leafNode" if node['type'] == 'leaf' else "opNode"
            mermaid_string += f"    class {s_id} {node_class};\n"
            # Mermaid 的 click 事件需要调用全局 JS 函数
            mermaid_string += f'    click {s_id} call window.handleNodeClick("{node["id"]}");\n'
            
        return mermaid_string
