import uuid
import os
from datetime import datetime

class WorkflowManager:
    """
    管理工作流树数据结构的类。
    负责节点的创建、删除、查找和修改。
    """
    def __init__(self):
        self.tree = None
        # 创建一个用于存储图像的目录
        self.workspace_dir = "workspace"
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _generate_id(self):
        """生成唯一的节点ID"""
        return str(uuid.uuid4())

    def _save_image(self, image_temp_path):
        """将Gradio上传的临时图像保存到工作区"""
        _, extension = os.path.splitext(image_temp_path)
        if not extension:
            extension = ".png" # 默认扩展名
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{self._generate_id()[:8]}{extension}"
        permanent_path = os.path.join(self.workspace_dir, new_filename)
        
        # Gradio 的 Image.upload 返回的是临时文件路径，需要移动或复制
        # 在实际应用中，这里可能需要 `shutil.move` 或 `Image.open().save()`
        # 为了简化，我们假设直接可以从temp_path读取
        # 注意: Gradio的临时文件处理可能需要更复杂的逻辑
        os.rename(image_temp_path, permanent_path)
        
        return permanent_path

    def find_node(self, node_id, node=None):
        """在树中递归查找指定ID的节点"""
        if node is None:
            node = self.tree
        
        if not node:
            return None

        if node.get("id") == node_id:
            return node
        
        for child in node.get("children", []):
            found = self.find_node(node_id, child)
            if found:
                return found
        
        return None

    def create_root_node(self, image_path):
        """根据上传的图像创建根节点"""
        permanent_path = self._save_image(image_path)
        file_type = os.path.splitext(permanent_path)[1].replace(".", "")

        self.tree = {
            "id": self._generate_id(),
            "parentId": None,
            "name": file_type.upper(),
            "type": "leaf",
            "file": {
                "path": permanent_path,
                "type": file_type,
            },
            "children": []
        }
        return self.tree

    def add_child_node(self, parent_id, new_image_path, operation_details):
        """
        在指定父节点下添加一个新的子节点（通常是操作的结果）。
        如果父节点是叶子，会将其转换成分支。
        """
        parent_node = self.find_node(parent_id)
        if not parent_node:
            raise ValueError(f"Parent node with ID {parent_id} not found.")

        # 如果父节点是叶子，升级为分支节点
        if parent_node["type"] == "leaf":
            parent_node["type"] = "branch"
            parent_node["name"] = operation_details.get("name", "Operation")
            parent_node["operation"] = operation_details

        # 创建新的叶子节点
        permanent_path = self._save_image(new_image_path)
        file_type = os.path.splitext(permanent_path)[1].replace(".", "")
        
        new_node = {
            "id": self._generate_id(),
            "parentId": parent_id,
            "name": file_type.upper(),
            "type": "leaf",
            "file": {
                "path": permanent_path,
                "type": file_type
            },
            "children": []
        }
        
        parent_node["children"].append(new_node)
        return self.tree

    def delete_subtree(self, node_id):
        """删除以指定节点为根的子树，并处理父节点的退化"""
        if not self.tree or self.tree["id"] == node_id:
            # 如果删除的是根节点，整个树就没了
            self.tree = None
            return None

        parent_node = self.find_parent_of(node_id)
        if not parent_node:
            return self.tree # 节点未找到

        # 从父节点的子列表中移除目标节点
        parent_node["children"] = [child for child in parent_node["children"] if child["id"] != node_id]

        # 检查父节点是否需要退化
        if not parent_node["children"]: # 如果没有子节点了
            parent_node["type"] = "leaf"
            # 退化后，名称变回文件类型
            parent_node["name"] = parent_node.get("file", {}).get("type", "File").upper()
            if "operation" in parent_node:
                del parent_node["operation"]

        return self.tree
    
    def find_parent_of(self, node_id, current_node=None):
        """递归查找指定节点的父节点"""
        if current_node is None:
            current_node = self.tree
        
        if not current_node:
            return None

        for child in current_node.get("children", []):
            if child["id"] == node_id:
                return current_node
            found_parent = self.find_parent_of(node_id, child)
            if found_parent:
                return found_parent
        
        return None

    def get_tree(self):
        """获取当前的整个树结构"""
        return self.tree
