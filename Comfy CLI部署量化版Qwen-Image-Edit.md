## 1.安装Comfy CLI
[ComfyUI Linux Installation Tutorial | ComfyUI Wiki](https://comfyui-wiki.com/en/install/install-comfyui/install-comfyui-on-linux)

# 2.创建Qwen-Image-Edit工作流
```bash
comfy launch -- --enable-cors#启动comfyUI
#通过http://localhost:8081访问
```
在templates->image中找到Qwen-Image-Edit，下载并按指定位置放好模型文件，搭建工作流。

# 3.ComfyUI-API和Gradio前端构建
[comfyui API实战教程：零基础学会工作流转API，实现AI图像批量生成自动化，提升10倍工作效率的完整指南 ai自动化基础教程之comfyui自动化（附提示词） · Issue #10 · aii.mobi/blog](https://cnb.cool/aii.mobi/blog/-/issues/10)
用上面文章的提示词，但实现方式不是双栏网页，而是gradio。AI应该知道怎么处理POST和返回值。
增加其它功能，例如通过文件拖拽实现图像上传、下载，预设提示词。

# 白嫖API
首次开通阿里云百炼时有免费API，大约够300次调用。
[新人免费额度_大模型服务平台百炼(Model Studio)-阿里云帮助中心](https://help.aliyun.com/zh/model-studio/new-free-quota?spm=5176.28197581.d_index.4.145d29a4kKhJoe)
测试模型时可以使用web chat，等开发弄好上API。

# UI
主页面与多个编辑功能子页面（工作区）组成，通过缓存文件夹共享数据（主要就是上传图像/下载图像）
主页面：左侧是文件预览区，右侧工作区
工作区：通过在导航栏点选功能按钮切换编辑功能（加载子页面）
文件预览区：文件树的形式呈现历史编辑记录，文件树上方放个小窗用于预览。
数据交互：文件预览区，选中一张图，点击“确认”，后台从缓存区将图像上传到工作区；工作区，选中已加工好图像，点击“下载”，后台下载到缓存区。
缓存数据区分：首先，按“时间戳-用户自定义名称”为文件命名，后台维护一个json文件，保存结构。每次下载文件到缓存区，都会出发文件预览区刷新。