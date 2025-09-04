# middletask-1-1-hunyuan
 进度监督会为每周四19:00
## 功能需求
### 基础功能
先实例分割，再重建
### 进阶功能
复杂功能和finetune模型
1. ***处理因遮挡导致的物体分割不全的问题。***
2. ***对于穿衣服的人，人体和衣物分别生成。***
3. ***同时多次点击后区分部件再多个组件同时生成***。分开后的组件之间怎么保证不穿模
## 项目依托
### SAM2/SAM等实例分割demo
实例分割控件
多线程并发（参考SAM2 demo backend）
### HunYuan Gradio Demo
前端和后端都用python实现，上手应该比较简单。
## 分工
王雯睿：汇报，联络；算法和模型
杜斌：组长；算法和模型
刘文博：开发
党浩川：开发
## 项目协作
工作协调会：总结会议：每周三晚上9:00；工作安排会议：每周四晚上9:00
每周汇报：周三开完会后，组长列出提纲，组员填写内容。
服务器的问题。
## 工作流和UI设计
四种编辑功能：SAM2实例分割、身体和服饰分离、Qwen-Image-Edit、混元生成。
工作流记录，方便状态回退。
三栏工作区：图像候选区、图像编辑区、效果预览区。
## 第二周汇总
本周工作：
1、部署、分析SAM2的demo：很遗憾，该demo面向视频流分割设计，且UI与预期效果大相径庭，无法作为二次开发的基础；但对并发需求的处理、消息的封装逻辑可以作为参考（需要具体一下）；
2、拟基于gradio开发：hunyuan的gradio demo可以作为二次开发的基础，通过部署使用确定效果良好，满足需要；gradio支持热加载，方便开发中测试；gradio基于纯python，比较友好；
3、衣物和人体分开生成：首先从图片中分割出完整的角色；然后，基于visual try-on领域的模型，分离人体和衣物；最后，送到hunyuan中生成。
4、处理分割时的遮挡问题：通过qwen-image-edit模型处理；
5、多合一的图像预处理：测试新发布的阿里qwen-image-edit模型，可以通过提示工程完成各种生成需求：遮挡物分离、细节修补，但分割不行（因为是生成模型，不确定性高）。
难点：
1、visual try-on模型，目前找到的模型都面向现实数据集，可能需要面向anime风格重新训练。如果找不到合适的数据集，可能需要大量anime 3D资产（身体和衣物/配饰可以分离的）来构建数据集。
## 第二周工作成果
见周报
## 第三周工作安排
开发侧：
1. 尝试将SAM和hunyuan进行拼接，通过点击功能标签调出相应工作区；
2. 暂时不考虑并发，跑通线形流程再说；
3. 暂定于下周一/二开会讨论前端和后端的细部设计，开发小组内部可以先行调研和分析，与其他成员沟通。

模型侧：
1. 继续调研衣体分离模型，尽量能够适应3D Anime风格。
2. 量化版qwen-image-edit部署；
3. qwen-image-edit提示工程，设置一些示例提示词，能够达到较好的编辑效果；
4. 建立3D Anime图像数据集，风格接近混元Demo中的示例，用于平时的测试。
## 第三周工作安排更新
### 开发侧
葛俊辰：组织开发侧工作，混元编辑功能子页面；
刘文博：文件系统实现；
党浩川：代码解耦和归档、sam1编辑功能子页面
罗雅淇：qwen-image-edit功能子页面；

### 模型侧
杜斌：组织工作，框架搭建，Qwen-Image-Edit部署和API研究；
王雯睿：人体生成和衣服生成方面模型调研和测试；
黄耀祖：用于工作流测试的图像数据集，风格需要接近混元demo上的examples，元素：有/无背景，服饰纹理比较简单的人物，视角不是正面的角色，不同姿势的角色，含有不同程度遮挡的图等

## 第四周
### 人体和衣物生成
基于Qwen的局部重绘（InPainting）
已部署好comfyUI工作流
人体生成流程：
1. 绘制重绘区域（衣物，可以略微超界，方便模型做平滑过渡）
2. 正/负提示词（括号代表权重）：
   (masterpiece, best quality, photorealistic, 3D render:1.2), 1girl, ((perfect anatomy)), ((realistic skin texture)), (((detailed navel))), (clavicle), (slender body, slim waist, toned stomach), (simple white bikini, minimal underwear), grey background
   (clothing, dress, skirt, hanfu, robe, clothes:1.5), pink, red, jewelry, accessories, ribbon, tassel, fabric, pattern, complex patterns, blurry, deformed, mutated, fused fingers, extra limbs
3. 参数微调：蒙版模糊、重回幅度、迭代次数等
4. 生成。根据结果再微调条件、参数，或者二次加工。
最好的方法，是引入VLM来帮助编写Post Json。

你是一位顶级的图像分析师和 AI 绘画（Inpainting）专家。你的任务是基于我提供的用户图片，生成一段高质量、精确且带有权重的、用于指导 Inpainting 模型的英文提示词（Prompt）。

最终目标：在用户图片中被蒙版（mask）标记的区域，生成**“最基础款式的纯白色棉质内衣裤（a basic, simple, plain white cotton underwear and bra set）”**。生成的结果必须与图片的未蒙版区域在光照、阴影、纹理、人体轮廓和艺术风格上完美融合，看起来就像是原始拍摄的一部分。

请严格遵循以下的“思维链”步骤进行分析，并最后给出你的结论。

【思维链分析步骤】

第一步：全局图像分析 (Global Image Analysis)

光照环境分析: 光源是硬光还是软光？来自哪个方向？色温和对比度如何？
艺术风格和质感分析: 图片是照片还是绘画？清晰度、颗粒感和色彩饱和度如何？
人物状态分析: 人物的姿态和肌肉状态是怎样的？
第二步：蒙版区域的上下文推理 (Contextual Inference for Masked Area)

轮廓与形体推理: 根据上下文推断蒙版下方的身体轮廓。
光影衔接推理: 推断光影在将要生成的物体上的表现。
纹理和材质衔接推理: 推断新生成的材质应如何与周围皮肤的质感进行融合。
第三步：核心概念的权重分配 (Weight Assignment for Key Concepts)

识别核心要素 (Identify Core Elements): 在所有描述中，什么是绝对不能错的？什么是次要的修饰？

绝对核心: plain white cotton underwear and bra set (这是生成物的主体)。
高度重要: seamless integration, perfectly blended (这是 Inpainting 任务成功的关键)。
重要细节: soft lighting, subtle fabric texture, natural fit (这些是提升真实感的关键)。
质量标准: photorealistic, hyper-detailed (这些是整体画质的保证)。
分配权重值 (Assign Weights): 根据重要性，为关键短语分配权重。权重值大于 1 表示强调，小于 1 表示减弱。一般使用 1.1 到 1.5 之间的值来强化。

对“无缝衔接”和“完美融合”给予最高权重 (e.g., 1.4-1.5)，因为这是 Inpainting 的首要任务，如果这里失败了，其他都无意义。
对“纯白棉质内衣裤”这个核心主体给予高权重 (e.g., 1.3)，确保模型不会画出其他颜色或材质。
对光照、质感等重要细节给予中等权重 (e.g., 1.1-1.2)，以确保真实感，但又不过分影响主体。
对于负面提示词中需要极力避免的元素（如蕾丝、图案、错误的颜色），也应给予高权重 (e.g., 1.4)，以强制模型避开它们。
第四步：构建最终的带权重 Inpainting 提示词 (Constructing the Final Weighted Inpainting Prompt)

组合与语法: 将加权的短语与普通描述词组合在一起，使用 (keyword:weight) 的语法。
正面提示词 (Positive Prompt): 结合高质量词缀和加权后的核心概念。
负面提示词 (Negative Prompt): 强力排除所有不希望出现的元素，特别是那些与核心目标（纯白、简单）相冲突的。
【最终结论】

请在完成上述所有思考步骤后，基于你的专业分析，生成最终的、可以直接使用的、带有权重的 Inpainting 提示词（包括正面和负面）。

Positive Prompt:
(masterpiece, best quality, 8k, UHD, photorealistic:1.1), hyper-detailed, (seamless integration, perfectly blended with skin:1.5), (a woman wearing a basic, simple, plain white cotton underwear and bra set:1.3), (subtle fabric texture:1.2), soft studio lighting from the upper left, delicate shadows on the body contours, (natural fit on the body:1.2), realistic skin texture with pores and subtle imperfections.

Negative Prompt:
(lace, patterns, logos, text, embroidery, transparent:1.5), (shiny, silk, satin, plastic, leather:1.4), (deformed, distorted, disfigured:1.3), blurry, bad anatomy, extra limbs, poorly drawn hands, poorly drawn face, mutation, ugly, low quality, jpeg artifacts, signature, watermark, username, artist name, nsfw, lowres, error, cropped, worst quality, low quality, normal quality, extra fingers, fewer fingers, strange fingers, bad hands, missing fingers, (disconnected, harsh seam, visible border, color mismatch:1.5).

conda activate Hunyuan
python app.py --sam_device cuda:2 --device cuda:1

conda activate Qwen-Image-Edit
comfy launch -- --enable-cors
