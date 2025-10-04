import gradio as gr
import numpy as np
from PIL import Image
import time
import torch
from diffusers import FluxPipeline
from scripts.DNAEdit_utils import DNAEdit_FLUX

def edit_image(image, source_prompt, target_prompt, t_step, mvg, t_start, cfg):
    """
    图像编辑函数
    
    参数:
        image: 输入的PIL图像
        source_prompt: 源提示词
        target_prompt: 目标提示词
        t_step: T_step参数
        mvg: MVG参数
        t_start: T_start参数
        cfg: CFG参数
        progress: Gradio进度条对象
    
    返回:
        edited_image: 编辑后的PIL图像
    """
    if image is None:
        gr.Warning("请先上传图片！")
        return None
    
    if not source_prompt or not target_prompt:
        gr.Warning("请输入源提示词和目标提示词！")
        return None
    #
    global loaded_pipe
    
    # 检查模型是否已加载
    if loaded_pipe is None:
        gr.Warning("请先加载模型！")
        return None
    pipe = loaded_pipe
    # 总迭代步数（根据你的模型调整）
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    original_width, original_height = image.size
    image = image.resize((original_width,original_height),Image.BILINEAR)
    image_src = pipe.image_processor.preprocess(image)
    image_src = image_src.to("cuda").half()
    with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # send to cuda
    x0_src = x0_src.to("cuda")
    x0_tar = DNAEdit_FLUX(loaded_pipe,
                                                        pipe.scheduler,
                                                        x0_src,
                                                        source_prompt,
                                                        target_prompt,
                                                        "",
                                                        t_step,
                                                        1,
                                                        cfg,
                                                        mvg=mvg,
                                                        T_start=t_start)
    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)
    image_tar = image_tar[0].resize((original_width,original_height),Image.BILINEAR)
    edited_image = image_tar
    
    # ========================================================================
    
    return edited_image



loaded_pipe = None
loaded_scheduler = None

def load_model(model_type, progress=gr.Progress()):
    """
    加载模型函数
    
    参数:
        model_choice: 选择的模型名称
        progress: Gradio进度条对象
    
    返回:
        status_message: 加载状态信息
    """
    global loaded_pipe, loaded_scheduler
    
    try:
        progress(0.1, desc="正在初始化模型...")
        if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
            pipe = FluxPipeline.from_pretrained("/home/notebook/data/personal/S9055029/hf/FLUX.1-dev", torch_dtype=torch.float16)
        elif model_type == 'SD3':
            pipe = StableDiffusion3Pipeline.from_pretrained("/home/notebook/data/personal/S9055029/hf/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        pipe.to("cuda")
        loaded_pipe=pipe
        loaded_scheduler = pipe.scheduler

        

        
        # ========================================================================
        
        return f"✅ 模型 '{model_choice}' 加载成功！"
        
    except Exception as e:
        loaded_pipe = None
        loaded_scheduler = None
        return f"❌ 模型加载失败: {str(e)}"



# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 DNAEdit Demo
        """
    )
    
    # 模型加载区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 模型配置")
            model_choice = gr.Radio(
                choices=["FLUX"],
                value="FLUX",
                label="选择模型",
                info="选择要使用的扩散模型"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ")
            load_model_btn = gr.Button(
                "⬇️ 加载模型",
                variant="primary",
                size="lg"
            )
            model_status = gr.Textbox(
                label="模型状态",
                value="⚪ 未加载模型",
                interactive=False,
                lines=1
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 📤 输入图片")
            input_image = gr.Image(
                label="上传图片",
                type="pil",
                height=350
            )
            
            gr.Markdown("### 📝 提示词设置")
            source_prompt = gr.Textbox(
                label="Source Prompt（源提示词）",
                placeholder="描述原始图片的内容，例如：a photo of a cat",
                lines=2,
                info="描述输入图片当前的内容"
            )
            
            target_prompt = gr.Textbox(
                label="Target Prompt（目标提示词）",
                placeholder="描述期望编辑后的内容，例如：a photo of a dog",
                lines=2,
                info="描述你想要编辑成的目标内容"
            )
            
            gr.Markdown("### ⚙️ 编辑参数")
            
            with gr.Accordion("参数设置", open=True):
                t_step = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=28,
                    step=1,
                    label="T_step",
                    info="控制编辑迭代的总步数"
                )
                
                mvg = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="MVG系数",
                    info="越大的值图片编辑力度越大"
                )
                
                t_start = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=4,
                    step=1,
                    label="T_start",
                    info="越小的值图片编辑力度越大"
                )
                
                cfg = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=2,
                    step=0.5,
                    label="CFG",
                    info="越大的值图片编辑力度越大"
                )
            
            edit_btn = gr.Button(
                "🚀 开始编辑",
                variant="primary",
                size="lg"
            )
        
        # 右侧：输出区域
        with gr.Column(scale=1):
            gr.Markdown("### 📥 编辑结果")
            output_image = gr.Image(
                label="编辑后的图片",
                type="pil",
                height=350
            )
            
            gr.Markdown("### 📊 当前配置")
            with gr.Accordion("配置详情", open=True):
                config_display = gr.Markdown()
    
    # 更新配置显示
    def update_config(source_prompt, target_prompt, t_step, mvg, t_start, cfg):
        return f"""
        **提示词配置：**
        - **Source Prompt**: {source_prompt if source_prompt else '未设置'}
        - **Target Prompt**: {target_prompt if target_prompt else '未设置'}
        
        **参数配置：**
        - **T_step**: {t_step}
        - **MVG**: {mvg}
        - **T_start**: {t_start}
        - **CFG**: {cfg}
        """
    
    # 监听所有参数变化
    for param in [source_prompt, target_prompt, t_step, mvg, t_start, cfg]:
        param.change(
            fn=update_config,
            inputs=[source_prompt, target_prompt, t_step, mvg, t_start, cfg],
            outputs=config_display
        )
    
    # 加载模型按钮点击事件
    load_model_btn.click(
        fn=load_model,
        inputs=[model_choice],
        outputs=model_status
    )
    
    # 点击编辑按钮
    edit_btn.click(
        fn=edit_image,
        inputs=[input_image, source_prompt, target_prompt, t_step, mvg, t_start, cfg],
        outputs=output_image
    )
    
    # 示例
    gr.Markdown("### 💡 使用提示")
    gr.Markdown(
        """
        1. 首先在顶部 **选择模型** 并点击 **加载模型** 按钮
        2. 等待模型加载完成（状态显示为 ✅）
        3. 点击左侧 **上传图片** 区域选择要编辑的图片
        4. 在 **Source Prompt** 中输入描述原始图片内容的提示词
        5. 在 **Target Prompt** 中输入描述目标编辑结果的提示词
        6. 调整编辑参数以获得最佳效果
        7. 点击 **开始编辑** 按钮启动编辑过程
        8. 编辑过程中可以看到进度条显示
        9. 编辑完成后，结果将显示在右侧
        
        **示例提示词：**
        - Source: "a photo of a cat on the grass"
        - Target: "a photo of a dog on the grass"
        """
    )
    


# 启动应用
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )