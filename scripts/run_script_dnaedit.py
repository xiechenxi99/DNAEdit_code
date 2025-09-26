import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
import json
import time
from DNAEdit_utils import DNAEdit_SD3, DNAEdit_FLUX

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="FLUX_exp.yaml", help="experiment yaml file")
    parser.add_argument("--save",type=str)
    args = parser.parse_args()

    # set device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model_type = exp_configs[0]["model_type"] # currently only one model type per run

    if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
        pipe = FluxPipeline.from_pretrained("/home/notebook/data/personal/S9055029/hf/FLUX.1-dev", torch_dtype=torch.float16)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("/home/notebook/data/personal/S9055029/hf/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    scheduler = pipe.scheduler
    pipe = pipe.to(device)

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        # model_type = exp_dict["model_type"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        with open('PIE-bench/long_mapping_file.json','r',encoding='utf-8') as f:
            dataset_dict= json.load(f)

        for data_dict in dataset_dict.values():
            for jmp in [13]:
                src_prompt = data_dict["original_prompt"]
                tar_prompt = data_dict["editing_prompt"]
                print(src_prompt)
                print(tar_prompt)
                negative_prompt =  "" # optionally add support for negative prompts (SD3)
                image_src_path = data_dict["image_path"]
                image_src_path = os.path.join("PIE-bench/annotation_images",image_src_path)
                # load image
                image = Image.open(image_src_path)
                # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                original_width, original_height = image.size
                image_relative=data_dict["image_path"]
                # if os.path.exists(f"outputs/{args.save}/{image_relative}"):
                #     print("exist")
                #     continue
                image = image.resize((original_width,original_height),Image.BILINEAR)
                image_src = pipe.image_processor.preprocess(image)
                # cast image to half precision
                image_src = image_src.to(device).half()
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                # send to cuda
                x0_src = x0_src.to(device)
                start_time = time.time()



                if model_type == 'SD3':
                    x0_tar = DNAEdit_SD3(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,
                                                            jmp=jmp)
                    
                elif model_type == 'FLUX':
                    x0_tar = DNAEdit_FLUX(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,)
                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented")

                end_time = time.time()

                print(f"函数执行耗时: {end_time - start_time:.6f} 秒")
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)
                
                # make sure to create the directories before saving
                save_dir = f"outputs/{args.save}/"
                save_dir = os.path.join(save_dir,"/".join(data_dict["image_path"].split('/')[:-1]))
                os.makedirs(save_dir, exist_ok=True)
                image_tar = image_tar[0].resize((original_width,original_height),Image.BILINEAR)
                image_tar.save(f"{save_dir}/{data_dict['image_path'].split('/')[-1]}")
                




    print("Done")

    # %%
