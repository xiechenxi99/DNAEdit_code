from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
import ipdb
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn.functional as F


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



@torch.no_grad()
def DNAEdit_SD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    jmp:int = 6,
    ratio : float = 0.85):
    
    device = x_src.device
    orig_height, orig_width = x_src.shape[2], x_src.shape[3]
    num_channels_latents = pipe.transformer.config.in_channels // 4

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    # ipdb.set_trace()
    # CFG prep
    pipe._guidance_scale = src_guidance_scale

    if pipe.do_classifier_free_guidance:
        src_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds], dim=0)
        src_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds], dim=0)
    pipe._guidance_scale = tar_guidance_scale
    if pipe.do_classifier_free_guidance:
        tar_prompt_embeds = torch.cat([ tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
        tar_pooled_prompt_embeds = torch.cat([tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    # initialize our ODE Zt_edit_1=x_src

    timesteps=torch.cat([timesteps,torch.tensor([0],device=timesteps.device)])
    inver_timesteps=torch.flip(timesteps,dims=[0])

    jmp=jmp
    last = x_src.clone()
    last_lst=[]
    v_lst=[]
    comp_lst=[]
    random_noise=torch.randn_like(x_src)
    dx_lst=[]
    for i,(t_curr,t_prev) in enumerate(zip(inver_timesteps[:-1],inver_timesteps[1:])):
        # ipdb.set_trace()
        if len(inver_timesteps)-1-i==jmp:
            break
        # scheduler._init_step_index(t)
        # t_i = scheduler.sigmas[scheduler.step_index]
        # if i < len(timesteps):
        #     t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        # else:
        #     t_im1 = t_i
        t_curr,t_prev = t_curr/1000.0,t_prev/1000.0
        x_curr = last
        x_prev = (t_prev-t_curr)/(1-t_curr)*(random_noise-last)+last
        k=1
        dx = None
        for i in range(k):
            if pipe.do_classifier_free_guidance:
                src_latent_model_input=torch.cat([x_prev,x_prev],dim=0)
            else:
                src_latent_model_input=x_prev
            with torch.no_grad():
                noise_pred_src = pipe.transformer(
                    hidden_states=src_latent_model_input,
                    timestep=(t_prev*1000).expand(src_latent_model_input.shape[0]),
                    encoder_hidden_states=src_prompt_embeds,
                    pooled_projections=src_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            if pipe.do_classifier_free_guidance:
                src_noise_pred_uncond, src_noise_pred_text = noise_pred_src.chunk(2)
                noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
        
        # print(noise_pred_src.mean(),noise_pred_src.var())
            delta_v = ((x_prev-x_curr)/(t_prev-t_curr)-noise_pred_src)/k
            l2_norm = torch.norm(delta_v, p=2)
            
        # print(random_noise.mean(),random_noise.var())
        # print(last.mean(),last.var())
            x_prev=x_prev.to(torch.float32)
            last =x_prev-delta_v*(t_prev-t_curr)
            dx=delta_v*(t_prev-t_curr)
            last = last.to(delta_v.dtype)
            x_prev = last.clone()
        # print(last.mean(),last.var())
            random_noise=random_noise.to(torch.float32)
            random_noise-=delta_v*(1-t_curr)
            random_noise=random_noise.to(delta_v.dtype)
        # print(random_noise.mean(),random_noise.var())
        with torch.no_grad():
            # ipdb.set_trace()
            noise_pred_tgt = pipe.transformer(
                hidden_states=torch.cat([last,last],dim=0),
                timestep=(t_prev*1000).expand(src_latent_model_input.shape[0]),
                encoder_hidden_states=tar_prompt_embeds,
                pooled_projections=tar_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            tgt_noise_pred_uncond, tgt_noise_pred_text = noise_pred_tgt.chunk(2)
            noise_pred_tgt = tgt_noise_pred_uncond + tar_guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)
        
        last_lst.append(last)
        dx_lst.append(dx)
        v_lst.append(noise_pred_src)
    last_lst=last_lst[::-1]
    v_lst = v_lst[::-1]
    dx_lst = dx_lst[::-1]
    random_noise=last_lst[0]
    x_ref=x_src.clone()
    pipe._guidance_scale = tar_guidance_scale

    for i,(t_curr,t_prev) in enumerate(zip(timesteps[:-1],timesteps[1:])):
        if i<jmp:
            continue
        timestep = t_curr
        # ipdb.set_trace()
        t_curr,t_prev = t_curr/1000.0,t_prev/1000.0
        if pipe.do_classifier_free_guidance:
            tgt_latent_model_input=torch.cat([random_noise+dx_lst[i-jmp],random_noise+dx_lst[i-jmp]],dim=0)
        else:
            tgt_latent_model_input=random_noise+dx_lst[i-jmp]
        with torch.no_grad():
            # ipdb.set_trace()
            noise_pred_tgt = pipe.transformer(
                hidden_states=tgt_latent_model_input,
                timestep=timestep.expand(tgt_latent_model_input.shape[0]),
                encoder_hidden_states=tar_prompt_embeds,
                pooled_projections=tar_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        if pipe.do_classifier_free_guidance:
            tgt_noise_pred_uncond, tgt_noise_pred_text = noise_pred_tgt.chunk(2)
            noise_pred_tgt = tgt_noise_pred_uncond + tar_guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)
        x=ratio
        x_ref = x_ref.to(torch.float32)
        x_ref=x_ref+(t_prev-t_curr)*(noise_pred_tgt-v_lst[i-jmp])
        x_ref = x_ref.to(noise_pred_tgt.dtype)
        v = noise_pred_tgt*x+(random_noise-x_ref)/t_curr*(1-x)
        # v = noise_pred_tgt
        random_noise=random_noise.to(torch.float32)
        # random_noise = scheduler.step(noise_pred_tgt, timestep, random_noise, return_dict=False)[0]
        random_noise+=v*(t_prev-t_curr)
        random_noise=random_noise.to(v.dtype)
    return random_noise




