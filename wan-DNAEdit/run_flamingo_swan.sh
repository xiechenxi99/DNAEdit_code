python edit.py --task t2v-1.3B \
--size 832*480 \
--frame_num 41 \
--ckpt_dir ../hf/wan13/ \
--prompt "A black swan is gliding smoothly across a calm pond, surrounded by lush green foliage. The camera remains fixed, capturing the swan's serene movement. " \
--tgt_prompt "A pink flamingo is gliding smoothly across a calm pond, surrounded by lush green foliage. The camera remains fixed, capturing the swan's serene movement." \
--video_path "data/blackswan" \
--sample_steps 50


