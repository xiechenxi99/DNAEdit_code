CUDA_VISIBLE_DIVICES=1 python edit.py --task t2v-1.3B \
--size 832*480 \
--frame_num 41 \
--ckpt_dir ../hf/wan13/ \
--prompt "A large brown bear is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the bear's deliberate movments. " \
--tgt_prompt "A large panda is walking slowly across a rocky terrain in a zoo enclosure, surrounded by stone walls and scattered greenery. The camera remains fixed, capturing the panda's deliberate movments." \
--video_path "data/bear" \
--sample_steps 50


