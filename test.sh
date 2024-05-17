CUDA_VISIBLE_DEVICES=3 python test.py --gpu_id 0 \
--ddim_steps 50 \
--outdir results/try/ \
--config configs/viton512.yaml \
--dataroot /datasets/NVG \
--ckpt checkpoints/mvg.ckpt  \
--n_samples 1 \
--seed 23 \
--scale 1 \
--H 512 \
--W 384 

#!/bin/bash
