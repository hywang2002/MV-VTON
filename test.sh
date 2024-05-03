CUDA_VISIBLE_DEVICES=3 python test.py --gpu_id 0 \
--ddim_steps 50 \
--outdir results/dci-vton-all/2k/ \
--config configs/viton512.yaml \
--dataroot /mnt/pfs-mc0p4k/cvg/team/didonglin/why/datasets/mv_1000_split \
--ckpt /mnt/pfs-mc0p4k/cvg/team/didonglin/why/DCI-VTON-Virtual-Try-On-skip/models/dci-vton-mv1000-warp-code-wo-res-zero-skip-softmax/2024-04-10T11-49-55_viton512/checkpoints/epoch=000006.ckpt \
--n_samples 1 \
--seed 23 \
--scale 1 \
--H 512 \
--W 384 

#!/bin/bash

# code
