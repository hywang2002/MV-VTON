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

# # 循环遍历变量 i 从 0 到 40
# for ((i=2; i<10; i++)); do
#     CUDA_VISIBLE_DEVICES=6 python test.py --gpu_id 0 \
#     --ddim_steps 50 \
#     --outdir results/dci-vton-local-mv-unpaired/ep$i \
#     --config configs/viton512.yaml \
#     --dataroot /mnt/pfs-mc0p4k/cvg/team/didonglin/why/datasets/mv_1000_split \
#     --ckpt /mnt/pfs-mc0p4k/cvg/team/didonglin/why/DCI-VTON-Virtual-Try-On-skip/models/dci-vton-mv1000-only-local/2024-04-15T14-28-50_viton512/checkpoints/epoch=00000$i.ckpt \
#     --n_samples 1 \
#     --seed 23 \
#     --scale 1 \
#     --H 512 \
#     --W 384 
# done

# code