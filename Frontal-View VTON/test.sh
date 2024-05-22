# CUDA_VISIBLE_DEVICES=3 python test_global_local.py --gpu_id 0 \
# --ddim_steps 50 \
# --outdir results/dresscode_gp_all_weights/ep6 \
# --config configs/viton512.yaml \
# --dataroot /mnt/pfs-mc0p4k/cvg/team/didonglin/why/datasets/VITON-HD \
# --ckpt /mnt/pfs-mc0p4k/cvg/team/didonglin/why/DCI-VTON-Virtual-Try-On/models/dci-vton-only-full-cloth-wo-zero/2024-03-21T19-03-56_viton512/checkpoints/epoch=000006.ckpt \
# --n_samples 2 \
# --seed 23 \
# --scale 1 \
# --H 1024 \
# --W 768

#!/bin/bash
# 循环遍历变量 i 从 0 到 40
for ((i=10; i<100; i+=1)); do
    CUDA_VISIBLE_DEVICES=7 python test_global_local.py --gpu_id 0 \
    --ddim_steps 50 \
    --outdir results/dresscode_gp_all_weights/ep$i \
    --config configs/viton512.yaml \
    --dataroot /mnt/pfs-mc0p4k/cvg/team/didonglin/why/datasets/Dresscode_upper_body_copy \
    --ckpt /mnt/pfs-mc0p4k/cvg/team/didonglin/why/DCI-VTON-Virtual-Try-On/models/dci-vton-gp-dresscode-all-weights/2024-04-27T20-53-05_viton512/checkpoints/epoch=0000$i.ckpt \
    --n_samples 1 \
    --seed 23 \
    --scale 1 \
    --H 512 \
    --W 384 
done

# pose