# --- coding: utf-8 ---
# @create time: 2023-09-25 17:32
# @author: GunZ
# @software: PyCharm


import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Literal
import json
import numpy as np
import torch

import sys
from pathlib import Path
import random
from einops import rearrange


PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)


class MultiViewDataset(Dataset):
    def __init__(self,
                 dataroot_path: str,
                 phase: Literal['train', 'test'],
                 radius=5,
                 order='paired',
                 outputlist: Tuple[str] = (
                         'gt', 'frontal_cloth', 'back_cloth', 'inpaint_mask', 'im_mask', 'skeleton', 'gt_name'),
                 size: Tuple[int, int] = (512, 384),
                 # label_num=3
                 ):
        self.dataroot_path = dataroot_path
        self.phase = phase
        self.radius = radius
        self.height = size[0]
        self.width = size[1]
        self.outputlist = outputlist
        # self.label_num = label_num
        # self.all_list = [0, 1, 2, 3, 4]
        # self.cloth_list = [0, 1, 2, 3, 4]
        # self.cloth_list.remove(self.label_num)

        self.outputlist = outputlist
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

        self.names = []  # 00000_1, 00000_2, 00000_3, 00001_1, 00001_2, 00001_3, ...
        for file_name in os.listdir(os.path.join(self.dataroot_path, "image")):
            self.names.append(file_name)
        self.names = sorted(self.names)
        if self.phase == "train":
            self.folder_list = [s for i, s in enumerate(self.names) if i % 15 != 6 and i % 15 != 7 and i % 15 != 8 ]
        elif self.phase == "test":
            self.folder_list = [s for i, s in enumerate(self.names) if i % 15 == 6 or i % 15 == 7 or i % 15 == 8]

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]  # 00000_1
        gt_name = folder_name[-1]
        file_name_dci = folder_name + '.jpg'
        category = 'upper_body'

        file_name = []  # folder_name 下的文件 [1.jpg, 2.jpg, 5.jpg]
        for file in os.listdir(os.path.join(self.dataroot_path, "image", folder_name)):
            file_name.append(file)
        file_name = sorted(file_name)

        # cropped cloth [-1,1]
        frontal_cloth_path = os.path.join(self.dataroot_path, "cloth", folder_name, file_name[0])
        frontal_cloth = self.transform(Image.open(frontal_cloth_path).convert('RGB').resize((self.width, self.height)))

        back_cloth_path = os.path.join(self.dataroot_path, "cloth", folder_name, file_name[2])
        back_cloth = self.transform(Image.open(back_cloth_path).convert('RGB').resize((self.width, self.height)))

        # image_agnostic [-1,1]
        image_agnostic_path = os.path.join(self.dataroot_path, "image_agnostic", folder_name, file_name[1])
        inpaint_image = self.transform(
            Image.open(image_agnostic_path).convert('RGB').resize((self.width, self.height)))

        # label [-1,1]
        label_path = os.path.join(self.dataroot_path, "image", folder_name, file_name[1])
        GT = self.transform(Image.open(label_path).convert('RGB').resize((self.width, self.height)))

        # skeleton [-1,1]
        skeleton_path = os.path.join(self.dataroot_path, "skeletons", folder_name, file_name[1])
        skeleton = self.transform(Image.open(skeleton_path).convert('RGB').resize((self.width, self.height)))

        # mask [0,1]
        mask_path = os.path.join(self.dataroot_path, "mask", folder_name, file_name[1])
        inpaint_mask = self.transform_mask(Image.open(mask_path).convert('L').resize((self.width, self.height)))

        cm_path = os.path.join(self.dataroot_path, "cloth-mask", folder_name, file_name[0])
        cm = self.transform_mask(Image.open(mask_path).convert('L').resize((self.width, self.height)))

        down, up, left, right = mask2bbox(cm[0].numpy())
        ref_image = frontal_cloth[:, down:up, left:right]
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = transforms.Resize((224, 224))(ref_image)

        # ref_image_np = 255. * rearrange(ref_image.cpu().numpy(), 'c h w -> h w c')
        # img = Image.fromarray(ref_image_np.astype(np.uint8))
        # img.save(os.path.join("/hdd1/why/DCI-VTON-Virtual-Try-On/results/mv_1/result", file_name_dci[:-4] + "_cloth.png"))

        # max_value = torch.max(ref_image)
        # min_value = torch.min(ref_image)

        # print("最大值:", max_value.item())
        # print("最小值:", min_value.item())

        ref_imgs = self.clip_normalize(ref_image)


        # # keypoints [0,1]
        # keypoints = []
        # for i in self.all_list:
        #     keypoints_path = os.path.join(self.dataroot_path, "keypoints", folder_name,
        #                                   file_name[i].split('.')[0] + '.json')
        #     with open(keypoints_path, 'r') as f:
        #         pose_data = json.load(f)
        #         # pose_data = pose_data.reshape((-1, 4))
        #         keypoints.append(pose_data)
        # 
        # cloth_pose = []
        # 
        # for i in self.cloth_list:
        #     d1 = []
        #     for pose_d in keypoints[i]:
        #         ux = pose_d[0] / 384.0
        #         uy = pose_d[1] / 512.0
        # 
        #         # scale posemap points
        #         px = ux * self.width
        #         py = uy * self.height
        #         d1.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))
        #     d1 = torch.stack(d1)
        #     d1 = padding_tensor(d1, (21, 512, 384))
        #     cloth_pose.append(d1)
        # 
        # d = []
        # for pose_d in keypoints[self.label_num]:
        #     ux = pose_d[0] / 384.0
        #     uy = pose_d[1] / 512.0
        # 
        #     # scale posemap points
        #     px = ux * self.width
        #     py = uy * self.height
        #     d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))
        # d = torch.stack(d)
        # agnostic_pose = padding_tensor(d, target_size=(21, 512, 384))

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result
