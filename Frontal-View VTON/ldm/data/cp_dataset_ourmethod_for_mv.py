# coding=utf-8
# cp_dataset_ourmethod_for_mv.py
import os

import PIL
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json

import random
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader


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


class CPDataset(data.Dataset):
    """
        Dataset for CP-VTON.
    """

    def __init__(self, dataroot, image_size=512, mode='train', semantic_nc=13, unpaired=False):
        super(CPDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.unpaired = unpaired
        self.datamode = mode  # train or test or self-defined
        self.data_list = mode + '_pairs.txt'
        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 256)
        self.semantic_nc = semantic_nc
        self.data_path = osp.join(dataroot, mode)
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])

        # load data list
        folders = []
        # c_names = []

        for folder in os.listdir(osp.join(self.data_path, "image-wo-bg")):
            folders.append(folder)
        folders = sorted(folders)  # 00000_1, 00000_2, 00000_3, ...

        # with open(osp.join(dataroot, self.data_list), 'r') as f:
        #     for line in f.readlines():
        #         im_name, c_name = line.strip().split()
        #         im_names.append(im_name)
        #         c_names.append(c_name)

        self.folders = folders
        # self.c_names = dict()
        # self.c_names['paired'] = im_names
        # self.c_names['unpaired'] = im_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        folder = self.folders[index]  # 00001_1
        order = folder.split('_')[1]

        im_names = []
        for im_name in os.listdir(osp.join(self.data_path, 'image-wo-bg', folder)):
            im_names.append(im_name)
        im_names = sorted(im_names)  # [1.jpg, 2.jpg, 5.jpg]
        if len(im_names) !=3 :
            raise ValueError("folder {} imgs num != 3".format(folder))
        # print("====")
        # print(im_names)
        # person image
        im_pil_big = Image.open(osp.join(self.data_path, 'image-wo-bg', folder, im_names[1]))
        im_pil = transforms.Resize(self.crop_size, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # agnostic image
        inpaint_big = Image.open(osp.join(self.data_path, 'agnostic-wo-bg', folder, im_names[1]))
        inpaint = transforms.Resize(self.crop_size, interpolation=2)(inpaint_big)
        inpaint = self.transform(inpaint)

        # inpaint mask
        inpaint_mask_big = Image.open(osp.join(self.data_path, "inpaint_mask", folder, im_names[1])).convert('L')
        inpaint_mask = transforms.Resize(self.crop_size, interpolation=2)(inpaint_mask_big)
        inpaint_mask = self.transform_mask(inpaint_mask)

        # ref_image_f
        c = Image.open(osp.join(self.data_path, 'cloth', folder, im_names[0]))
        c = transforms.Resize(self.crop_size, interpolation=2)(c)
        c = self.transform(c)

        cm = Image.open(osp.join(self.data_path, 'cloth-mask', folder, im_names[0]))
        cm = transforms.Resize(self.crop_size, interpolation=0)(cm)

        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)
        cm.unsqueeze_(0)

        down, up, left, right = mask2bbox(cm[0].numpy())
        ref_image_f = c[:, down:up, left:right]
        ref_image_f = (ref_image_f + 1.0) / 2.0
        ref_image_f = transforms.Resize((224, 224))(ref_image_f)
        ref_image_f = self.clip_normalize(ref_image_f)

        feat = inpaint

        result = {
            "GT": im,  # [-1,1]
            "inpaint_image": inpaint,  # mask 身体 手臂 手 [128,128,128], [-1,1]
            "inpaint_mask": 1.0 - inpaint_mask,  # [0,1]
            "ref_imgs": ref_image_f,  # clip_normalize
            'warp_feat': feat,  # [-1,1]
            "file_name": self.folders[index] + '.jpg'
        }
        return result

    def __len__(self):
        return len(self.folders)

if __name__ == '__main__':
    dataset = CPDataset('/mnt/pfs-mc0p4k/cvg/team/didonglin/why/datasets/mv_800_split', 512, mode='test', unpaired=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in loader:
        print("==")
        pass
