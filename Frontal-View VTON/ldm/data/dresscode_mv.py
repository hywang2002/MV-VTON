# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from numpy.linalg import lstsq

from src.utils.labelmap import label_map
from src.utils.posemap import kpoint_to_heatmap


class DressCodeDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 image_size=512, mode='train', semantic_nc=13, unpaired=False,
                 radius=5,
                 order='paired',
                 outputlist=None,
                 size: Tuple[int, int] = (512, 512),
                 ):

        super().__init__()
        if outputlist is None:
            outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']
        self.dataroot = dataroot_path
        self.phase = mode
        self.category = 'upper_body'
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        folders = []
        for folder in os.listdir(os.path.join(self.dataroot, 'image')):
            folders.append(folder)
        folders = sorted(folders)

        self.folders = folders

    def __getitem__(self, index):
        folder = self.folders[index]
        category = 'upper_body'
        imgs = []
        for img in os.listdir(os.path.join(self.dataroot, 'image', folder)):
            imgs.append(img)
        imgs = sorted(imgs)

        c_name = imgs[0]
        im_name = imgs[1]
        dataroot = self.dataroot

        if "cloth" in self.outputlist:  # In-shop clothing image
            # Clothing image
            cloth = Image.open(os.path.join(dataroot, 'cloth', folder, c_name))
            mask = Image.open(os.path.join(dataroot, 'cloth-mask', folder, c_name))  # 我们的是jpg

            # Mask out the background
            cloth = Image.composite(ImageOps.invert(mask.convert('L')), cloth, ImageOps.invert(mask.convert('L')))
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            # Person image
            image = Image.open(os.path.join(dataroot, 'image', folder, im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist or \
                "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist or "pose_map" in \
                self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist:
            # Label Map
            parse_name = im_name.replace('.jpg', '.png')
            im_parse = Image.open(os.path.join(dataroot, 'parsing', folder, parse_name))
            im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(np.float32)

            parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            label_cat = 4
            parse_cloth = (parse_array == 4).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                 (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
            shape = self.transform2D(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('.jpg', '.json')
            with open(os.path.join(dataroot, 'keypoints_18_copy', folder, pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []
            for pose_d in pose_data:
                ux = pose_d[0] / 384.0
                uy = pose_d[1] / 512.0

                # scale posemap points
                px = ux * self.width
                py = uy * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            # if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body' or dataroot.split('/')[
            #     -1] == 'lower_body':
            with open(os.path.join(dataroot, 'keypoints_18_copy', folder, pose_name), 'r') as f:
                data = json.load(f)
                shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                # if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                parse_mask += im_arms
                parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            # if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
            with open(os.path.join(dataroot, 'keypoints_18_copy', folder, pose_name), 'r') as f:
                data = json.load(f)
                points = []
                points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                x_coords, y_coords = zip(*points)
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]
                for i in range(parse_array.shape[1]):
                    y = i * m + c
                    parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            inpaint_mask = inpaint_mask.unsqueeze(0)
            parse_mask_total = parse_mask_total.numpy()
            parse_mask_total = parse_array * parse_mask_total
            parse_mask_total = torch.from_numpy(parse_mask_total)

        # if "dense_uv" in self.outputlist:
        #     dense_uv = np.load(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5_uv.npz')))
        #     dense_uv = dense_uv['uv']
        #     dense_uv = torch.from_numpy(dense_uv)
        #     dense_uv = transforms.functional.resize(dense_uv, (self.height, self.width), antialias=True)
        #
        # if "dense_labels" in self.outputlist:
        #     labels = Image.open(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
        #     labels = labels.resize((self.width, self.height), Image.NEAREST)
        #     labels = np.array(labels)
        im_name = folder + '_' + im_name
        result = {
            "GT": image,
            "inpaint_image": im_mask,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": cloth,
            'warp_feat': im_mask,
            "file_name": im_name
        }

        return result

    def __len__(self):
        return len(self.folders)


if __name__ == '__main__':
    outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']
    dataset = DressCodeDataset('/hdd1/why/datasets/multiview_split',
                               phase='test',
                               order='paired',
                               radius=5,
                               outputlist=outputlist,
                               category='upper_body',
                               size=(512, 384))

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in loader:
        print("==")
        pass
