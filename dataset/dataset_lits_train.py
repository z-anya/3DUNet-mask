from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, sliceCrop

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
                sliceCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                #RandomRotate()
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)#image
        filter = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt16)#filter
        mask = sitk.ReadImage(self.filename_list[index][2], sitk.sitkInt8)#mask

        ct_array = sitk.GetArrayFromImage(ct)
        filter_array = sitk.GetArrayFromImage(filter)
        mask_array = sitk.GetArrayFromImage(mask)

        # slice_data = ct_array[20, :, :]
        # slice_target_branch2 = filter_array[20, :, :]
        # slice_target_branch1 = mask_array[20, :, :]
        #
        # # 使用 matplotlib 查看这个切片
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(slice_data, cmap='gray')
        # plt.show()
        # plt.imshow(slice_target_branch2, cmap='gray')
        # plt.show()
        # plt.imshow(slice_target_branch1, cmap='gray')
        # plt.show()

        min_ct = np.min(ct_array)
        max_ct = np.max(ct_array)
        nom_01_ct = (ct_array - min_ct) / (max_ct - min_ct)
        nom_11_ct = 2 * nom_01_ct - 1
        ct_array = nom_11_ct.astype(np.float32)

        min_filter = np.min(filter_array)
        max_filter = np.max(filter_array)
        nom_01_filter = (filter_array - min_filter) / (max_filter - min_filter)
        nom_11_filter = 2 * nom_01_filter - 1
        filter_array = nom_11_filter.astype(np.float32)

        mask_array = mask_array.astype(np.int8)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        filter_array = torch.FloatTensor(filter_array).unsqueeze(0)
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)

        if self.transforms:
            ct_array,filter_array,mask_array = self.transforms(ct_array, filter_array, mask_array)


        return ct_array, filter_array, mask_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    #sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    print("Train_Dataset length:", len(train_ds))
    print("train_dl length:", len(train_dl))

    for i, (ct, filter, mask) in enumerate(train_dl):
        print(i, ct.size(), filter.size(), mask.size())