from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from dataset.transforms import Center_Crop, Compose


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        filter = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt16)  # filter
        mask = sitk.ReadImage(self.filename_list[index][2], sitk.sitkInt16)

        ct_array = sitk.GetArrayFromImage(ct)
        filter_array = sitk.GetArrayFromImage(filter)
        mask_array = sitk.GetArrayFromImage(mask)

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
            ct_array, filter_array, mask_array = self.transforms(ct_array, filter_array, mask_array)

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

# if __name__ == "__main__":
#     #sys.path.append('/ssd/lzq/3DUNet')
#     from config import args
#     val_ds = Val_Dataset(args)
#
#     # 定义数据加载
#     val_dl = DataLoader(val_ds, 2, False, num_workers=1)
#     print("Val_Dataset length:", len(val_ds))
#     print("val_dl length:", len(val_dl))
#
#     for i, (ct, filter, mask) in enumerate(val_dl):
#         print(i,ct.size(),filter.size(), mask.size())