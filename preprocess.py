import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config


class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate

    def fix_data(self):
        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'mask'))
        file_list = os.listdir(join(self.raw_root_path, 'ct'))
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)
        for ct_file, i in zip(file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i + 1, Numbers))
            ct_path = os.path.join(self.raw_root_path, 'ct', ct_file)
            filter_path = os.path.join(self.raw_root_path, 'filter', ct_file)
            mask_path = os.path.join(self.raw_root_path, 'mask', ct_file)

            new_ct, new_mask = self.process(ct_path, filter_path, mask_path)
            if new_ct != None and new_mask !=None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
                sitk.WriteImage(new_mask, os.path.join(self.fixed_path, 'mask', ct_file.replace('.nii', '.nii.gz')))

    def process(self, ct_path, filter_path, mask_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkFloat64)
        ct_array = sitk.GetArrayFromImage(ct)
        filter = sitk.ReadImage(filter_path, sitk.sitkFloat64)
        filter_array = sitk.GetArrayFromImage(filter)
        mask = sitk.ReadImage(mask_path, sitk.sitkInt8)
        mask_array = sitk.GetArrayFromImage(mask)



        print("Ori shape:", ct_array.shape, filter_array.shape, mask_array.shape)

        # if classes==2:
        #     # 将金标准中肝脏和肝肿瘤的标签融合为一个
        #     mask_array[mask_array > 0] = 1
        # # 将灰度值在阈值之外的截断掉
        # ct_array[ct_array > self.upper] = self.upper
        # ct_array[ct_array < self.lower] = self.lower
        #
        # filter_array[filter_array > self.upper] = self.upper
        # filter_array[filter_array < self.lower] = self.lower

        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        ct_array = ndimage.zoom(ct_array,
                                (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                order=3)
        mask_array = ndimage.zoom(mask_array,
                                    (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale,
                                     self.xy_down_scale),
                                    order=0)

        # 找到肝脏区域开始和结束的slice，并各向外扩张
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张个slice
        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= mask_array.shape[0]:
            end_slice = mask_array.shape[0] - 1
        else:
            end_slice += self.expand_slice

        print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
        if end_slice - start_slice + 1 < self.size:
            print('Too little slice，give up the sample:', ct_path)
            return None, None
        #截取保留区域
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        mask_array = mask_array[start_slice:end_slice + 1, :, :]

        # slice_data = ct_array[40, :, :]
        # slice_target_branch2 = filter_array[40, :, :]
        # slice_target_branch1 = mask_array[40, :, :]
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

        print("Preprocessed shape:", ct_array.shape, mask_array.shape)

        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                           ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))


        new_mask = sitk.GetImageFromArray(mask_array)
        new_mask.SetDirection(ct.GetDirection())
        new_mask.SetOrigin(ct.GetOrigin())
        new_mask.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                             ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_mask

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, "ct"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            filter_path = os.path.join(self.fixed_path, 'filter', name)
            mask_path = os.path.join(self.fixed_path, 'mask', name)
            f.write(ct_path + ' ' + filter_path + ' ' + mask_path + "\n")
        f.close()


if __name__ == '__main__':
    raw_dataset_path = '/opt/data/private/3DUNet2branch/3DUnet/afterwindow'
    fixed_dataset_path = '/opt/data/private/3DUNet-Pytorch-master2/dataset/fixed_data'

    args = config.args
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args)
    tool.fix_data()  # 对原始图像进行修剪并保存
    tool.write_train_val_name_list()  # 创建索引txt文件
