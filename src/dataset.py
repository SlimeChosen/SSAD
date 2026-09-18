import os
import sys
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import  train_test_split
import SimpleITK as sitk
import tqdm
from scipy.ndimage import distance_transform_edt

import yaml
from src.utils import get_mask_edge, voxel2points


def split_data(label_path, save_path):
    all_paths = glob.glob(os.path.join(label_path, '*.npz'))

    train_paths, test_paths = train_test_split(all_paths, test_size=0.2)

    dataset_dict = {
        'train': train_paths,
        'test': test_paths,
    }

    print("train:" + str(len(train_paths)))
    print("test:" + str(len(test_paths)))

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=4, ensure_ascii=False)


class SSAD_Dataset(Dataset):
    def __init__(self, json_path, label_id):
        with open(json_path, 'r', encoding='utf-8') as f:
            data_paths = json.load(f)['train']

        self.all_voxel = []
        self.label_id = label_id

        for path in tqdm.tqdm(data_paths, desc='initializing data'):
            data = np.load(path)
            voxel = data['voxel']

            voxel[voxel != self.label_id] = 0
            voxel[voxel > 0] = 1

            voxel = self.crop_voxel(voxel)

            self.all_voxel.append(voxel)

    def __getitem__(self, index):
        voxel = self.all_voxel[index]

        return voxel

    def __len__(self):
        return len(self.all_voxel)

    def crop_voxel(self, voxel):
        dhw = np.where(voxel > 0)

        d0 = max(0, min(min(dhw[0]) - 1, voxel.shape[0]))
        d1 = max(d0, min(max(dhw[0]) + 1, voxel.shape[0]))

        h0 = max(0, min(min(dhw[1]) - 1, voxel.shape[1]))
        h1 = max(h0, min(max(dhw[1]) + 1, voxel.shape[1]))

        w0 = max(0, min(min(dhw[2]) - 1, voxel.shape[2]))
        w1 = max(w0, min(max(dhw[2]) + 1, voxel.shape[2]))

        voxel = voxel[d0 : d1 + 1, h0 : h1 + 1, w0 : w1 + 1]

        return voxel

    @staticmethod
    def reorient_image(image):
        current_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())

        if current_orientation != 'LPS':
            image = sitk.DICOMOrient(image, 'LPS')

        return image

    @staticmethod
    def get_tar_spacing(image, tar_size=128):
        current_spacing = image.GetSpacing()
        current_size = image.GetSize()

        scale = current_size[0] / tar_size

        new_spacing_x = current_spacing[0] * scale
        new_spacing_y = current_spacing[1] * scale
        new_spacing_z = current_spacing[2] * scale

        new_spacing = (new_spacing_x, new_spacing_y, new_spacing_z)
        new_size = (tar_size, tar_size, int(current_size[2] / scale))

        return new_spacing, new_size

    @staticmethod
    def resample_volumeImage(volume_image, new_spacing, new_size, is_mask=True):
        resample_filter = sitk.ResampleImageFilter()

        if is_mask is True:
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample_filter.SetInterpolator(sitk.sitkLinear)

        resample_filter.SetOutputDirection(volume_image.GetDirection())
        resample_filter.SetOutputOrigin(volume_image.GetOrigin())
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(new_spacing)

        new_volume_image = resample_filter.Execute(volume_image)

        if is_mask is True:
            array = sitk.GetArrayFromImage(new_volume_image)
            label = sitk.GetImageFromArray(array)
            label.SetOrigin(new_volume_image.GetOrigin())
            label.SetSpacing(new_volume_image.GetSpacing())
            label.SetDirection(new_volume_image.GetDirection())
            new_volume_image = label

        return new_volume_image

    @staticmethod
    def proprocess_data(image_dir, labels_dir, out_dir, tar_size=128, split_json='../config/data.json'):

        image_reorient_save_dir = out_dir + '/0_image_reorient'
        label_reorient_save_dir = out_dir + '/0_label_reorient'

        image_resize_save_dir = out_dir + '/1_image_resize'
        label_resize_save_dir = out_dir + '/1_label_resize'

        npz_save_folder = out_dir + '/2_npz'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(image_reorient_save_dir):
            os.mkdir(image_reorient_save_dir)
        if not os.path.exists(label_reorient_save_dir):
            os.mkdir(label_reorient_save_dir)

        if not os.path.exists(image_resize_save_dir):
            os.mkdir(image_resize_save_dir)
        if not os.path.exists(label_resize_save_dir):
            os.mkdir(label_resize_save_dir)

        if not os.path.exists(npz_save_folder):
            os.mkdir(npz_save_folder)

        image_list = sorted(dir for dir in os.listdir(image_dir))
        label_list = sorted(dir for dir in os.listdir(labels_dir))

        if len(image_list) != len(label_list):
            print('The number of images and labels do not match')
            return

        for i in tqdm.tqdm(range(len(image_list))):
            file_name = image_list[i].split('.')[0]

            image = sitk.ReadImage(image_dir + '/' + image_list[i])
            label = sitk.ReadImage(labels_dir + '/' + label_list[i])

            """reorient"""
            image = SSAD_Dataset.reorient_image(image)
            label = SSAD_Dataset.reorient_image(label)

            sitk.WriteImage(image, image_reorient_save_dir + '/' + file_name + '.nii.gz', useCompression=True)
            sitk.WriteImage(label, label_reorient_save_dir + '/' + file_name + '.nii.gz', useCompression=True)

            """resize"""
            new_sapcing, new_size = SSAD_Dataset.get_tar_spacing(image, tar_size)

            image_resize = SSAD_Dataset.resample_volumeImage(image, new_sapcing, new_size, False)
            label_resize = SSAD_Dataset.resample_volumeImage(label, new_sapcing, new_size, True)

            sitk.WriteImage(image_resize, image_resize_save_dir + '/' + file_name + '.nii.gz', useCompression=True)
            sitk.WriteImage(label_resize, label_resize_save_dir + '/' + file_name + '.nii.gz', useCompression=True)

            """npz"""
            voxel = sitk.GetArrayFromImage(label_resize)
            np.savez_compressed(npz_save_folder + '/' + file_name + '.npz', voxel=voxel)

        """split"""
        split_data(npz_save_folder, split_json)


if __name__ == '__main__':
    """
    You may need to convert the data to NIFTI format
    And then run this code 
    """

    image_dir = '/home/cl/Datasets/CHAOS/imagesTr'
    labels_dir = '/home/cl/Datasets/CHAOS/labelsTr'
    out_dir = '/home/cl/Datasets/CHAOS/SSAD'
    tar_size = 128
    split_json = '../config/data.json'

    SSAD_Dataset.proprocess_data(image_dir, labels_dir, out_dir, tar_size=128, split_json=split_json)









