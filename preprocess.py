import numpy as np
import os
import cv2

import torch


def read_image(_path):
    '''
    Args:
        _path: str. etc: "datasets/eth/train\\biwi_hotel_train.png"

    Returns:
        img: array of [H, W]
    '''
    img = cv2.imread(_path, 0)

    return img

def cal_homo_path(path):
    '''
    Args:
        path: str. etc: ./datasets/eth/train\\biwi_hotel_train.txt or ./datasets/eth/test\\biwi_eth.txt

    Returns:
        homo_path: str. etc: ./datasets/hotel_H.txt
    '''
    path = path.replace("\\", "/")
    dataset = os.path.splitext(path)[0].split('/')[-1]
    dataset_name = dataset.replace('_train','')
    homo_path = './datasets' + f'/{dataset_name}_H.txt'

    return homo_path, dataset_name

def to_image_frame(Hinv, loc):
    '''
    :param Hinv: array of shape (3,3)
    :param loc: tensor of shape (num_peds, 2, seq_len)
    :return: array of shape (num_peds, seq_len, 2)
    '''
    loc = loc.permute((0,2,1)).cpu().numpy()
    locHomogenous = np.concatenate((loc, np.ones((loc.shape[0],loc.shape[1], 1))),axis=2)
    locHomogenous_mat = locHomogenous.reshape((-1,3))
    loc_tr = np.transpose(locHomogenous_mat)
    loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
    locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
    locXYZ = locXYZ.reshape((locHomogenous.shape[0],locHomogenous.shape[1],3))
    return locXYZ[:, :, :2].astype(int)



def scene_attn(traj, seg_map, homo_mat, kernel_radius=128, classes=2, scene_name="biwi_eth"):
    '''
    Args:
        traj: tensor of shape (num_peds, 2, seq_len)
        seg_map: tensor of shape (H, W)
        homo_mat: tensor of shape (3, 3)
    Returns:
        scene_v: tensor of shape (num_peds, seq_len, classes)
    '''
    seg_map_ = seg_map.data.cpu().numpy()
    h_mat = homo_mat.data.cpu().numpy()
    Hinv = np.linalg.inv(h_mat)
    patch_size = 2 * kernel_radius

    H, W = seg_map_.shape
    reflect_map = np.zeros((H+patch_size, W+patch_size))
    reflect_map[kernel_radius:H+kernel_radius, kernel_radius:W+kernel_radius] = seg_map_

    num_peds, _, seq_len = traj.shape
    scene_patches = np.zeros((num_peds, seq_len, patch_size, patch_size))
    scene_v = np.zeros((num_peds, seq_len, classes))
    # traj_in_img: array of shape (num_peds, seq_len, 2)
    traj_in_img = to_image_frame(Hinv, traj.data)
    for i in range(num_peds):
        for j in range(seq_len):
            x_in_img = traj_in_img[i, j, 0] + kernel_radius
            y_in_img = traj_in_img[i, j, 1] + kernel_radius
            if "eth" in scene_name or "hotel" in scene_name:
                scene_patches[i,j] = reflect_map[(x_in_img-kernel_radius):(x_in_img+kernel_radius), (y_in_img-kernel_radius):(y_in_img+kernel_radius)]
            else:
                scene_patches[i, j] = reflect_map[(y_in_img - kernel_radius):(y_in_img + kernel_radius),
                                      (x_in_img - kernel_radius):(x_in_img + kernel_radius)]
            if j >= 1:
                scene_img = [(scene_patches[i, j] == v) for v in range(classes)]
                scene_img_last = [(scene_patches[i, j-1] == v) for v in range(classes)]
                scene_v[i, j] = np.array([(scene_img[v] ^ scene_img_last[v]).sum() / (patch_size * patch_size) for v in range(classes)])

    return torch.from_numpy(scene_v).type(torch.float)



