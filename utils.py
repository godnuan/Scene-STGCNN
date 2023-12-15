import os
import math

import torch
import numpy as np

from torch.utils.data import Dataset

import networkx as nx
from tqdm import tqdm
from preprocess import read_image, scene_attn, cal_homo_path
import warnings

warnings.filterwarnings('ignore')


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, num_classes=2, ker_radius=128, skip=1,
            min_ped=1, delim='\t', norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.ker_radius = ker_radius
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files_ = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files_ if _path.endswith('txt')]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        fet_map = {}
        fet_list = []
        homo_list = []
        scene_name_list = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()

            img_path = os.path.splitext(path)[0] + ".png"  # img_path: "datasets/eth/train\\biwi_hotel_train.png"
            homo_path, scene_name = cal_homo_path(path)

            # new_fet: array of shape (H, W) except eth or hotel
            new_fet = read_image(img_path)
            fet_map[img_path] = torch.from_numpy(new_fet)
            # homo_mat: tensor of shape (3, 3)
            homo_mat = torch.from_numpy(np.loadtxt(homo_path))

            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                num_peds_considered = 0

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    fet_list.append(img_path)
                    homo_list.append(homo_mat)
                    scene_name_list.append(scene_name)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        self.scene_name_list = scene_name_list
        self.fet_map = fet_map
        self.fet_list = fet_list
        self.homo_list = homo_list

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.scene_v_obs = []
        self.v_pred = []
        self.A_pred = []
        self.scene_v_pred = []
        print("Processing Data .....")
        for ss in tqdm(range(len(self.seq_start_end)), ncols=100):
            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            scene_v = scene_attn(self.obs_traj[start:end, :], self.fet_map[self.fet_list[ss]], self.homo_list[ss],
                                 self.ker_radius, self.num_classes, self.scene_name_list[ss])
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            self.scene_v_obs.append(scene_v.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            scene_v = scene_attn(self.pred_traj[start:end, :], self.fet_map[self.fet_list[ss]], self.homo_list[ss],
                                 self.ker_radius, self.num_classes, self.scene_name_list[ss])
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            self.scene_v_pred.append(scene_v.clone())

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        '''
        Args:
            index: int
        Returns:
            out: list
            self.obs_traj: tensor of shape (num_peds, 2, obs_len)
            self.v_obs: tensor of shape (obs_len, num_peds, 2)
            self.A_obs: tensor of shape (obs_len, num_peds, num_peds)
            self.A_scene_obs: tensor of shape (obs_len, num_peds, num_peds)
            self.scene_v_obs: tensor of shape (num_peds, obs_len, classes)
        '''
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.v_obs[index], self.A_obs[index], self.scene_v_obs[index],
            self.v_pred[index], self.A_pred[index], self.scene_v_pred[index],
        ]
        return out

# if __name__ == '__main__':
#     dataset = 'zara1'
#     obs_seq_len = 8
#     pred_seq_len = 12
#     data_set = './datasets/' + dataset + '/'
#
#     dset_train = TrajectoryDataset(
#         data_set + 'test/',
#         obs_len=obs_seq_len,
#         pred_len=pred_seq_len,
#         skip=1, norm_lap_matr=True)

