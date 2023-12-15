import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import scene_stgcnn
import copy

def visualise(dset_name, seq_id, raw_data_dic_=None):
    '''
    :param dset_name: str
    :param seq_id: int
    :param raw_data_dic_: dic
    :param tag: str
    :return: None
    '''
    # obs_traj: array of shape (obs_len, num_peds, 2)
    obs_traj = raw_data_dic_[seq_id]['obs']
    # target_traj: array of shape (pred_len, num_peds, 2)
    target_traj = raw_data_dic_[seq_id]['trgt']
    # pred_traj: list consisted of array of shape (pred_len, num_peds, 2)
    pred_traj = raw_data_dic_[seq_id]['pred']
    # pred_traj: array of shape (pred_len * KSTEPS, num_peds, 2)
    pred_traj = np.concatenate(pred_traj, axis=0)

    if dset_name in ['univ', 'zara1', 'zara2']:
        obs_traj = obs_traj[:, :, ::-1]
        target_traj = target_traj[:, :, ::-1]
        pred_traj = pred_traj[:, :, ::-1]

    # visualization of target traj
    fig = plt.figure(0)
    ax = plt.axes()
    plt.axis(False)
    plt.grid(False)
    color = ['g', 'b', 'r', 'm', 'c']

    for k in range(obs_traj.shape[1]):

        if k < len(color) and k in [3, 4]:
            ax.plot(obs_traj[:, k, 1], obs_traj[:, k, 0],
                    color[k] + '.--', linewidth=2)
            ax.plot(target_traj[:, k, 1], target_traj[:, k, 0],
                    color[k] + '-', linewidth=2)
            x0 = target_traj[-2, k, 1]
            y0 = target_traj[-2, k, 0]
            x1 = target_traj[-1, k, 1]
            y1 = target_traj[-1, k, 0]
            plt.annotate('', xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(color=color[k],
                                         width=1,
                                         headwidth=10,
                                         headlength=10))
            sns.kdeplot(pred_traj[:, k, 1],
                        pred_traj[:, k, 0],
                        shade=True,
                        thresh=0.3,
                        color=color[k],
                        alpha=0.8,
                        levels=8,
                        )
    print("x-axis range: ", plt.xlim())
    print("y-axis range: ", plt.ylim())
    plt.show()

def test(KSTEPS):
    global loader_test, model
    model.eval()
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1

        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, V_obs, A_obs, scene_v_obs, V_tr, A_tr, scene_v_pred = batch

        # V_obs_tmp: batch, 2, obs_len, num_peds
        scene_v_obs_tmp = scene_v_obs.permute(0, 3, 2, 1)

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), scene_v_obs_tmp)

        # V_pred: tensor of shape (batch, pred_len, num_peds, 5)
        V_pred = V_pred.permute(0, 2, 3, 1)

        # V_tr: tensor of shape (pred_len, num_peds, 2)
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()

        # V_pred: tensor of shape (pred_len, num_peds, 5)
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        #  V_x : array of shape (obs_len, num_peds, 2)
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        # V_x_rel_to_abs: array of shape (obs_len, num_peds, 2)
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        # V_y_rel_to_abs: array of shape (pred_len, num_peds, 2)
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for k in range(KSTEPS):
            V_pred = mvnormal.sample()
            # V_pred_rel_to_abs: array of shape (pred_len, num_peds, 2)
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

        raw_data_dict[step]['mean'] = copy.deepcopy(mean.data.cpu().numpy())
        raw_data_dict[step]['cov'] = copy.deepcopy(cov.data.cpu().numpy())

    return raw_data_dict


# Drawing Options
paths = ['./checkpoint/scene-stgcnn-zara2']
dset_name = "zara2"
seq_id = 25
KSTEPS = 20

print("*" * 50)

for feta in range(len(paths)):
    path = paths[feta]

    print("*" * 50)
    print("Evaluating model:", path)

    model_path = path + '/val_best.pth'
    args_path = path + '/args.pkl'

    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    stats = path + '/constant_metrics.pkl'
    with open(stats, 'rb') as f:
        cm = pickle.load(f)
    print("Stats:", cm)

    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/' + args.dataset + '/'

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        num_classes=args.num_classes,
        ker_radius=args.ker_radius,
        skip=1, norm_lap_matr=True)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=0)

    model = scene_stgcnn(n_tcnns=args.n_tcnns, input_feat=args.input_size, output_feat=args.output_size,
                         classes=args.num_classes,
                         seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len).cuda()
    model.load_state_dict(torch.load(model_path))

    ade_ = 999999
    fde_ = 999999
    raw_data_dic_ = test(KSTEPS)
    print("visualing ....")
    visualise(dset_name=dset_name,
              seq_id=seq_id,
              raw_data_dic_=raw_data_dic_)

