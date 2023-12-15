import matplotlib.pyplot as plt
import pickle
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import scene_stgcnn
import copy

def to_image_frame(Hinv, loc):
    '''
    :param Hinv: array of shape (3, 3)
    :param loc: array of shape (seq_len, num_peds, 2)
    :return: array of shape (seq_len, num_peds, 2)
    '''
    locHomogenous = np.concatenate((loc, np.ones((loc.shape[0],loc.shape[1], 1))),axis=2)
    locHomogenous_mat = locHomogenous.reshape((-1,3))
    loc_tr = np.transpose(locHomogenous_mat)
    loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
    locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
    locXYZ = locXYZ.reshape((locHomogenous.shape[0],locHomogenous.shape[1],3))
    return locXYZ[:, :, :2].astype(int)

def draw_target_traj(seq_id, obs_traj_in_img, target_traj_in_img, img, dset_name="zara1"):
    '''
    :param seq_id: int
    :param obs_traj_in_img: array of shape (obs_len, num_peds, 2)
    :param target_traj_in_img: array of shape (pred_len, num_peds,2)
    :param img: array of shape (H, W, C)
    :return: None
    '''
    color = ['b', 'r', 'c', 'm', 'y', 'k', 'w', 'g']
    fig = plt.figure(0)
    ax = plt.axes()
    plt.axis(False)
    plt.grid(False)
    plt.imshow(img)

    for k in range(obs_traj_in_img.shape[1]):

        if k < len(color):
            ax.plot(obs_traj_in_img[:, k, 1], obs_traj_in_img[:, k, 0],
                    color[k] + '.--', linewidth=1)
            ax.plot(target_traj_in_img[:, k, 1], target_traj_in_img[:, k, 0],
                    color[k] + '-+', linewidth=1)
            dx = target_traj_in_img[-1, k, 1] - target_traj_in_img[-2, k, 1]
            dy = target_traj_in_img[-1, k, 0] - target_traj_in_img[-2, k, 0]
            init_x = target_traj_in_img[-2, k, 1]
            init_y = target_traj_in_img[-2, k, 0]
            ax.arrow(init_x, init_y, dx, dy,
                     color=color[k], width=6, length_includes_head=True)

    plt.savefig(f"vis_traj_in_img/target/{dset_name}_{seq_id}.png", bbox_inches='tight', pad_inches=-0.1)
    plt.show()

def draw_pred_traj(seq_id, obs_traj_in_img, pred_traj_in_img, img, dset_name="zara1"):
    '''
    :param seq_id: int
    :param obs_traj_in_img: array of shape (obs_len, num_peds, 2)
    :param pred_traj_in_img: array of shape (pred_len, num_peds,2)
    :param img: array of shape (H, W, C)
    :return: None
    '''
    color = ['b', 'r', 'c', 'y', 'm', 'k', 'w', 'g']
    fig = plt.figure(0)
    ax = plt.axes()
    plt.axis(False)
    plt.grid(False)
    plt.imshow(img)

    for k in range(obs_traj_in_img.shape[1]):
        x = pred_traj_in_img[:, k, 1]
        y = pred_traj_in_img[:, k, 0]
        if k < len(color):
            ax.plot(obs_traj_in_img[:, k, 1], obs_traj_in_img[:, k, 0],
                    color[k] + '.--', linewidth=1)
            ax.plot(x, y, color[k] + '-', linewidth=1)
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            init_x = x[-2]
            init_y = y[-2]
            ax.arrow(init_x, init_y, dx, dy,
                     color=color[k], width=6, length_includes_head=True)

    plt.savefig(f"vis_traj_in_img/pred/{dset_name}_{seq_id}.png", bbox_inches='tight', pad_inches=-0.1)
    plt.show()

def draw_traj(dset_name="zara1", seq_id=70, raw_data_dic_=None, img_path="vis_traj/690.png", H_path="datasets/crowds_zara01_H.txt", tag="pred"):

    img = plt.imread(img_path)
    h_mat = np.loadtxt(H_path)
    h_inv = np.linalg.inv(h_mat)

    # obs_traj: array of shape (obs_len, num_peds, 2)
    obs_traj = raw_data_dic_[seq_id]['obs']
    print("the location of pedestrian 0 in first step is", obs_traj[0, 0, :])
    print("the number of peds is", obs_traj.shape[1])

    # target_traj_1: array of shape (pred_len, num_peds, 2)
    target_traj = raw_data_dic_[seq_id]['trgt']
    # pred_traj_1: array of shape (pred_len, num_peds, 2)
    pred_traj = raw_data_dic_[seq_id]['pred']

    obs_traj_in_img = to_image_frame(h_inv, obs_traj)
    target_traj_in_img = to_image_frame(h_inv, target_traj)
    pred_traj_in_img = to_image_frame(h_inv, pred_traj)

    if dset_name in ["univ", "zara1", "zara2"]:
        obs_traj_in_img = np.concatenate((obs_traj_in_img[:, 2:3, ::-1],obs_traj_in_img[:, 4:5, ::-1]), axis=1)
        target_traj_in_img = np.concatenate((target_traj_in_img[:, 2:3, ::-1],target_traj_in_img[:,4:5,::-1]),axis=1)
        pred_traj_in_img = np.concatenate((pred_traj_in_img[:, 2:3, ::-1],pred_traj_in_img[:,4:5,::-1]),axis=1)

    if tag == "pred":
        draw_pred_traj(seq_id, obs_traj_in_img, pred_traj_in_img, img, dset_name)
    elif tag == "target":
        draw_target_traj(seq_id, obs_traj_in_img, target_traj_in_img, img, dset_name)
    else:
        print("this option is not existed!")

def test():
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

        V_pred = mvnormal.sample()

        # V_pred_rel_to_abs: array of shape (pred_len, num_peds, 2)
        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                   V_x[-1, :, :].copy())
        raw_data_dict[step]['pred'] = copy.deepcopy(V_pred_rel_to_abs)


    return raw_data_dict


# Drawing Options
paths = ['./checkpoint/scene-stgcnn-zara1']
dset_name = "zara1"
seq_id = 70
img_path = "vis_traj_in_img/690.png"
H_path = "datasets/crowds_zara01_H.txt"
tag = "target"

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
    raw_data_dic_ = test()

    print("Drawing ....")
    draw_traj(dset_name=dset_name,
              seq_id=seq_id,
              raw_data_dic_=raw_data_dic_,
              img_path=img_path,
              H_path=H_path,
              tag=tag)

