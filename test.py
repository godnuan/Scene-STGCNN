import pickle
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import scene_stgcnn
import copy

def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, V_obs, A_obs, scene_v_obs, V_tr, A_tr, scene_v_pred = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        # V_obs = batch,seq,node,feat

        # V_obs_tmp: batch, 2, obs_len, num_peds
        scene_v_obs_tmp = scene_v_obs.permute(0, 3, 2, 1)

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), scene_v_obs_tmp)

        # V_pred: tensor of shape (batch, pred_len, num_peds, 5)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
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

        ade_ls = {}
        fde_ls = {}
        #  V_x : array of shape (obs_len, num_peds, 2)
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        # V_x_rel_to_abs: array of shape (obs_len, num_peds, 2)
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


paths = ['./checkpoint/scene-stgcnn-eth', './checkpoint/scene-stgcnn-hotel',
         './checkpoint/scene-stgcnn-univ', './checkpoint/scene-stgcnn-zara1',
         './checkpoint/scene-stgcnn-zara2']
KSTEPS = 20

print("*" * 50)
print('Number of samples:', KSTEPS)
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

    # Data prep
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

    # Defining the model
    model = scene_stgcnn(n_tcnns=args.n_tcnns, input_feat=args.input_size, output_feat=args.output_size,
                         classes=args.num_classes,
                         seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len).cuda()
    model.load_state_dict(torch.load(model_path))

    ade_ = 999999
    fde_ = 999999
    print("Testing ....")
    ad, fd, raw_data_dic_ = test()
    ade_ = min(ade_, ad)
    fde_ = min(fde_, fd)

    print("ADE: {:.2f}, FDE: {:.2f}".format(ade_, fde_))


