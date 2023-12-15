import copy
import random

import torch.cuda

from utils import *
from metrics import *
import pickle
import argparse
from model import *
import torch.distributions.multivariate_normal as torchdist

parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--n_tcnns', type=int, default=1)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--ker_radius', type=int, default=64)

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')

# Training specifc parameters
parser.add_argument('--seed', type=int, default=520)
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--tag', default='scene-stgcnn-eth',
                    help='personal tag for the model ')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

print('* ' *30)
print("Training initiating....")
print(args)


def graph_loss(V_pred ,V_target):
    return bivariate_loss(V_pred ,V_target)


# Data prep
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len

dataset = './datasets/' + args.dataset +'/'

dset_train = TrajectoryDataset(
    dataset +'train/',
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    num_classes=args.num_classes,
    ker_radius=args.ker_radius,
    skip=1 ,norm_lap_matr=True)

loader_train = DataLoader(
    dset_train,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle =True,
    num_workers=0)


dset_val = TrajectoryDataset(
            dataset+'test/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            num_classes=args.num_classes,
            ker_radius=args.ker_radius,
            skip=1,norm_lap_matr=True)

loader_val = DataLoader(
        dset_val,
        batch_size=1,#This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=0)

# Defining the model

model = scene_stgcnn(n_tcnns=args.n_tcnns,input_feat=args.input_size,output_feat=args.output_size ,
                      classes=args.num_classes,
                      seq_len=args.obs_seq_len,pred_seq_len=args.pred_seq_len).cuda()

# Training settings

optimizer = optim.Adam(model.parameters(),lr=args.lr)

checkpoint_dir = './checkpoint/' +args.tag +'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir +'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)


# Training
metrics = {'train_loss' :[], 'val_loss' :{'ade': [], 'fde': []}}
constant_metrics = {'min_val_epoch' :-1, 'min_val_loss' :{'ade': 9999, 'fde': 9999}}

def train(epoch):
    global metrics ,loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, V_obs, A_obs, scene_v_obs, V_tr, A_tr, scene_v_pred = batch


        optimizer.zero_grad()
        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        # scene_v_obs_tmp = batch, classes, seq, node

        scene_v_obs_tmp = scene_v_obs.permute(0, 3, 2, 1)

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), scene_v_obs_tmp)

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward(retain_graph=True)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)


def val(epoch,KSTEPS=20):
    global metrics, loader_val, constant_metrics
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_val:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,V_obs,A_obs,A_scene_obs,V_tr,A_tr,A_scene_tr = batch
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

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
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

    metrics['val_loss']['ade'].append(ade_)
    metrics['val_loss']['fde'].append(fde_)
    if metrics['val_loss']['ade'][-1] < constant_metrics['min_val_loss']['ade']:
        constant_metrics['min_val_loss']['ade'] = metrics['val_loss']['ade'][-1]
        constant_metrics['min_val_loss']['fde'] = metrics['val_loss']['fde'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    val(epoch, KSTEPS=20)

    print('*' * 30)
    print('Epoch:', args.tag, ":", epoch)
    print("train_loss: ",metrics['train_loss'][-1])
    print("val_loss: ade: {:.2f}, fde: {:.2f}".format(metrics['val_loss']['ade'][-1], metrics['val_loss']['fde'][-1]))
    print(constant_metrics)
    print('*' * 30)

    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)




