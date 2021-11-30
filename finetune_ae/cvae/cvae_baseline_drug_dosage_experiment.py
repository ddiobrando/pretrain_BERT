"""
Code adapted from https://github.com/cagatayyildiz/ODE2VAE/blob/94860eb46ab13da45faff406b69285c3355e0253/torch_ode2vae_minimal.py.
"""

import torch
import argparse
import json
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.switch_backend('agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from multiprocessing import Process, freeze_support

torch.multiprocessing.set_start_method('spawn', force="True")

sys.path.append(os.path.join(sys.path[0], '../../'))

import drug_dosage
import conditional_regressor
import initialize_experiment

parser = argparse.ArgumentParser('drug dosage experiment')
parser.add_argument('--device', type=str, default='cuda:0', help="Device to use")
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--viz-freq', type=int, default=500, help='Number of epochs to plot after')
parser.add_argument('--val-freq', type=int, default=10, help='Number of epochs to validate after')
parser.add_argument('--unseen-combos', nargs='*', default=None,
                    help='List of drug_id, cell_id combinations to withhold during training. Must provide groups of 2')
parser.add_argument('--num-unseen-combos', type=int, default=None,
                    help='Number of (drug, cell) combinations to randomly withhold during training')
parser.add_argument('--use-unseen-dosages', action='store_true', default=False,
                    help='Set if we should withhold some dosages from training data.')
parser.add_argument('--latent-dim', type=int, default=1, help='Dimension of latent space.')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size to use.')
parser.add_argument('--hidden-layers', type=int, nargs='*', default=None,
                    help='Size of hidden layers to use for the encoder and (in reverse) for the decoder')
parser.add_argument('--num-genes', type=int, default=978, help='Num of genes')
parser.add_argument('--data-loc', type=str, default='', help='Modify DRUG_EXPRESSION_DATA_LOC in drug_dosage')
parser.add_argument('--result-loc', type=str, default='', help='Modify MULTICLASS_CVAE_DRUG_DOSAGE_DIR')


args = parser.parse_args()
print(args)

drug_dosage.DRUG_EXPRESSION_DATA_LOC=args.data_loc
MULTICLASS_CVAE_DRUG_DOSAGE_DIR=args.result_loc

device = torch.device(args.device)
assert args.unseen_combos is None or len(args.unseen_combos) % 2 == 0, \
    'Unseen combos must be in groups of 2!'
unseen_dc = None if args.unseen_combos is None else {
    (args.unseen_combos[i], args.unseen_combos[i + 1]) for i in range(0, len(args.unseen_combos), 2)}
# unseen_times = None if args.unseen_times is None else {int(t) for t in args.unseen_times}
hidden_dims = [] if args.hidden_layers is None else [h for h in args.hidden_layers]

freeze_support()
initialize_experiment.initialize_random_seed(args.seed)


dl = drug_dosage.DrugExpressionDosageWithUnseenDataLoaderForCVAE(
    device=torch.device(args.device), unseen=unseen_dc, num_unseen_combos=args.num_unseen_combos,
    use_unseen_dosages=args.use_unseen_dosages)
dl.set_batch_size(args.batch_size)


drug_listing = dl.get_drug_list()
cell_listing = dl.get_cell_list()
unseen_listing = dl.get_unseen_combos()
unseen_dosages = dl.get_unseen_dosages()

trainset = dl.get_data_loader('train')
valset = dl.get_data_loader('val')

model_dir = MULTICLASS_CVAE_DRUG_DOSAGE_DIR + \
            'smaller_d_{}_hidden_dims_{}_unseen_dc_{}_unseen_ts_{}/model_{}/'.format(
                args.latent_dim, hidden_dims, unseen_listing, unseen_dosages, args.seed)
print('model_dir',model_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# write the args to a file
with open(model_dir + 'arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

num_unique_drugs = len(drug_listing)
num_unique_cells = len(cell_listing)
num_genes = args.num_genes


net = conditional_regressor.cVAE_Sequence(
    input_size=num_genes, latent_dim=args.latent_dim, num_classes=3,
    class_sizes=[num_unique_drugs, num_unique_cells, 1],
    hidden_dims=hidden_dims, device=device
).to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
val_mse = []
train_mse = []
train_loss = []
best_model_loss = None
best_model_epoch = None

for ep in tqdm(range(args.epochs)):
    for i, batch_info in enumerate(trainset):
        drug_id, cell_id, dosages, expr, mask = batch_info
        # drug_id: [N, T, 162]
        # cell_id: [N, T, 9]
        # ts: [N, T, 1]
        # expr: [N, T, 978]
        # mask: [N, T]

        # Apply the masking here to get rid of irrelevant time points.
        drug_id = torch.masked_select(drug_id, torch.stack(
            [mask for _ in range(num_unique_drugs)], dim=-1).bool()).view([-1, num_unique_drugs]).to(
            device)  # drug_id.view([N*T, 315]).to(device)
        cell_id = torch.masked_select(cell_id, torch.stack(
            [mask for _ in range(num_unique_cells)], dim=-1).bool()).view([-1, num_unique_cells]).to(
            device)  # cell_id.view([N*T, 1004]).to(device)
        dosages = torch.masked_select(dosages, mask.bool()).view([-1, 1]).to(device)  # conc.view([N*T, 1]).to(device)
        expr = torch.masked_select(expr, torch.stack(
            [mask for _ in range(num_genes)], dim=-1).bool()).view([-1, num_genes]).to(device)  # y.view([N*T, 1]).to(device)

        outputs = net(expr, [drug_id.float(), cell_id.float(), dosages.float()])
        tr_loss = net.loss_fn(*outputs, beta=1.0)

        optimizer.zero_grad()
        tr_loss['loss'].backward()
        optimizer.step()

        print('Ep: {:3d} Iter:{:<2d} mse: {:<5.3f}'.format(
            ep, i, tr_loss['loss'].item()))

    if ep % args.val_freq == args.val_freq - 1:
        if ep % args.viz_freq == args.viz_freq - 1:
            ep_dir = model_dir + 'ep={}/'.format(ep)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            torch.save(net.state_dict(), ep_dir + 'model.pth')
        with torch.set_grad_enabled(False):
            for val_info in valset:
                val_drug, val_cell, val_dosages, val_expr, val_mask = val_info  # val mask shape is [N, T]
                N, T, _ = val_drug.shape
                val_drug = torch.masked_select(val_drug, torch.stack(
                    [val_mask for _ in range(num_unique_drugs)], dim=-1).bool()).view([-1, num_unique_drugs]).to(device)  # val_drug.to(device)
                val_cell = torch.masked_select(val_cell, torch.stack(
                    [val_mask for _ in range(num_unique_cells)], dim=-1).bool()).view([-1, num_unique_cells]).to(device)  # val_cell.to(device)
                val_dosages = torch.masked_select(val_dosages, val_mask.bool()).view([-1, 1]).to(
                    device)
                val_expr = torch.masked_select(val_expr, torch.stack(
                    [val_mask for _ in range(num_genes)], dim=-1).bool()).view([-1, num_genes]).to(
                    device)  # val_y.to(device)

                outputs = net(val_expr, [val_drug.float(), val_cell.float(), val_dosages.float()])
                val_loss = net.loss_fn(*outputs, beta=1.0)

                train_loss.append(tr_loss['loss'].item())
                train_mse.append(tr_loss['MSE'].item())
                val_mse.append(val_loss['MSE'].item())
                break

        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f} val_mse:{:5.3f}'.format(
            ep, args.epochs, tr_loss['loss'].item(), val_loss['MSE'].item()
        ))

        if best_model_loss is None or val_mse[-1] < best_model_loss:
            print('\tupdated to new best model at epoch', ep)
            best_model_loss = val_mse[-1]
            best_model_epoch = ep
            torch.save(net.state_dict(), model_dir + 'tmp_best_model.pth')

    if initialize_experiment.check_for_early_break_condition(val_mse):
        break

# save the best model
print('...best model at epoch {}'.format(best_model_epoch))
net.load_state_dict(torch.load(model_dir + 'tmp_best_model.pth'))
torch.save(net.state_dict(), model_dir + 'best_model.pth')

plt.style.use('seaborn')
plt.plot(train_loss, label='train loss')
plt.plot(train_mse, label='train mse')
plt.plot(val_mse, label='val mse')
plt.legend()
plt.xlabel('Epoch / {}'.format(args.viz_freq))
plt.ylabel('Loss')
plt.savefig(model_dir + 'losses.png', bbox_inches='tight')
