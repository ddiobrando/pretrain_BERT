import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl
from torchdiffeq import odeint
from typing import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append(os.path.join(sys.path[0], '../../'))
from models.unseen_attribute_generation.ode2vae.torch_bnn import BNN
from models import conditional_regressor
# from models.unseen_attribute_generation.ode2vae.torch_ode2cvae_minimal


class ODE2cVAERegression(nn.Module):
    def __init__(self, cReg, q_ode: int=8, device='cuda:0',
                 conversion_dims: List[int]=None, use_full_second_order: bool=False,
                 reg_freq: int = 1, bnn_num_hidden_layers: int=3, bnn_hidden: int=50):
        """
        :param cReg: the conditional variational regression model to use as encoder/regressor object.
        :param q_ode: latent dimension to use for the ODE operations.
        :param device: device to use.
        """
        super().__init__()

        self.cReg = cReg
        self.q_style = cReg.latent_dim
        self.q_ode = q_ode
        self.num_classes = cReg.num_classes
        self.class_sizes = cReg.class_sizes
        self.device = device
        self.reg_freq = reg_freq

        # move from cReg's latent dim to latent dim for ODE -> should have shape 2 * q_ode
        if conversion_dims == None:
            conversion_dims = []
        prev = self.q_style + np.sum(self.class_sizes)
        conversion_layers = []
        for d in conversion_dims:
            conversion_layers.append(nn.Sequential(nn.Linear(prev, d), nn.ReLU()))
            prev = d
        conversion_layers.append(nn.Linear(prev, 2 * q_ode))
        self.embed_to_ode = nn.Sequential(*conversion_layers)

        if isinstance(cReg, conditional_regressor.cVariationalRegressor):
            # update regressor's input to take ode's latent rep
            prev_d = q_ode + np.sum(self.class_sizes)
            dec_layers = []
            for i in range(len(cReg.hidden_dims_dec)):
                dec_layers.append(nn.Sequential(nn.Linear(prev_d, cReg.hidden_dims_dec[i]), nn.ReLU()))
                prev_d = cReg.hidden_dims_dec[i]
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, 1), nn.ReLU()))
            cReg.regressor = nn.Sequential(*dec_layers)
        else:
            assert isinstance(cReg, conditional_regressor.cVAE_Sequence)
            # update the decoder's input to take ode's latent rep
            prev_d = q_ode + np.sum(self.class_sizes)
            dec_layers = []
            for i in range(len(cReg.hidden_dims)-1, -1, -1):
                dec_layers.append(nn.Sequential(nn.Linear(prev_d, cReg.hidden_dims[i]), nn.ReLU()))
                prev_d = cReg.hidden_dims[i]
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, cReg.input_size)))
            cReg.decoder = nn.Sequential(*dec_layers)

        # differential function
        # to use a deterministic differential function, set bnn=False and self.beta=0.0
        self.bnn = BNN(2 * q_ode, q_ode, n_hid_layers=bnn_num_hidden_layers, n_hidden=bnn_hidden,
                       act='celu', layer_norm=True, bnn=True)
        # downweighting the BNN KL term is helpful if self.bnn is heavily overparameterized
        self.beta = 1.0  # 2 * q_ode / self.bnn.kl().numel()
        self.gamma = 1.0

        self.mvn_style = MultivariateNormal(torch.zeros(self.q_style).to(device),
                                            torch.eye(self.q_style).to(device))
        self.mvn_ode = MultivariateNormal(torch.zeros(2 * self.q_ode).to(device),
                                          torch.eye(2 * self.q_ode).to(device))
        self.use_full_second_order = use_full_second_order

    def ode2vae_rhs(self, t, vs_logp, f, t_step=1):
        """
        :param t: Time points, N
        :param vs_logp: (samples OI, probability of those samples), N, 2q & N
        :param f: integration method
        """
        vs, logp = vs_logp  # N, 2q & N
        N = vs.shape[0]
        q = vs.shape[1] // 2
        dv = f(vs)  # N, q
        ds = vs[:, :q]  # N, q
        dvs = torch.cat([dv, ds], 1)  # N, 2q
        # Removed second-order piece
        if self.use_full_second_order:
            if min((t / t_step).item() % self.reg_freq,
                   self.reg_freq - (t / t_step).item() % self.reg_freq) > 1e-3:
                tr_ddvi_dvi = torch.zeros(N).to(vs.device)
            else:
                ddvi_dvi = torch.stack(
                    [torch.autograd.grad(dv[:, i], vs, torch.ones_like(dv[:, i]),
                                         retain_graph=True, create_graph=True)[0].contiguous()[:, i]
                     for i in range(q)], 1)  # N, q --> df(x)_i, dx_i, i=1,...,q
                tr_ddvi_dvi = torch.sum(ddvi_dvi, 1)  # N
        else:
            tr_ddvi_dvi = -logp
        return (dvs, -tr_ddvi_dvi)

    def elbo(self, qz_m, qz_logv, zode_L, logpL, X, XrecL, Ndata, mask, qz_enc_m=None,
             qz_enc_logv=None):
        """
        :param qz_m: latent means, [N, q_style]
        :param qz_logv: latent logvars [N, q_style]
        :param zode_L: latent trajectory samples [L, N, T, 2*q_ode]
        :param logpL: densities of latent trajectory samples [L, N, T]
        :param X: true outputs [N, T, 1]
        :param XrecL: predicted outputs [L, N, T, 1]
        :param Ndata: number of sequences in the dataset (required for elbo)
        :param mask: masks for data [N, T, 1]
        :param qz_enc_m: encoder density means, [NT, q_style]
        :param qz_enc_logv: encoder density log variances [NT, q_style]
        :return: likelihood
            prior on ODE trajectories, KL[q_ode(z_{0:T} || N(0, I)]
            prior on BNN weights
            instant encoding term KL[q_ode(z_{0:T} || q_enc(z{0:T}|X_{0:T})]
        """
        [N, T, d] = X.shape
        L = zode_L.shape[0]

        # prior
        log_pzt = self.mvn_ode.log_prob(zode_L.contiguous().view([L * N * T, 2 * self.q_ode])) # LNT
        log_pzt = log_pzt.view([L, N, T])  # L, N, T
        kl_zt = logpL - log_pzt  # L, N, T
        # Now, apply the mask to only penalize relevant datapoints.
        # Go through each trajectory, select the appropriate time points and sum them for each N.
        res = []
        for l in range(L):
            partL = kl_zt[l]  # N, T
            res.append(torch.stack([
                torch.masked_select(partL[i], mask.bool()).sum() for i in range(N)], dim=0).view(N)) # N
        kl_z = torch.stack(res, dim=0).mean(0)  # N
        kl_w = self.bnn.kl().sum()

        # likelihood
        XL = X.repeat([L, 1, 1, 1])  # L, N, T, d
        # Need to change likelihood function because these are no longer binary values for predictions!
        lhood_L = (XL - XrecL)**2  # L, N, T, output_dim
        assert L == 1, 'Not implemented yet!'
        lhood = torch.stack([
            torch.masked_select(lhood_L[0][ex], torch.stack(
                [mask[ex] for _ in range(XrecL.shape[-1])], dim=-1).bool()).sum() for ex in range(N)], dim=0)  # N

        if qz_enc_m is not None:  # instant encoding
            qz_enc_mL = qz_enc_m.repeat([L, 1])  # LNT, 2q
            qz_enc_logvL = qz_enc_logv.repeat([L, 1])  # LNT, 2q
            mean_ = qz_enc_mL.contiguous().view(-1)  # 2LNTq
            std_ = 1e-3 + qz_enc_logvL.exp().contiguous().view(-1)  # 2LNTq
            qenc_zt_ode = Normal(mean_, std_).log_prob(zode_L.contiguous().view(-1)).view([L, N, T, self.q_style])
            qenc_zt_ode = qenc_zt_ode.sum([3])  # L, N, T
            inst_enc_KL = logpL - qenc_zt_ode
            inst_enc_KL = inst_enc_KL.sum(2).mean(0)  # N
            return Ndata * lhood.mean(), Ndata * kl_z.mean(), kl_w, Ndata * inst_enc_KL.mean()
        else:
            return Ndata * lhood.mean(), Ndata * kl_z.mean(), kl_w

    def forward(self, X, cats, ts, mask, Ndata, L=1, inst_enc=False, method='dopri5', dt=0.1,
                t_step=1):
        """
        :param X: true outputs, [N, T, 1]
        :param cats: list of one-hot class label encodings, each with shape [N, T, class_size]
        :param ts: time points for each X, [N, T]
        :param mask: mask for the data with appropriate time points
        :param Ndata: number of sequences in the dataset (required for elbo)
        :param L: number of Monte Carlo draws (from BNN)
        :param inst_enc: whether instant encoding is used or not
        :param method: numerical integration method
        :param dt: numerical integration step size
        :return:
            Xrec_mu: reconstructions from the mean embedding, [N, T]
            X_rec_L: reconstructions from the latent samples: [L, N, T]
            qz_m: mean of the latent embeddings: [N, T, q_ode]
            qz_logv: log variance of the latent embeddings: [N, T, q_ode]
            lhood-kl_z: ELBO
            lhood: prediction mse
            kl_z: KL
        """
        # encode
        [N, T, d] = X.shape
        x0 = torch.cat([X[:, 0], *[c[:, 0] for c in cats]], dim=-1)
        qz0_m, qz0_logv = self.cReg.encode(x0)  # shapes: [N, q_style]
        
        z0, eps = self.cReg.reparameterize(qz0_m, qz0_logv, return_eps=True)
        logp0 = self.mvn_style.log_prob(eps)

        # Now, get the ODE samples
        z0 = self.embed_to_ode(torch.cat([z0, *[c[:, 0] for c in cats]], dim=-1))  # output; [N, 2 * q_ode]
        # ODE
        ztL = []
        logpL = []

        max_T = torch.max(ts).item() + t_step
        tdense = dt * torch.arange(start=0, end=max_T, step=t_step, dtype=torch.float
                                  ).to(z0.device)
        # # mask for where the actual points are that we want after integration.
        tdense_mask = torch.zeros(tdense.shape).to(z0.device)
        tdense_mask[(ts[0] / t_step).long()] = 1
        stacked_mask_TxN = torch.stack([tdense_mask for _ in range(N)], dim=-1)
        stacked_mask_TxNx2q = torch.stack([stacked_mask_TxN for _ in range(2 * self.q_ode)], dim=-1)

        # sample L trajectories
        for l in range(L):
            f = self.bnn.draw_f()  # draw a differential function
            oderhs = lambda t, vs: self.ode2vae_rhs(t, vs, f, t_step)  # t, vs, f)
            zt, logp = odeint(oderhs, (z0, logp0), tdense, method=method)  # t, method=method)  # T, N, 2q & T, N
            # zt = torch.masked_select(zt, stacked_mask_TxNx2q.bool()).view([T, N, 2*self.q_ode])
            # logp = torch.masked_select(logp, stacked_mask_TxN.bool()).view([T, N])

            ztL.append(zt.permute([1, 0, 2]).unsqueeze(0))  # 1, N, T, 2q
            logpL.append(logp.permute([1, 0]).unsqueeze(0))  # 1, N, T

        ztL = torch.cat(ztL, 0)  # L, N, T, 2q
        logpL = torch.cat(logpL)  # L, N, T

        # restart from a random non-zero index
        uIdx = np.random.choice(range(1, len(ts[0])), size=1)[0]
        uIdx_dense = int(ts[0][uIdx] / t_step)
        z0_useq = ztL[0, :, uIdx_dense, :]  # N, 2 * q_ode

        def ode2vae_mean_rhs(t, vs, f):
            q = vs.shape[1] // 2
            dv = f(vs)
            ds = vs[:, :q]
            return torch.cat([dv, ds], 1)

        f = self.bnn.draw_f()
        odef = lambda t, vs: ode2vae_mean_rhs(t, vs, f)
        zt_useq = odeint(odef, z0_useq, tdense[uIdx_dense:] - tdense[uIdx_dense], method=method)
        zt_useq = zt_useq.permute([1, 0, 2]).unsqueeze(0)  # 1, N, T', 2q
        st_useq = zt_useq[:, :, :, self.q_ode:]  # 1, N, T', q
        num_ts = len(tdense) - uIdx_dense
        if isinstance(self.cReg, conditional_regressor.cVariationalRegressor):
            Xrec_useq = self.cReg.decode(
                torch.cat([st_useq.contiguous().view([N * num_ts, self.q_ode]),
                           *[torch.stack([c[:, 0] for _ in range(num_ts)], dim=1
                                         ).contiguous().view([N * num_ts, -1]) for c in cats]], dim=-1)
            )
        else:
            assert isinstance(self.cReg, conditional_regressor.cVAE_Sequence)
            Xrec_useq = self.cReg.decode(
                torch.cat([st_useq.contiguous().view([N * num_ts, self.q_ode]),
                           *[torch.stack([c[:, 0] for _ in range(num_ts)], dim=1
                                         ).contiguous().view([N * num_ts, -1]) for c in cats]], dim=-1)
            )
        Xrec_useq = Xrec_useq.view([N, num_ts, -1])
        # mask is currently N x T_full. Need to get to N x T_obs x num_genes.
        stacked_mask_useq = torch.stack([stacked_mask_TxN[uIdx_dense:].permute(1, 0) for _ in range(d)], dim=-1)
        stacked_mask_orig = torch.stack([mask[:, uIdx:] for _ in range(d)], dim=-1)
        
        lhoods = []
        for ex in range(N):
            ex_useq = torch.masked_select(Xrec_useq[ex], stacked_mask_useq[ex].bool()).view([-1, d]) # first, apply actual-time mask
            ex_useq = torch.masked_select(ex_useq, stacked_mask_orig[ex].bool()).view([-1, d])  # second, apply observed mask
            ex_orig = torch.masked_select(X[ex, uIdx:], stacked_mask_orig[ex].bool()).view([-1, d])
            lhoods.append(((ex_useq - ex_orig)**2).sum())
        lhood_useq = torch.stack(lhoods, dim=0)  # N
        lhood_useq = T * lhood_useq / (T - uIdx)


        # decode
        assert L == 1, 'Currently only works with L=1!'
        ztL = torch.masked_select(
            ztL, torch.stack([stacked_mask_TxNx2q.transpose(0, 1) for _ in range(L)], dim=0
                            ).bool()).view([L, N, T, 2 * self.q_ode])
        logpL = torch.masked_select(
            logpL, torch.stack([stacked_mask_TxN.transpose(0, 1) for _ in range(L)], dim=0
                              ).bool()).view([L, N, T])
        st_muL = ztL[:, :, :, self.q_ode:]  # L, N, T, q_ode
#         print('st_muL shape is', st_muL.shape)
#         print('c shapes are', [c.shape for c in cats])
#         print('st_mu view is', st_muL.view([L*N*T, self.q_ode]).shape)
#         reg_input = torch.cat([st_muL.view([L*N*T, self.q_ode]), *[c.view([L*N*T, -1]) for c in cats]], dim=-1)
        if isinstance(self.cReg, conditional_regressor.cVariationalRegressor):
            Xrec = self.cReg.regress(  # L*N*T, 1
                torch.cat([st_muL.view([L*N*T, self.q_ode]), *[c.view([L*N*T, -1]) for c in cats]], dim=-1))
            Xrec = Xrec.view([L, N, T, 1])
        else:
            assert isinstance(self.cReg, conditional_regressor.cVAE_Sequence)
            Xrec = self.cReg.decode(
                torch.cat([st_muL.view([L*N*T, self.q_ode]), *[c.view([L*N*T, -1]) for c in cats]], dim=-1))
            Xrec = Xrec.view([L, N, T, -1])

        # likelihood and elbo
        if inst_enc:
            assert False, "Not implemented yet!"
        else:
            lhood, kl_z, kl_w = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, Ndata, mask)
            elbo = -lhood - kl_z - self.beta * kl_w

        elbo -= self.gamma * Ndata * lhood_useq.mean()
        return Xrec, qz0_m, qz0_logv, ztL, elbo, lhood, kl_z, self.beta * kl_w, self.gamma * Ndata * lhood_useq.mean()

    def mean_rec(self, X, orig_cats, new_cats, ts, mask, method='dopri5', dt=0.1, t_step=1, generate_mode=False):
        [N, T, d] = X.shape

        max_T = torch.max(ts).item() + t_step
        tdense = dt * torch.arange(start=0, end=max_T, step=t_step, dtype=torch.float).to(mask.device)
        # mask for where the actual points are that we want after integration.
        tdense_mask = torch.zeros(tdense.shape).to(mask.device)
        tdense_mask[(ts[0] / t_step).long()] = 1
        stacked_mask_TxN = torch.stack([tdense_mask for _ in range(N)], dim=-1)
        stacked_mask_TxNx2q = torch.stack([stacked_mask_TxN for _ in range(2 * self.q_ode)], dim=-1)
     
        # encode
        x0 = torch.cat([X[:, 0], *[c[:, 0] for c in orig_cats]], dim=-1)
        qz0_m, qz0_logv = self.cReg.encode(x0)  # shapes: [N, q_style]

        # Now, get the ODE samples
        z0 = self.embed_to_ode(torch.cat([qz0_m, *[c[:, 0] for c in new_cats]], dim=-1))

        # ode
        def ode2vae_mean_rhs(t, vs, f):
            q = vs.shape[1] // 2
            dv = f(vs)
            ds = vs[:, :q]  # N, q
            return torch.cat([dv, ds], 1)  # N, 2q

        f = self.bnn.draw_f(mean=True)
        odef = lambda t, vs: ode2vae_mean_rhs(t, vs, f)  # make the ODE forward function

        zt_mus = []
        z0 = z0.view(N, 2 * self.q_ode)  # N, 2q
        zt = odeint(odef, z0, tdense, method=method)  # T, N, 2q
        # zt = torch.masked_select(zt, stacked_mask_TxNx2q.bool()).view([T, N, 2*self.q_ode])
        zt_mus.append(zt)
        zt_mu = torch.cat(zt_mus, dim=1).permute([1, 0, 2])  #  N, T, 2q

        # decode
        zt_mu = torch.masked_select(
            zt_mu, stacked_mask_TxNx2q.transpose(0, 1).bool()).view([N, T, 2 * self.q_ode])
        st_mu = zt_mu[:, :, self.q_ode:]  # N, T, q_ode
        if isinstance(self.cReg, conditional_regressor.cVariationalRegressor):
            Xrec_mu = self.cReg.regress(
                torch.cat([st_mu.contiguous().view([N*T, self.q_ode]),
                           *[c.view([N*T, -1]) for c in new_cats]], dim=-1)
            )
            Xrec_mu = Xrec_mu.view([N, T, 1])  # N, T, 1
        else:
            assert isinstance(self.cReg, conditional_regressor.cVAE_Sequence)
            Xrec_mu = self.cReg.decode(
                torch.cat([st_mu.contiguous().view([N*T, self.q_ode]),
                           *[c.view([N*T, -1]) for c in new_cats]], dim=-1)
            )
            Xrec_mu = Xrec_mu.view([N, T, -1])

        # error
        if not generate_mode:
            mse = (Xrec_mu - X) ** 2
            mse = torch.mean(torch.stack([
                torch.masked_select(mse[ex], mask[ex].unsqueeze(-1).bool()).sum() for ex in range(N)], dim=0))
        else:
            mse = 0

        return Xrec_mu, mse