import gen_models
import classifiers
import data_helper
import utils

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
#import matplotlib
#matplotlib.use('agg')
#from matplotlib import pyplot as plt
# import seaborn as sns
from itertools import compress
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', type=str, default='', help='path to generator weights')
parser.add_argument('--model_path', type=str, default='', help='path to model weights')
parser.add_argument('--nz', type=int, default=50, help='length of latent vector')
parser.add_argument('--lr', type=float, default=0.017, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.8, help='Adam parameter beta1')
parser.add_argument('--beta2', type=float, default=0.59, help='Adam parameter beta2')
parser.add_argument('--c', type=int, default=0, help='Class to optimize for')
parser.add_argument('--run_id', type=int, default=0, help='id for which optimization run')
opt = parser.parse_args()


'''
Generate optimized sequences using pretrained generator and evaluation models.
Process is as follows:
    A given number of z input vectors are intialized randomly
    Each of these z vectors can be mapped to a corresponding one-hot DNA sequence via generator
    Each of these corresponding sequences can be evaluated by the evaluation model
    We search for sequences with certain properties by looking for z vectors that map to sequences
     that are evaluated in a desirable way by the evaluation function
        Optimization:

'''
class SequenceOptimizer():
    def __init__(self):
        self.opt_z_arr = None
        self.zs = []
        self.optimized_seqs = []
        self.initial_zs = []
        self.before_seqs = np.array([])
        self.track_seqs = None

        self.G = gen_models.snp_generator_2d_temp_2a(opt.nz, 320, 11) #TODO which gen model framework to load/how to specify?
        # self.G.load_state_dict(torch.load("/home/pbromley/generative_dhs/saved_models/no-condition-ms-g.pth"))
        self.G.load_state_dict(torch.load("saved_models/no-condition-ms-g.pth"))
        self.G.train(False)
        self.G.to("cuda")

        self.model = classifiers.tmp(0.2) #TODO which framework to load/how to specify?
        # self.model.load_state_dict(torch.load("/home/pbromley/generative_dhs/saved_models/classifiers/aug.pth"))
        self.model.load_state_dict(torch.load("saved_models/classifiers/aug.pth"))
        self.model.train(False)
        self.model.to("cuda")
        self.m1 = list(self.model.children())[0]
        self.m2 = list(self.model.children())[-1][:-1]

        self.G.eval()
        self.model.eval()
        self.m1.eval()
        self.m2.eval()

        ##self.train_hist = np.zeros((5, 2)) ######


    def optimize(self, opt_z, c, num_iters, lr, verbose=False, track=False):
        opt_z = opt_z.cuda().requires_grad_()
        # opt_z = opt_z.requires_grad_()

        ###self.loss = []
        if verbose:
            #start = self.G(opt_z).cpu().detach().numpy().squeeze().transpose()
            start = self.G(opt_z).cpu().detach().numpy().squeeze()
            #start = self.G(opt_z).detach().numpy().squeeze()
            print("Starting seq: " + utils.one_hot_to_seq(start))

        optimizer = optim.Adam([opt_z], lr=lr, betas=(opt.beta1, opt.beta2))
        h = opt_z.register_hook(lambda grad: grad + torch.zeros_like(grad).normal_(0, 1e-4))

        for i in range(num_iters):
            self.G.zero_grad()
            self.m1.zero_grad()
            self.m2.zero_grad()
            optimizer.zero_grad()
            seq = self.G(opt_z).transpose(2, 3).view(1, 4, 100)
            tmp = self.m1(seq).view(-1, 512)
            pred = self.m2(tmp).squeeze()
            loss = -(pred[c])
            loss.backward()
            optimizer.step()


            ###self.loss.append(loss.item())
            ##if (-1*loss.item() >= 100) and (self.train_hist[self.seq_num][0] == 0): ######
            ##    self.train_hist[self.seq_num][0] = i ######

            if i % 999 == 0:
                print("    Iter: {0}, Prediction: {1}".format(i, pred[c].cpu().detach().item()))
                # print("    Iter: {0}, Prediction: {1}".format(i, pred[c].detach().item()))

            if track:
                s = utils.one_hot_to_seq(seq.cpu().detach().numpy().squeeze().transpose())
                z = opt_z.cpu().clone().detach().numpy()
                self.track_seqs.append((s, z))

        ##self.train_hist[self.seq_num][1] = -1 * loss.item() ######

        one_hot = seq.cpu().detach().numpy().squeeze().transpose()
        opt_z = opt_z.cpu().detach().numpy()
        # one_hot = seq.detach().numpy().squeeze().transpose()
        # opt_z = opt_z.detach().numpy()

        if verbose:
            after_seq = utils.one_hot_to_seq(one_hot)
            model_out = self.model(seq)
            print("Optimized seq: " + after_seq)
            print("Model output: ", end="")
            print(model_out)

        return opt_z, one_hot


    def sample_seqs(self, num_seqs, c, num_iters, lr=0.01, verbose=False, use_fixed_zs=True, track=False):
        self.init_dirs(c)
        print("Optimizing {0} sequences for component {1}".format(num_seqs, c+1))
        self.zs = []
        self.optimized_seqs = []
        ##self.train_hist = np.zeros((5, 2)) ######
        if use_fixed_zs:
            # all_fixed_zs = np.load("/home/pbromley/generative_dhs/data_numpy/fixed_zs_{0}.npy".format(opt.nz))
            all_fixed_zs = np.load("seqs/initialized/nz{0}/fixed_zs_{0}.npy".format(opt.nz))
            self.initial_zs = all_fixed_zs[:num_seqs]
            self.opt_z_arr = torch.from_numpy(self.initial_zs).float()
        else:
            self.initial_zs = np.random.normal(0, 1, (num_seqs, opt.nz))
            self.opt_z_arr = torch.from_numpy(self.initial_zs)
        self.before_seqs = self.G(self.opt_z_arr.cuda()).cpu().detach().numpy().squeeze()
        # self.before_seqs = self.G(self.opt_z_arr).cpu().detach().numpy().squeeze()
        for i in range(num_seqs):
            print("Seq {0}: ".format(i))
            if track:
                self.track_seqs = []
            z = self.opt_z_arr[i]

        ##    self.seq_num = i ######
            opt_z, one_hot = self.optimize(z, c, num_iters, lr, verbose=verbose, track=track)
            self.zs.append(opt_z)
            self.optimized_seqs.append(one_hot)

            if track:
                path = Path(
                    "seqs/progress/nz{0}/comp{1}/run{2}/".format(opt.nz, c+1, opt.run_id)
                )
                with open(str(path / "seq{0}.txt".format(i)), "w") as t:
                    for seq, _ in self.track_seqs:
                        t.write(str(seq) + "\n")
                #### LOGIC FOR SAVING ALL Zs
                # track_zs = np.zeros((len(self.track_seqs), opt.nz))
                # for j in range(len(self.track_seqs)):
                #     track_zs[j] = self.track_seqs[j][1]
                # np.save(path / "z{0}".format(i), track_zs)

        ##for i in range(5):  ######
        ##    print(np.around(self.train_hist[i][0], decimals=2), end="\t")  ######
        ##    print(np.around(self.train_hist[i][1], decimals=2), end="\t")  ######
        ##    print(opt.lr, end="\t")  ######
        ##    print(opt.beta1, end="\t") ######
        ##    print(opt.beta2) ######

        print("Completed optimization of {0} sequences for component {1}".format(i+1, c+1))
        self.optimized_seqs = np.array(self.optimized_seqs)
        #self.motif_stats(c, save_deltas=False)


    def init_dirs(self, c):
        save_paths = [
            "seqs/initialized/nz{0}/",
            "seqs/progress/nz{0}/comp{1}/run{2}/",
            "seqs/optimized/nz{0}/comp{1}/run{2}/",
        ]

        for path in save_paths:
            p = Path(path.format(opt.nz, c+1, opt.run_id))
            p.mkdir(parents=True, exist_ok=True)

        print("Initialized save directories")



    # def motif_stats(self, c, save_deltas=True):
    #     x_train, y_train, _, _ = data_helper.load_the_data("strong")
    #     real_seqs = [utils.one_hot_to_seq(o) for o in x_train]
    #     before_seqs = [utils.one_hot_to_seq(o) for o in self.before_seqs]
    #     after_seqs = [utils.one_hot_to_seq(o) for o in self.optimized_seqs]
    #
    #     real_c_comp = utils.calc_overall_composition(list(compress(real_seqs, (y_train == c))))
    #     real_notc_comp = utils.calc_overall_composition(list(compress(real_seqs, (y_train != c))))
    #     fake_c_comp = utils.calc_overall_composition(after_seqs)
    #     fake_notc_comp = utils.calc_overall_composition(before_seqs)
    #     print("Composition of Real Component Seqs: ", end="")
    #     print(real_c_comp)
    #     print("Composition of Real Non-Component Seqs: ", end="")
    #     print(real_notc_comp)
    #     print("Composition of Synthetic Component Seqs: ", end="")
    #     print(fake_c_comp)
    #     print("Composition of Synthetic Seqs Before Optimization: ", end="")
    #     print(fake_notc_comp)
    #
    #     print("Plotting Motif Histogram...")
    #     real_c_scan = utils.get_motif_scan(x_train[y_train == c], c)
    #     real_notc_scan = utils.get_motif_scan(x_train[y_train != c], c)
    #     fake_c_scan = utils.get_motif_scan(self.optimized_seqs, c)
    #     fake_notc_scan = utils.get_motif_scan(self.before_seqs, c)
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
    #     sns.distplot(real_c_scan, hist=False, ax=ax[0], color=utils.get_dhs_colors()[4].tolist(), label="Lymphoid Sequences")
    #     sns.distplot(real_notc_scan, hist=False, ax=ax[0], color='black', label="Non-Lymphoid Sequences")
    #     ax[0].set_title("Real Seqs", fontsize=18)
    #     ax[0].set_xlabel("Motif Scores for Real Seqs", fontsize=18)
    #     ax[0].set_ylabel("Relative Frequency", fontsize=18)
    #     ax[0].legend(fontsize=12)
    #     ax[0].tick_params(labelsize=16)
    #     ax[0].tick_params(labelsize=16)
    #
    #     sns.distplot(fake_c_scan, hist=False, color=utils.get_dhs_colors()[4].tolist(), ax=ax[1], label="Tuned to Lymphoid")
    #     sns.distplot(fake_notc_scan, hist=False, color='black', ax=ax[1], label="Before Tuning")
    #     ax[1].set_title("Synthetic Seqs", fontsize=18)
    #     ax[1].set_xlabel("Motif Scores for Synthetic Seqs", fontsize=18)
    #     ax[1].legend(fontsize=12)
    #     ax[0].tick_params(labelsize=16)
    #     ax[0].tick_params(labelsize=16)
    #
    #     motif = utils.get_motif_from_comp(c, ext=False)
    #     title = "Motif Distribution Comparisons (Component {0}, ".format(c+1) + motif + ")"
    #     fig.text(0.5, 0.93, title, fontsize=24, ha="center", va="center")
    #     path = "/home/pbromley/generative_dhs/optimization_figures/motif-{0}.png".format(c+1)
    #     fig.savefig(path)
    #     plt.close()
    #     print("Plotted Motif Histogram in " + path)
    #
    #     if save_deltas:
    #         print("Saving Deltas...")
    #         deltas = fake_c_scan - fake_notc_scan
    #         path = "/home/pbromley/generative_dhs/optimized/nz{0}/deltas-ss-{1}.npy".format(opt.nz, c+1)
    #         np.save(path, deltas)
    #         print("Saved Deltas in " + path)


    def save(self, c):
        # np.save("/home/pbromley/generative_dhs/optimized/nz{0}/zs-ss-{1}.npy".format(opt.nz, c+1), np.array(self.zs))
        # np.save("/home/pbromley/generative_dhs/optimized/nz{0}/seqs-ss-{1}.npy".format(opt.nz, c+1), self.optimized_seqs)
        path = Path("seqs/optimized/nz{0}/comp{1}/run{2}".format(opt.nz, c+1, opt.run_id))
        np.save(path / "zs.npy", np.array(self.zs))
        np.save(path / "seqs.npy", self.optimized_seqs)
        np.save(path / "initial_zs.npy", self.initial_zs)


if __name__ == "__main__":
    SO = SequenceOptimizer()

    for component in range(0, 16):  # all 16 components
        SO.sample_seqs(
            1000,               # num seqs to optimize
            component,          # component to optimize for
            2000,               # num optimization iterations
            lr=opt.lr,          # learning rate
            verbose=True,       # print stats about optimization process
            use_fixed_zs=True,  # use fixed initial seeds
            track=True          # track all seq data during optimization
        )
        SO.save(component)
