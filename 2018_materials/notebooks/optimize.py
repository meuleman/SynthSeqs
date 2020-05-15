import gen_models
import classifiers
import data_helper
import utils

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import compress
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', type=str, default='', help='path to generator weights')
parser.add_argument('--model_path', type=str, default='', help='path to model weights')
parser.add_argument('--nz', type=int, default=100, help='length of latent vector')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.8, help='Adam parameter beta1')
parser.add_argument('--beta2', type=float, default=0.9, help='Adam parameter beta2')
parser.add_argument('--c', type=int, default=0, help='Class to optimize for')
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
        self.before_seqs = np.array([])
        self.track_seqs = None 

        self.G = gen_models.snp_generator_2d_temp_2a(opt.nz, 320, 11) #TODO which gen model framework to load/how to specify?
        self.G.load_state_dict(torch.load("/home/pbromley/generative_dhs/saved_models/no-condition-ms-g.pth"))
        self.G.train(False)
        self.G.to("cuda")

        self.model = classifiers.tmp(0.2) #TODO which framework to load/how to specify?
        self.model.load_state_dict(torch.load("/home/pbromley/generative_dhs/saved_models/classifiers/aug.pth"))
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

        ###self.loss = []
        if verbose:
            start = self.G(opt_z).cpu().detach().numpy().squeeze().transpose()
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

            if track:
                if i % 20 == 0:
                    s = utils.one_hot_to_seq(seq.cpu().detach().numpy().squeeze().transpose())
                    self.track_seqs.append(s)


        ##self.train_hist[self.seq_num][1] = -1 * loss.item() ######

        one_hot = seq.cpu().detach().numpy().squeeze().transpose()
        opt_z = opt_z.cpu().detach().numpy()

        if verbose:
            after_seq = utils.one_hot_to_seq(one_hot)
            model_out = self.model(seq)
            print("Optimized seq: " + after_seq)
            print("Model output: ", end="")
            print(model_out)

        return opt_z, one_hot


    def sample_seqs(self, num_seqs, c, num_iters, lr=0.01, verbose=False, use_fixed_zs=True, track=False):
        print("Optimizing {0} sequences for component {1}".format(num_seqs, c+1))
        self.zs = []
        self.optimized_seqs = []
        ##self.train_hist = np.zeros((5, 2)) ######
        if use_fixed_zs:
            all_fixed_zs = np.load("/home/pbromley/generative_dhs/data_numpy/fixed_zs_{0}.npy".format(opt.nz))
            self.opt_z_arr = torch.from_numpy(all_fixed_zs[:num_seqs]).float()
        else:
            self.opt_z_arr = torch.zeros(num_seqs, opt.nz).normal_(0, 1)
        self.before_seqs = self.G(self.opt_z_arr.cuda()).cpu().detach().numpy().squeeze()
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
                with open("/home/pbromley/generative_dhs/notebooks/track2/for-deltas-{0}-{1}.txt".format(i, c+1), "w") as t:
                    for seq in self.track_seqs:
                        t.write(str(seq) + "\n")

        ##for i in range(5):  ######
        ##    print(np.around(self.train_hist[i][0], decimals=2), end="\t")  ######
        ##    print(np.around(self.train_hist[i][1], decimals=2), end="\t")  ######
        ##    print(opt.lr, end="\t")  ######
        ##    print(opt.beta1, end="\t") ######
        ##    print(opt.beta2) ######

        print("Completed optimization of {0} sequences for component {1}".format(i+1, c+1))
        self.optimized_seqs = np.array(self.optimized_seqs)
        #self.motif_stats(c, save_deltas=False)


    def motif_stats(self, c, save_deltas=True):
        x_train, y_train, _, _ = data_helper.load_the_data("strong")
        real_seqs = [utils.one_hot_to_seq(o) for o in x_train]
        before_seqs = [utils.one_hot_to_seq(o) for o in self.before_seqs]
        after_seqs = [utils.one_hot_to_seq(o) for o in self.optimized_seqs]

        real_c_comp = utils.calc_overall_composition(list(compress(real_seqs, (y_train == c))))
        real_notc_comp = utils.calc_overall_composition(list(compress(real_seqs, (y_train != c))))
        fake_c_comp = utils.calc_overall_composition(after_seqs)
        fake_notc_comp = utils.calc_overall_composition(before_seqs)
        print("Composition of Real Component Seqs: ", end="")
        print(real_c_comp)
        print("Composition of Real Non-Component Seqs: ", end="")
        print(real_notc_comp)
        print("Composition of Synthetic Component Seqs: ", end="")
        print(fake_c_comp)
        print("Composition of Synthetic Seqs Before Optimization: ", end="")
        print(fake_notc_comp)

        print("Plotting Motif Histogram...")
        real_c_scan = utils.get_motif_scan(x_train[y_train == c], c)
        real_notc_scan = utils.get_motif_scan(x_train[y_train != c], c)
        fake_c_scan = utils.get_motif_scan(self.optimized_seqs, c)
        fake_notc_scan = utils.get_motif_scan(self.before_seqs, c)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
        sns.distplot(real_c_scan, hist=False, ax=ax[0], color=utils.get_dhs_colors()[4].tolist(), label="Lymphoid Sequences")
        sns.distplot(real_notc_scan, hist=False, ax=ax[0], color='black', label="Non-Lymphoid Sequences")
        ax[0].set_title("Real Seqs", fontsize=18)
        ax[0].set_xlabel("Motif Scores for Real Seqs", fontsize=18)
        ax[0].set_ylabel("Relative Frequency", fontsize=18)
        ax[0].legend(fontsize=12)
        ax[0].tick_params(labelsize=16)
        ax[0].tick_params(labelsize=16)

        sns.distplot(fake_c_scan, hist=False, color=utils.get_dhs_colors()[4].tolist(), ax=ax[1], label="Tuned to Lymphoid")
        sns.distplot(fake_notc_scan, hist=False, color='black', ax=ax[1], label="Before Tuning")
        ax[1].set_title("Synthetic Seqs", fontsize=18)
        ax[1].set_xlabel("Motif Scores for Synthetic Seqs", fontsize=18)
        ax[1].legend(fontsize=12)
        ax[0].tick_params(labelsize=16)
        ax[0].tick_params(labelsize=16)

        motif = utils.get_motif_from_comp(c, ext=False)
        title = "Motif Distribution Comparisons (Component {0}, ".format(c+1) + motif + ")"
        fig.text(0.5, 0.93, title, fontsize=24, ha="center", va="center")
        path = "/home/pbromley/generative_dhs/optimization_figures/motif-{0}.png".format(c+1)
        fig.savefig(path)
        plt.close()
        print("Plotted Motif Histogram in " + path)

        if save_deltas:
            print("Saving Deltas...")
            deltas = fake_c_scan - fake_notc_scan
            path = "/home/pbromley/generative_dhs/optimized/nz{0}/deltas-ss-{1}.npy".format(opt.nz, c+1)
            np.save(path, deltas)
            print("Saved Deltas in " + path)


    def save(self, c):
        np.save("/home/pbromley/generative_dhs/optimized/nz{0}/zs-ss-{1}.npy".format(opt.nz, c+1), np.array(self.zs))
        np.save("/home/pbromley/generative_dhs/optimized/nz{0}/seqs-ss-{1}.npy".format(opt.nz, c+1), self.optimized_seqs)


if __name__ == "__main__":
    SO = SequenceOptimizer() 
    c = opt.c 

    ##lrs = np.arange(0.009, 0.1, 0.008)
    ##b1s = np.arange(0.3, 1.0, 0.1)
    ##b2s = np.arange(0.59, 1.0, 0.1)
    ##for i in range(len(lrs)):
    ##    for j in range(len(b1s)):
    ##        for k in range(len(b2s)):
    ##            opt.lr = lrs[i]
    ##            opt.beta1 = b1s[j]
    ##            opt.beta2 = b2s[k]
    ##            SO.sample_seqs(5, 12, 5000, lr=opt.lr)  
 
    for i in np.array([10, 11, 12, 13, 14, 15]):
        c = i 
        SO.sample_seqs(200, c, 4000, lr=opt.lr, use_fixed_zs=True, track=True)
    
    #SO.sample_seqs(2000, 4, 2000, lr=opt.lr, use_fixed_zs=False, track=False)
    #SO.save(4)

    #SO.sample_seqs(1000, 8, 5000, lr=opt.lr, use_fixed_zs=False, track=False)
    #SO.save(8)

    #SO.sample_seqs(1000, 12, 5000, lr=opt.lr, use_fixed_zs=False, track=False)
    #SO.save(12)

    #SO.sample_seqs(2000, 14, 2000, lr=opt.lr, use_fixed_zs=True, track=False)
    #SO.save(c+3)

    #SO.sample_seqs(2000, 5, 2000, lr=opt.lr, use_fixed_zs=True, track=False)
    #SO.save(c+4)

    #SO.sample_seqs(2000, c+5, 2000, lr=opt.lr, use_fixed_zs=True, track=False)
    #SO.save(c+5)

    #SO.sample_seqs(2000, c+6, 2000, lr=opt.lr, use_fixed_zs=True, track=False)
    #SO.save(c+6)

    #SO.sample_seqs(2000, c+7, 2000, lr=opt.lr, use_fixed_zs=True, track=False)
    #SO.save(c+7)

