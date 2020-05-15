import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import torch
import data_helper
import utils
import classifiers

MODEL_PATH = "/home/pbromley/generative_dhs/saved_models/classifiers/aug.pth"
N_COMPONENTS = 16
BS = 256 

_, _, X_TEST, Y_TEST = data_helper.load_the_data('strong')
X_TEST = X_TEST.transpose(0, 2, 1)
_, NMF_TEST, _, SIG_TEST = data_helper.load_nmf_and_sig_data()

MODEL = classifiers.tmp(0.2)
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.train(False)
MODEL.eval()
MODEL.to("cuda")


def get_pred_idxs(comp, comp_idx):
    x_comp = X_TEST[comp_idx]
    nmf_comp = NMF_TEST[comp_idx]
    comp_pred = np.zeros((x_comp.shape[0], N_COMPONENTS))
    for i in range(x_comp.shape[0]//BS + 1):
        if i < x_comp.shape[0]//BS:
            seq_batch = torch.from_numpy(x_comp[i*BS:(i+1)*BS]).cuda().float()
            comp_pred[i*BS:(i+1)*BS] = MODEL(seq_batch).detach().cpu().numpy()
        else:
            seq_batch = torch.from_numpy(x_comp[i*BS:]).cuda().float()
            comp_pred[i*BS:] = MODEL(seq_batch).detach().cpu().numpy()
    correct_idx = comp_pred.argmax(axis=1) == comp
    wrong_idx = comp_pred.argmax(axis=1) != comp
    return nmf_comp, correct_idx, wrong_idx

def get_dicts():
    pred_dict = {}
    nmf_dict = {}
    idx_dict = {}
    for i in range(16):
        idx = Y_TEST == i
        n, r, w = get_pred_idxs(i, idx)
        pred_dict["Correct " + str(i)] = r
        pred_dict["Wrong " + str(i)] = w
        nmf_dict[i] = n
        idx_dict[i] = idx
    return idx_dict, nmf_dict, pred_dict

IDX_DICT, NMF_DICT, PRED_DICT = get_dicts()
print(PRED_DICT["Correct 6"].shape)        

def plot_figure(comps, which="val"):
    rc = np.ceil(np.sqrt(len(comps))).astype(int)
    fig, ax = plt.subplots(rc, rc, figsize=(rc*10, rc*10))
    for i in range(len(comps)):
        comp = comps[i]-1
        nmf_comp = NMF_DICT[comp]
        correct_idx = PRED_DICT["Correct " + str(comp)]
        wrong_idx = PRED_DICT["Wrong " + str(comp)]
        if which == "val":
            to_plot_c = nmf_comp[correct_idx, comp]
            to_plot_w = nmf_comp[wrong_idx, comp]
            xlab = "NMF Loading Value for Component {0}".format(comp+1)
            title = "NMF Loading Value for Correct, Incorrect Seqs"
        elif which == "proportion":
            to_plot_c = nmf_comp[correct_idx, comp]/nmf_comp[correct_idx].sum(axis=1)
            to_plot_w = nmf_comp[wrong_idx, comp]/nmf_comp[wrong_idx].sum(axis=1)
            xlab = "Proportion of NMF Loading for {0}".format(comp+1)
            title = "Proportion of NMF Loading Taken Up By Majority Component for Correct, Incorrect Seqs"
        elif which == "total":
            to_plot_c = nmf_comp[correct_idx].sum(axis=1)
            to_plot_w = nmf_comp[wrong_idx].sum(axis=1)
            xlab = "Sum of NMF Loading"
            title = "Sum of NMF Loading for Correct, Incorrect Seqs"
        elif which == "motif":
            x_comp = X_TEST[IDX_DICT[comp]].transpose(0, 2, 1)
            to_plot_c = utils.get_motif_scan(x_comp[correct_idx], comp)
            to_plot_w = utils.get_motif_scan(x_comp[wrong_idx], comp)
            motif = utils.get_motif_from_comp(comp, ext=False)
            xlab = "Max of Motif Scan Over Sequence (" + motif + ")"
            title = "Motif Scan Maxes for Correctly/Incorrectly Classified Sequences"
        elif which == "cpg":
            x_comp = X_TEST[IDX_DICT[comp]].transpose(0, 2, 1)
            correct_seqs = [utils.one_hot_to_seq(o) for o in x_comp[correct_idx]]
            wrong_seqs = [utils.one_hot_to_seq(o) for o in x_comp[wrong_idx]]
            to_plot_c, to_plot_w = [], []
            for j in range(len(correct_seqs)):
                to_plot_c.append(utils.count_cpg_sites([correct_seqs[j]]))
            for j in range(len(wrong_seqs)):
                to_plot_w.append(utils.count_cpg_sites([wrong_seqs[j]]))
            xlab = "Number of CpG Sites in Sequence"
            title = "CpG Sites in Correctly/Incorrectly Classified Sequences"
        elif which == "cg":
            x_comp = X_TEST[IDX_DICT[comp]].transpose(0, 2, 1)
            correct_seqs = [utils.one_hot_to_seq(o) for o in x_comp[correct_idx]]
            wrong_seqs = [utils.one_hot_to_seq(o) for o in x_comp[wrong_idx]]
            to_plot_c, to_plot_w = [], []
            for j in range(len(correct_seqs)):
                to_plot_c.append(utils.calc_overall_composition([correct_seqs[j]])[2:].sum())
            for j in range(len(wrong_seqs)):
                to_plot_w.append(utils.calc_overall_composition([wrong_seqs[j]])[2:].sum())
            xlab = "G/C Pct"
            title = "Percent of Sequences that are G/C" 
        ax[i//rc][i%rc].hist(to_plot_c, bins=int(max(to_plot_c)) - int(min(to_plot_c)), density=True, alpha=0.5)
        ax[i//rc][i%rc].hist(to_plot_w, bins=int(max(to_plot_w)) - int(min(to_plot_w)), density=True, alpha=0.5)
        ax[i//rc][i%rc].set_title("Component " + str(comp+1), fontsize=30)
        ax[i//rc][i%rc].set_xlabel(xlab, fontsize=20)
        ax[i//rc][i%rc].set_ylabel("Density", fontsize=20)
        ax[i//rc][i%rc].legend(["Correct", "Incorrect"])
    fig.text(0.5, 0.9, title, fontsize=24, ha="center", va="center")
    fig.savefig("/home/pbromley/generative_dhs/classifier_figures/" + which + ".png")
    plt.close()


def get_motif_arrays(comp, correct_idx, wrong_idx):
    motif = utils.get_motifs()[np.where(utils.get_order() == (comp+1))[0][0]] + ".txt"
    motif_mat = np.loadtxt("/home/pbromley/generative_dhs/memes/new_memes/" + motif)[:, np.array([0, 3, 1, 2])]
    m = utils.MotifMatch(motif_mat)
    motif_correct = np.zeros(correct_idx.sum())
    motif_wrong = np.zeros(wrong_idx.sum())
    x_comp = X_TEST[IDX_DICT[comp]].transpose(0, 2, 1)
    for i in range(len(motif_correct)):
        motif_correct[i] = m(torch.FloatTensor(x_comp[correct_idx][i].reshape(1, 1, 100, 4))).item()
    for i in range(len(motif_wrong)):
        motif_wrong[i] = m(torch.FloatTensor(x_comp[wrong_idx][i].reshape(1, 1, 100, 4))).item()

    return motif_correct, motif_wrong, motif[:-4]
    

if __name__ == "__main__":
    plot_figure(utils.get_order(), which="cg")
    #plot_figure(utils.get_order(), which="cpg")
    #plot_figure(utils.get_order(), which="motif")
    #plot_figure(utils.get_order(), which="val")
    #plot_figure(utils.get_order(), which="proportion")
    #plot_figure(utils.get_order(), which="total")
