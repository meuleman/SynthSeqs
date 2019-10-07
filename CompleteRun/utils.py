import numpy as np
#import matplotlib
#matplotlib.use('agg')
#from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.Alphabet import single_letter_alphabet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.detach().normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.detach().normal_(1.0, 0.02)
        m.bias.detach().fill_(0)

def initialize_train_hist():
    train_hist = {}
    train_hist['d_loss'] = []
    train_hist['g_loss'] = []
    train_hist['epoch_time'] = []
    train_hist['total_time'] = []
    return train_hist

def update_train_hist(train_hist, d_loss, g_loss):
    train_hist['d_loss'].append(d_loss.item())
    train_hist['g_loss'].append(g_loss.item())
    return train_hist

def plot_loss(train_hist, path):
    x = range(len(train_hist['d_loss']))
    d_loss_hist = train_hist['d_loss']
    g_loss_hist = train_hist['g_loss']
    plt.figure(figsize=(20, 20))
    plt.plot(x, d_loss_hist, label='d_loss')
    plt.plot(x, g_loss_hist, label='g_loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.savefig(path)
    plt.close()

def create_fixed_inputs(num_total_imgs, nz):
    fixed_noise = torch.Tensor(num_total_imgs, nz).normal_(0, 1).to("cuda")
    return fixed_noise

# Assumes no seqs have N, S, etc
def seq_to_one_hot(seq):
    order_dict = {'A':0, 'T':1, 'C':2, 'G':3}
    x = np.zeros((len(seq), 4))
    for (i, bp) in enumerate(seq):
        x[i, order_dict[bp]] = 1
    return x

def one_hot_to_seq(one_hot):
    order_dict = {0:'A', 1:'T', 2:'C', 3:'G'}
    seq = ""
    idxs = np.argmax(one_hot, axis=1)
    for elt in idxs:
        seq += order_dict[elt]
    return Seq(seq, single_letter_alphabet)

def clean_up_one_hot(one_hot):
    o = np.zeros_like(one_hot)
    idxs = np.argmax(one_hot, axis=1)
    o[np.arange(100), idxs] = 1
    return o

# Calculate the avg A, T, C, G content per sequence for an array of sequences
#  (output is np array of format [A/n, T/n, C/n, G/n] where n = len(seqs))
def calc_overall_composition(seqs):
    atcg = np.array([0, 0, 0, 0])
    for seq in seqs:
        atcg[0] += seq.count("A")
        atcg[1] += seq.count("T")
        atcg[2] += seq.count("C")
        atcg[3] += seq.count("G")
    return atcg / len(seqs)

def count_cpg_sites(seqs):
    count = 0
    for seq in seqs:
        count += seq.count("CG")
    return count/len(seqs)

def save_imgs(G, bs, f_noise, it, path, one_dim=False):
    G.train(False)
    with torch.no_grad():
        num_imgs = len(f_noise)
        if one_dim:
            np_one_hot = np.zeros((num_imgs, 4, 100))
        else:
            np_one_hot = np.zeros((num_imgs, 100, 4))
        for i in range(num_imgs//bs):
            np_one_hot[i*bs:(i+1)*bs] = G(f_noise[i*bs:(i+1)*bs]).detach().cpu().numpy().squeeze()
        leftover = num_imgs % bs
        if leftover != 0:
            np_one_hot[num_imgs-leftover:] = G(f_noise[num_imgs-leftover:]).detach().cpu().numpy().squeeze()
    if one_dim:
        np_one_hot = np_one_hot.transpose(0, 2, 1)
    seqs = [one_hot_to_seq(one_hot) for one_hot in np_one_hot]
    img_path = path + str(it) + ".png"

    plt.figure(figsize=(18, 20))
    for i in range(num_imgs):
        plt.text(0, i/num_imgs, seqs[i], fontsize=14)
    plt.savefig(img_path)
    plt.close()
    G.train(True)

def get_dhs_colors():
    return np.array([[195,195,195],
                     [187,45,212],
                     [5,193,217],
                     [122,0,255],
                     [254,129,2],
                     [74,104,118],
                     [255,229,0],
                     [4,103,253],
                     [7,175,0],
                     [105,33,8],
                     [185,70,29],
                     [76,125,20],
                     [0,149,136],
                     [65,70,19],
                     [255,0,0],
                     [8,36,91]])/255

def get_order():
    return np.array([7,5,15,9,12,14,3,8,13,2,4,6,16,11,10,1])

def get_motifs():
    return ["GCM1_GCM_1", "IRF4_IRF_1", "SPI1_ETS_1", "MEF2A_MADS_1",
              "MSC_bHLH_1", "ERG_ETS_2", "V_OCT4_01", "NEUROD2_bHLH_1",
              "HNF4A_nuclearreceptor_3", "JDP2_bZIP_1", "JDP2_bZIP_1", "HNF1B_homeodomain_1",
              "TP63_p53l_1", "Foxl1_primary", "PAX2_PAX_1", "CTCF_C2H2_1"]

def get_abbrev_motifs():
    return ["GCM1", "IRF", "ETS/SPI1", "MEF2",
              "E-box", "ETS/ERG", "Oct-4", "NeuroD/Olig",
              "HNF4A", "AP-1", "AP-1", "HNF1",
              "TP63", "FOX", "PAX2", "CTCF"]

def get_component_class_names():
    return ["placenta", "lymphoid", "HSC/myeloid/erythroid", "cardiac", "musculoskeletal",
            "vascular/endothelial", "embryonic", "neuronal", "digestive", "fibroblast1",
            "fibroblast2", "epithelial/kidney(cancer)", "epithelial", "fetal lung",
            "fetal kidney", "tissue-invariant"]

def get_motif_from_comp(comp, ext=True):
    motif = get_motifs()[np.where(get_order() == (comp+1))[0][0]]
    if ext:
        return motif + ".txt"
    else:
        return motif

class MotifMatch(nn.Module):
    def __init__(self, pwm, where=False):
        super(MotifMatch, self).__init__()
        self.motif_length = pwm.shape[0]
        kernel = pwm.reshape(1, 1, self.motif_length, 4)
        self.kernel = nn.Parameter(data=torch.FloatTensor(kernel), requires_grad=False)
        self.where = where

    def forward(self, x):
        seq = one_hot_to_seq(x.cpu().numpy().squeeze())
        rc_seq = seq.reverse_complement()
        rc_seq_one_hot = torch.FloatTensor(seq_to_one_hot(rc_seq).reshape(1, 1, x.size(2), x.size(3))).cuda()
        conv1 = F.conv2d(x, self.kernel, padding=(self.motif_length-1, 0))
        conv2 = F.conv2d(rc_seq_one_hot, self.kernel, padding=(self.motif_length-1, 0))
        conv = torch.cat([conv1, conv2], 2)
        out = torch.max(conv)
        if self.where:
            l = conv1.size(2)
            idx = torch.argmax(conv)
            loc = (idx % l)-(self.motif_length-1)
            if idx >= l:
                strand = "-"
                loc = (len(seq)-1) - loc
            else:
                strand = "+"
            return out.squeeze(), loc, strand
        else:
            return out.squeeze()

def get_motif_scan(x, comp):
    motif = get_motif_from_comp(comp, ext=True)
    motif_mat = np.loadtxt("/home/pbromley/generative_dhs/memes/new_memes/" + motif)[:, np.array([0, 3, 1, 2])]
    m = MotifMatch(motif_mat).cuda()
    motif_scan_arr = np.zeros(len(x))
    for i in range(len(x)):
        motif_scan_arr[i] = m(torch.FloatTensor(x[i].reshape(1, 1, 100, 4)).cuda()).item()
    return motif_scan_arr


def get_motif_scan_from_meme(x, meme):
    motif_mat = np.loadtxt("/home/pbromley/generative_dhs/memes/new_memes/meme_set/" + meme)[:, np.array([0, 3, 1, 2])]
    m = MotifMatch(motif_mat).cuda()
    motif_scan_arr = np.zeros(len(x))
    for i in range(len(x)):
        motif_scan_arr[i] = m(torch.FloatTensor(x[i].reshape(1, 1, 100, 4)).cuda()).item()
    return motif_scan_arr


class MotifMatchLong(nn.Module):
    def __init__(self, pwm, where=False):
        super(MotifMatchLong, self).__init__()
        self.motif_length = pwm.shape[0]
        kernel = pwm.reshape(1, 1, self.motif_length, 4)
        self.kernel = nn.Parameter(data=torch.FloatTensor(kernel), requires_grad=False)
        self.where = where

    def forward(self, x):
        seq = one_hot_to_seq(x.cpu().numpy().squeeze())
        rc_seq = seq.reverse_complement()
        rc_seq_one_hot = torch.FloatTensor(seq_to_one_hot(rc_seq).reshape(1, 1, x.size(2), x.size(3)))
        conv1 = F.conv2d(x, self.kernel, padding=(self.motif_length-1, 0))
        conv2 = F.conv2d(rc_seq_one_hot, self.kernel, padding=(self.motif_length-1, 0))
        return conv1.detach().cpu().numpy(), conv2.detach().cpu().numpy()



def get_rough_motif_ranges():
    '''Extremely rough motif value ranges based on density plots, in 1-16 order NOT official'''
    return np.array([[7.0, 9.0],
                     [5.0, 7.0],
                     [6.4, 8.5],
                     [5.0, 5.0],
                     [7.0, 9.0],
                     [5.8, 8.0],
                     [5.0, 6.5],
                     [4.6, 6.0],
                     [5.8, 8.0],
                     [7.1, 9.0],
                     [6.8, 9.0],
                     [5.1, 7.0],
                     [7.1, 9.5],
                     [6.2, 8.1],
                     [7.0, 9.4],
                     [7.5, 10.2]])


def make_fixed_zs(nz, num_seqs, path, seed=None):
    if seed:
        np.random.seed(seed)
    zs = np.random.normal(0, 1, (num_seqs, nz))
    np.save(path, zs)
