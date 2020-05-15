import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import transforms
# Load pytorch modules
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.autograd as autograd
import torch.nn.functional as F
# Some other helpful modules
import time
from Bio.Seq import Seq
from Bio.Alphabet import single_letter_alphabet
import argparse

import data_helper
import utils


BS = 16
NZ = 100
NITER = 400000
LR = 0.0002
B1 = 0.5
NGF = 320
LGF = 11
GEMBED = 4
CBNHIDDEN = 128


# Helpful PyTorch code from external sources:


###################################################################################
##                            Conditional Batch Norm                             ##
## (https://github.com/ap229997/Conditional-Batch-Norm/blob/master/model/cbn.py) ##
###################################################################################
'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class CBN(nn.Module):

    def __init__(self, lstm_size, emb_size, out_size, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.lstm_size = lstm_size # size of the lstm emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = BS 
        self.channels = out_size
        self.height = 100
        self.width = 4

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, lstm_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)
    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''
    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        self.betas.data.resize_(self.batch_size, self.channels)
        self.gammas.data.resize_(self.batch_size, self.channels)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out




###################################################################################################################
##                                                  Spectral Norm                                                ##
## https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py ##
###################################################################################################################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=True)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=True)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data, requires_grad=True)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# class snp_generator(nn.Module):
#     def __init__(self):
#         super(snp_generator, self).__init__()
#
#         self.embed = nn.Embedding(15, 4)
#
#         self.fc = nn.Linear(100, 128*12*1)    ##
#
#         self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=(7, 1), stride=2,
#                          padding=(2, 0), bias=False)
#         self.cbn1 = CBN(4, 128, 64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=(8, 1), stride=2,
#                          padding=(3, 0), bias=False)
#         self.cbn2 = CBN(4, 128, 32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=(16, 4), stride=2,
#                          padding=(7, 0), bias=False)
#         self.softmax = nn.Softmax(dim=3)
#
#     def forward(self, nz, c):
#         embedding = self.embed(c)
#         h = self.fc(nz)
#         h = self.conv1(h.view(-1, 128, 12, 1))
#         h = self.cbn1(h, embedding)
#         h = self.relu1(h)
#         h = self.conv2(h)
#         h = self.cbn2(h, embedding)
#         h = self.relu2(h)
#         h = self.conv3(h)
#         output = self.softmax(h)
#         return output

# class snp_generator_1d(nn.Module):
#     def __init__(self):
#         super(snp_generator_1d, self).__init__()
#
#         self.embed = nn.Embedding(15, 4)
#
#         self.fc = nn.Linear(50, 100)    ##
#         self.dropout = nn.Dropout(0.5)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv1d(1, 256, 1, 1, bias=False)
#         self.cbn1 = CBN(4, 128, 256)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(256, 128, 7, 1, 3, bias=False)
#         self.cbn2 = CBN(4, 128, 128)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv1d(128, 4, 1, 1, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self):
#         embedding = self.embed(c)
#         h = self.fc(nz)
#         h = self.dropout(h)
#         h = self.relu1(h)
#         h = self.conv1(h.view(-1, 1, 100))
#         h = self.cbn1(h, embedding)
#         h = self.relu2(h)
#         h = self.conv2(h)
#         h = self.cbn2(h, embedding)
#         h = self.relu3(h)
#         h = self.conv3(h)
#         output = self.softmax(h)
#         return output.transpose(1, 2).view(-1, 1, 100, 4)

class snp_generator_2d(nn.Module):
    def __init__(self, nz, ne, cbn_h, num_filters, len_filters, dropout=False, concat=False):
        super(snp_generator_2d, self).__init__()

        self.num_filters = num_filters

        self.embed = nn.Embedding(16, ne)
        self.fc = SpectralNorm(nn.Linear(nz, num_filters//2*10))
        self.relu1 = nn.ReLU(True)
        self.cbn1 = CBN(ne, cbn_h, num_filters//2)
        self.up1 = SpectralNorm(nn.ConvTranspose2d(num_filters//2, num_filters, (10, 1), 10, bias=False))    # -1, 320, 100, 1
        self.relu2 = nn.ReLU(True)
        self.cbn2 = CBN(ne, cbn_h, num_filters)
        self.up2 = SpectralNorm(nn.ConvTranspose2d(num_filters, 1, (len_filters, 4), 1, (len_filters//2, 0)))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, nz, c):
        embedding = self.embed(c)
        h = self.fc(nz).view(-1, self.num_filters//2, 10, 1)
        h = self.relu1(h)
        h = self.cbn1(h, embedding)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.cbn2(h, embedding)
        h = self.up2(h)
        output = self.softmax(h)
        return output



class snp_discriminator(nn.Module):
    def __init__(self):
        super(snp_discriminator, self).__init__()

        self.embed = nn.Embedding(16, 8)

        self.conv1 = SpectralNorm(nn.Conv2d(1, 320, kernel_size=(11, 4), stride=1, padding=(5, 0), bias=False))
        self.pool1 = nn.MaxPool2d(kernel_size=(20, 1))
        self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc1 = SpectralNorm(nn.Linear(320*5, 8))
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = SpectralNorm(nn.Linear(8, 1))

    def forward(self, x, c):
        embedding = self.embed(c)
        h = self.conv1(x.view(-1, 1, 100, 4))
        h = self.pool1(h)
        h = self.lrelu1(h)
        h = self.fc1(h.view(-1, 320 * 5))
        h = self.lrelu2(h)
        fc = self.fc2(h)
        proj = (h * embedding).sum(1)
        return (fc + proj).squeeze()








G = snp_generator_2d(NZ, GEMBED, CBNHIDDEN, NGF, LGF).to("cuda")
D = snp_discriminator().to("cuda")

BATCH_SIZE = BS
NZ = NZ
NC = 16


# criterion = nn.BCELoss().to("cuda")
optD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.0, 0.9))
optG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.0, 0.9))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.detach().normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.detach().normal_(1.0, 0.02)
        m.bias.detach().fill_(0)



train_hist = {}
train_hist['d_loss'] = []
train_hist['g_loss'] = []
train_hist['epoch_time'] = []
train_hist['total_time'] = []


def update_train_hist(d_loss, g_loss):
    train_hist['d_loss'].append(d_loss.item())
    train_hist['g_loss'].append(g_loss.item())


def plot_loss(path):
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


def create_fixed_inputs(num_total_imgs):
    # For saving imgs
    fixed_noise = torch.Tensor(num_total_imgs, NZ).normal_(0, 1).to("cuda")

    fixed_c = torch.from_numpy(np.repeat(utils.get_order()[::-1]-1, num_total_imgs//NC, axis=0)).long().to("cuda")

    return fixed_noise, fixed_c


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum().item() / target.numel()
    return acc


def rainbow_text(x, y, strings, colors, boxes, ax=None, **kw):
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    # horizontal version
    for s, c, b in zip(strings, colors, boxes):
        text = ax.text(x, y, s, color=c, transform=t, bbox=b, fontsize=12, family='monospace', **kw)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units="dots")

def one_hot_to_seq(one_hot):
    order_dict = {0:'A', 1:'T', 2:'C', 3:'G'}
    seq = ""
    idxs = np.argmax(one_hot, axis=1)
    for elt in idxs:
        seq += order_dict[elt]
    return Seq(seq, single_letter_alphabet)


def split_seq_helper(seq, c):
    s = seq.reshape(1, 1, 100, 4)
    raw_seq = utils.one_hot_to_seq(seq) 
    motif = utils.get_motif_from_comp(c, ext=True)
    motif_mat = np.loadtxt("/home/pbromley/generative_dhs/memes/new_memes/" + motif)[:, np.array([0, 3, 1, 2])]
    m = utils.MotifMatch(motif_mat, where=True)
    strength, idx = m(torch.from_numpy(s).float())
    idx = max(idx, 0)
    idx = min(idx, 100-motif_mat.shape[0])
    before = raw_seq[0:idx]
    highlight = raw_seq[idx:idx+motif_mat.shape[0]]
    after = raw_seq[idx+motif_mat.shape[0]:]

    high = utils.get_rough_motif_ranges()[c][1]
    low = utils.get_rough_motif_ranges()[c][0]
    strength_norm = (strength-low)/(high-low)
    strength_norm = max(strength_norm, 0)
    strength_norm = min(strength_norm, 1)
    return [(before, "black", None), (highlight, "red", {'facecolor':'yellow', 'alpha':strength_norm}), (after, "black", None)]


def plot_seq(seq, c, pos):
    split = split_seq_helper(seq, c)
    l1, l2, l3 = [], [], []
    for i in range(len(split)):
        l1 += [split[i][0]]
        l2 += [split[i][1]]
        l3 += [split[i][2]]
    rainbow_text(0, 0.0206*pos, l1, l2, l3)


def save_imgs(G, f_noise, f_c, it):
    with torch.no_grad():
        one_hot = G(f_noise, f_c)
    np_one_hot = one_hot.detach().cpu().numpy().reshape(-1, 100, 4)
    components = f_c.detach().cpu().numpy()
    img_path = '/home/pbromley/generative_dhs/images/conditional_gan_strong/%d.png' % it

    plt.figure(figsize=(16, 20))
    for i in range(len(np_one_hot)):
        plot_seq(np_one_hot[i], components[i], i)

    for i, order in enumerate(utils.get_order()):
        plt.plot([-0.005, -0.005], [0.982-(i*(0.982/16)), 0.982-((i+1)*(0.982/16))], color=utils.get_dhs_colors()[order-1], linewidth=7)
    plt.xlim(-0.01, 1.05)
    plt.savefig(img_path)
    plt.close()


def train(G, D, iterations):
    dhs_dataloader, dhs_dataloader_test = data_helper.get_the_dataloaders(BATCH_SIZE, binary_class=None, weighted_sample=True, one_dim=False, data_type='strong')
    dataiter = iter(dhs_dataloader)
    noise = torch.zeros(BATCH_SIZE, NZ).to("cuda")
    fake_c = torch.zeros(BATCH_SIZE).to("cuda")
    label = torch.zeros(BATCH_SIZE).to("cuda")
    fixed_noise, fixed_c = create_fixed_inputs(48)
    for iteration in range(iterations):

        ## GENERATOR ##
        optG.zero_grad()
        noise.normal_(0, 1)
        label.fill_(1)
        fake_c = torch.randint_like(fake_c, 0, NC).long()
        fake = G(noise, fake_c)
        pred_g = D(fake, fake_c)
        g_loss = -pred_g.mean()
        g_loss.backward()
        optG.step()

        ## DISCRIMINATOR ##
        for d_iter in range(5):
            optD.zero_grad()
            batch = next(dataiter, None)
            if (batch is None) or (batch[0].size(0) != BATCH_SIZE):
                dataiter = iter(dhs_dataloader)
                batch = dataiter.next()
            x, c = batch
            x = x.float().to("cuda")
            c = c.long().to("cuda")
            label.fill_(1)
            pred_real = D(x, c)
            d_loss_real = torch.nn.ReLU()(1.0 - pred_real).mean()

            label.fill_(0)
            noise.normal_(0, 1)
            fake_c = torch.randint_like(fake_c, 0, NC).long()
            fake = G(noise, fake_c)
            pred_fake = D(fake.detach(), fake_c)
            d_loss_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()

            d_loss_total = d_loss_fake + d_loss_real
            d_loss_total.backward()

            optD.step()

        if iteration % 1000 == 0:
            update_train_hist(d_loss_total, g_loss)

        if (iteration % 5000 == 0):

            # acc_real = accuracy(pred_real, torch.ones(BATCH_SIZE).cuda())
            # acc_fake = accuracy(pred_fake, torch.zeros(BATCH_SIZE).cuda())
            print('Iter:{0}, Dloss: {1}, Gloss: {2}'.format(
                            iteration, d_loss_total.item(), g_loss.item())
                 )
            save_imgs(G, fixed_noise, fixed_c, iteration)
    path = "/home/pbromley/generative_dhs/saved_models/conditional_gan_strong"
    torch.save(G.state_dict(), path + "-g.pth")
    torch.save(D.state_dict(), path + "-d.pth")
    plot_loss("/home/pbromley/generative_dhs/loss_plots/conditional_gan_strong.png")



#train(G, D, 400000)
