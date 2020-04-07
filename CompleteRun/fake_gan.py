import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.datasets as dset
import gen_models

class FakeGCDataset(Dataset):
    """
    Dataset for loading fake GC skewed data for GAN analysis.
    """

    def __init__(self, seqs, transform=None):
        """
        Args:
            seqs (list/np.array): List of one-hot numpy array DNA sequences
            components (list/np.array): List of integers indicating components 1-15
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seqs = seqs
        self.transform = transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        one_hot = self.seqs[idx]

        if self.transform:
            image = self.transform(one_hot)

        return one_hot

BS = 128
NZ = 100
N_ITERS = 300000
LR_G = 0.0002
LR_D = 0.0002
BETA_1 = 0.5
BETA_2 = 0.99
N_GF = 320
L_GF = 11
G_ITERS = 1
D_ITERS = 5


def train(path, dataloader):
    G = gen_models.snp_generator_2d_temp_2a(NZ, N_GF, L_GF).to("cuda")
    D = gen_models.snp_discriminator_2d().to("cuda")

    G.apply(utils.weights_init)
    optD = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()),
                      lr=LR_D,
                      betas=(BETA_1, BETA_2))
    optG = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()),
                      lr=LR_G,
                      betas=(BETA_1, BETA_2))

    noise = torch.zeros(BS, NZ).to("cuda")
    fixed_noise = utils.create_fixed_inputs(40, NZ)

    train_hist = utils.initialize_train_hist()

    for iteration in range(N_ITERS):
        ## GENERATOR ##
        for g_iter in range(G_ITERS):
            optG.zero_grad()
            noise.normal_(0, 1)
            fake = G(noise)
            pred_g = D(fake)
            if loss == "hinge":
                g_loss = -pred_g.mean()
                g_loss.backward()
            optG.step()

        ## DISCRIMINATOR ##
        for d_iter in range(D_ITERS):
            optD.zero_grad()
            batch = next(dataiter, None)
            if (batch is None) or (batch.size(0) != BS):
                dataiter = iter(dataloader)
                batch = dataiter.next()

            batch = batch.float().to("cuda")
            pred_real = D(batch)

            noise.normal_(0, 1)
            fake = G(noise)
            pred_fake = D(fake.detach())

            if loss == "hinge":
                d_loss_real = torch.nn.ReLU()(1.0 - pred_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()
                d_loss_total = d_loss_fake + d_loss_real
                d_loss_total.backward()

            optD.step()

        if iteration % 1000 == 0:
            train_hist = utils.update_train_hist(train_hist, d_loss_total, g_loss)

        if iteration % 2000 == 0:
            print('Iter: {0}, Dloss: {1}, Gloss: {2}'.format(iteration,
                                                             d_loss_total.item(),
                                                             g_loss.item()))
            img_path = path + "fake_data_experiment/"
            utils.save_imgs(G, BS, fixed_noise, iteration, img_path, one_dim=one_dim)

    model_path = path + "fake_data_experiment/"
    torch.save(G.state_dict(), model_path + "g.pth")
    torch.save(D.state_dict(), model_path + "d.pth")
    lplot_path = path + "fake_data_experiment/loss.png"
    utils.plot_loss(train_hist, lplot_path)


fake_data = np.load(path + 'fake_data_experiment/fake_data.npy')
fake_data = fake_data.reshape(-1, 1, 100, 4)
fake_dataset = FakeGCDataset(fake_data)
fake_dataloader = DataLoader(dataset=fake_dataset,
                             batch_size=BS,
                             shuffle=True)


path = "/home/pbromley/SynthSeqs/CompleteRun/"
train(path, fake_dataloader)
