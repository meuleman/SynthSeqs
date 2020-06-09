from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

from utils.constants import (
    DISCRIMINATOR_MODEL_FILE,
    GENERATOR,
    GENERATOR_MODEL_FILE,
)
from utils.data import DHSDataLoader
from utils.net import weights_init
from utils.seq import one_hot_to_seq

from .constants import D_ITERS, G_ITERS, NZ

class GeneratorTrainer:
    def __init__(self,
                 iterations,
                 batch_size,
                 generator,
                 discriminator,
                 device,
                 data_dir):
        self.device = device
        self.generator_type = generator
        self.discriminator_type = discriminator
        self.iterations = iterations
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self,
              generator_params,
              discriminator_params,
              optimizer_params_g,
              optimizer_params_d):
        self.generator = (
            self.generator_type(**generator_params)
                .to(self.device)
        )
        self.discriminator = (
            self.discriminator_type(**discriminator_params)
                .to(self.device)
        )

        self.generator.apply(weights_init)

        self.opt_g = Adam(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            **optimizer_params_g,
        )
        self.opt_d = Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            **optimizer_params_d,
        )

        self.dataloaders = DHSDataLoader(self.batch_size,
                                         GENERATOR,
                                         self.data_dir)

        self.noise = torch.zeros(self.batch_size, NZ).to(self.device)

    def generator_loss(self, pred_g): 
        return -pred_g.mean()

    def discriminator_loss(self, pred_real, pred_fake):    
        d_loss_real = nn.ReLU()(1.0 - pred_real).mean()
        d_loss_fake = nn.ReLU()(1.0 + pred_fake).mean()
        return d_loss_fake + d_loss_real

    def new_data_iterator(self):
        return iter(self.dataloaders.train)

    def train(self,
              generator_params,
              discriminator_params,
              optimizer_params_g,
              optimizer_params_d):

        self.setup(generator_params,
                   discriminator_params,
                   optimizer_params_g,
                   optimizer_params_d)

        data_iter = self.new_data_iterator()

        for iteration in range(self.iterations):
            ## GENERATOR ##
            for g_iter in range(G_ITERS):
                self.opt_g.zero_grad()
                self.noise.normal_(0, 1)

                fake = self.generator(self.noise)
                pred_g = self.discriminator(fake)

                g_loss = self.generator_loss(pred_g) 
                g_loss.backward()
                self.opt_g.step()

            ## DISCRIMINATOR ##
            for d_iter in range(D_ITERS):
                self.opt_d.zero_grad()
                seqs, _ = next(data_iter, (None, None))

                if seqs is None: 
                    data_iter = self.new_data_iterator()
                    seqs, _ = data_iter.next()

                seqs = seqs.float().to(self.device).unsqueeze(1)
                pred_real = self.discriminator(seqs)

                self.noise.normal_(0, 1)
                fake = self.generator(self.noise)
                pred_fake = self.discriminator(fake.detach())

                d_loss = self.discriminator_loss(pred_real, pred_fake)
                d_loss.backward()

                self.opt_d.step()

    def plot_seqs(self, num_seqs, filename, figure_dir):
        self.generator.train(False)
        noise = torch.Tensor(num_seqs, NZ).normal_(0, 1).to(self.device)

        with torch.no_grad():
            pred = self.generator(noise).detach().cpu().numpy().squeeze()

        seqs = [one_hot_to_seq(one_hot) for one_hot in pred]

        plt.figure(figsize=(18, 20))
        for i in range(num_seqs):
            plt.text(0, i/num_seqs, seqs[i], fontsize=14)
        plt.savefig(figure_dir + filename)
        plt.close()

    def save(self, model_dir):
        torch.save(self.generator.state_dict(),
                   model_dir + GENERATOR_MODEL_FILE) 
        torch.save(self.discriminator.state_dict(),
                   model_dir + DISCRIMINATOR_MODEL_FILE) 
 

#def train(path, loss="hinge", one_dim=False, data_type="strong", **kwargs):
#    BS = kwargs.pop('bs', 128)
#    NZ = kwargs.pop('nz', 100)
#    N_ITERS = kwargs.pop('niter', 300000)
#    LR_G = kwargs.pop('lrG', 0.0002)
#    LR_D = kwargs.pop('lrD', 0.0002)
#    BETA_1 = kwargs.pop('beta1', 0.5)
#    BETA_2 = kwargs.pop('beta2', 0.99)
#    N_GF = kwargs.pop('ngf', 320)
#    L_GF = kwargs.pop('lgf', 11)
#    G_ITERS = kwargs.pop('g_iters', 1)
#    D_ITERS = kwargs.pop('d_iters', 5)
#    dataloaders = data_helper.get_the_dataloaders(BS, weighted_sample=False)
#    dataloader = dataloaders['train_seqs']
#    dataiter = iter(dataloader)
#
#    if one_dim:
#        G = gen_models.snp_generator_2d_temp(NZ, N_GF, L_GF).to("cuda")
#        D = gen_models.resnet_discriminator_1d(spec_norm=True).to("cuda")
#    else:
#        G = gen_models.snp_generator_2d_temp_2a(NZ, N_GF, L_GF).to("cuda")
#        D = gen_models.snp_discriminator_2d().to("cuda")
#    G.apply(utils.weights_init)
#    optD = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=LR_D, betas=(BETA_1, BETA_2))
#    optG = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=LR_G, betas=(BETA_1, BETA_2))
#
#    noise = torch.zeros(BS, NZ).to("cuda")
#    fixed_noise = utils.create_fixed_inputs(40, NZ)
#
#    train_hist = utils.initialize_train_hist()
#
#    for iteration in range(N_ITERS):
#        ## GENERATOR ##
#        for g_iter in range(G_ITERS):
#            optG.zero_grad()
#            noise.normal_(0, 1)
#            fake = G(noise)
#            pred_g = D(fake)
#            if loss == "hinge":
#                g_loss = -pred_g.mean()
#                g_loss.backward()
#            optG.step()
#
#        ## DISCRIMINATOR ##
#        for d_iter in range(D_ITERS):
#            optD.zero_grad()
#            batch = next(dataiter, None)
#            if (batch is None) or (batch[0].size(0) != BS):
#                dataiter = iter(dataloader)
#                batch = dataiter.next()
#            x, _ = batch
#            x = x.float().to("cuda")
#            pred_real = D(x)
#
#            noise.normal_(0, 1)
#            fake = G(noise)
#            pred_fake = D(fake.detach())
#
#            if loss == "hinge":
#                d_loss_real = torch.nn.ReLU()(1.0 - pred_real).mean()
#                d_loss_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()
#                d_loss_total = d_loss_fake + d_loss_real
#                d_loss_total.backward()
#            else:
#                d_loss_real = pred_real.mean()
#                d_loss_fake = pred_fake.mean()
#                gradient_penalty = calc_gradient_penalty(D, x, fake.detach(), BS, 10)
#                d_loss_total = d_loss_fake - d_loss_real + gradient_penalty
#                d_loss_total.backward()
#                w_dist = d_loss_fake - d_loss_real
#
#            optD.step()
#
#        if iteration % 1000 == 0:
#            train_hist = utils.update_train_hist(train_hist, d_loss_total, g_loss)
#
#        if iteration % 2000 == 0:
#            if loss == "hinge":
#                print('Iter: {0}, Dloss: {1}, Gloss: {2}'.format(iteration, d_loss_total.item(), g_loss.item()))
#            else:
#                print('Iter: {0}, Dloss: {1}, Gloss: {2}, WDist: {3}'.format(iteration, d_loss_total.item(), g_loss.item(), w_dist.item()))
#            img_path = path + "images/gan/"
#            utils.save_imgs(G, BS, fixed_noise, iteration, img_path, one_dim=one_dim)
#
#    model_path = path + "saved_models/gan"
#    torch.save(G.state_dict(), model_path + "-g.pth")
#    torch.save(D.state_dict(), model_path + "-d.pth")
#    lplot_path = path + "loss_plots/gan.png"
#    utils.plot_loss(train_hist, lplot_path)
#
#
#def run(path, loss="hinge", one_dim=False, data_type="strong"):
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--bs', type=int, default=128, help='input batch size')
#    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
#    parser.add_argument('--niter', type=int, default=300000, help='number of iterations to train for')
#    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for G, default=0.0002')
#    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for D, default=0.0002')
#    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam. default=0.99')
#    parser.add_argument('--ngf', type=int, default=320, help='number of generator filters')
#    parser.add_argument('--lgf', type=int, default=11, help='length of generator filters')
#    parser.add_argument('--g_iters', type=int, default=1, help='number of generator iters per global iter')
#    parser.add_argument('--d_iters', type=int, default=5, help='number of discriminator iters per global iter')
#
#    opt = parser.parse_args()
#    train(path, loss=loss, one_dim=one_dim, data_type=data_type, **vars(opt))
#
#
#if __name__ == "__main__":
#    path = "/home/pbromley/generative_dhs/peterbromley/CompleteRun/"
#    run(path, loss="hinge", one_dim=False, data_type="mean_signal")
