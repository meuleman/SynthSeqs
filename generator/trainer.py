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
    """
    Class for training a DHS generative network using adversarial training.
    """
    def __init__(self,
                 iterations,
                 batch_size,
                 generator,
                 discriminator,
                 device,
                 data_dir):
        self.device = device

        # generator and discriminator should be uninitialized PyTorch models.
        self.generator_type = generator
        self.discriminator_type = discriminator

        self.iterations = iterations
        self.batch_size = batch_size

        # Path to where the numpy datasets live.
        self.data_dir = data_dir

    def setup(self,
              generator_params,
              discriminator_params,
              optimizer_params_g,
              optimizer_params_d):
        """
        Set up the generator and discriminator models, optimizers and data
        iterators.

        Notes
        -----
        ``generator_params`` and ``discriminator_params`` should be kwarg
        dictionaries with specified hyperparams.
        """
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
        """Reset the train dataloader.
        """
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
        """Plot a set of generated sequences.

        Notes
        -----
        This figure is pretty messy, but useful for seeing if the model has
        mode collapsed.
        """
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
        """Save the generator and discriminator models.
        """
        torch.save(self.generator.state_dict(),
                   model_dir + GENERATOR_MODEL_FILE) 
        torch.save(self.discriminator.state_dict(),
                   model_dir + DISCRIMINATOR_MODEL_FILE) 
 
