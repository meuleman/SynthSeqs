import gen_models
import data_helper
import utils
import argparse
import torch
import torch.optim as optim


def calc_gradient_penalty(netD, real_data, fake_data, bs, lmbd):
    alpha = torch.rand(bs, 1)
    alpha = alpha.expand(bs, int(real_data.nelement()/bs)).contiguous()
    alpha = alpha.view(bs, 4, 100)
    alpha = alpha.to("cuda")

    fake_data = fake_data.view(bs, 4, 100)
    real_data = real_data.view(bs, 4, 100)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to("cuda")
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to("cuda"),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbd
    return gradient_penalty

def count_parameters(model):
    #return print(p.numel() for p in self.model.parameters() if p.requires_grad)
    for p in model.parameters():
        if p.requires_grad:
            print(p.numel())


def train(path, loss="hinge", one_dim=False, data_type="strong", **kwargs):
    BS = kwargs.pop('bs', 128)
    NZ = kwargs.pop('nz', 100)
    N_ITERS = kwargs.pop('niter', 300000)
    LR_G = kwargs.pop('lrG', 0.0002)
    LR_D = kwargs.pop('lrD', 0.0002)
    BETA_1 = kwargs.pop('beta1', 0.5)
    BETA_2 = kwargs.pop('beta2', 0.99)
    N_GF = kwargs.pop('ngf', 320)
    L_GF = kwargs.pop('lgf', 11)
    G_ITERS = kwargs.pop('g_iters', 1)
    D_ITERS = kwargs.pop('d_iters', 5)
    dataloaders = data_helper.get_the_dataloaders(BS, weighted_sample=False)
    dataloader = dataloaders['train_seqs']
    dataiter = iter(dataloader)

    if one_dim:
        G = gen_models.snp_generator_2d_temp(NZ, N_GF, L_GF).to("cuda")
        D = gen_models.resnet_discriminator_1d(spec_norm=True).to("cuda")
    else:
        G = gen_models.snp_generator_2d_temp_2a(NZ, N_GF, L_GF).to("cuda")
        D = gen_models.snp_discriminator_2d().to("cuda")
    G.apply(utils.weights_init)
    optD = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=LR_D, betas=(BETA_1, BETA_2))
    optG = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=LR_G, betas=(BETA_1, BETA_2))

    noise = torch.zeros(BS, NZ).to("cuda")
    fixed_noise = utils.create_fixed_inputs(40, NZ)
    mone = torch.FloatTensor([-1]).to("cuda")

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
            else:
                g_loss = pred_g.mean()
                g_loss.backward(mone)
                g_loss = -g_loss
            optG.step()

        ## DISCRIMINATOR ##
        for d_iter in range(D_ITERS):
            optD.zero_grad()
            batch = next(dataiter, None)
            if (batch is None) or (batch[0].size(0) != BS):
                dataiter = iter(dataloader)
                batch = dataiter.next()
            x, _ = batch
            x = x.float().to("cuda")
            pred_real = D(x)

            noise.normal_(0, 1)
            fake = G(noise)
            pred_fake = D(fake.detach())

            if loss == "hinge":
                d_loss_real = torch.nn.ReLU()(1.0 - pred_real).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()
                d_loss_total = d_loss_fake + d_loss_real
                d_loss_total.backward()
            else:
                d_loss_real = pred_real.mean()
                d_loss_fake = pred_fake.mean()
                gradient_penalty = calc_gradient_penalty(D, x, fake.detach(), BS, 10)
                d_loss_total = d_loss_fake - d_loss_real + gradient_penalty
                d_loss_total.backward()
                w_dist = d_loss_fake - d_loss_real

            optD.step()

        if iteration % 1000 == 0:
            train_hist = utils.update_train_hist(train_hist, d_loss_total, g_loss)

        if iteration % 2000 == 0:
            if loss == "hinge":
                print('Iter: {0}, Dloss: {1}, Gloss: {2}'.format(iteration, d_loss_total.item(), g_loss.item()))
            else:
                print('Iter: {0}, Dloss: {1}, Gloss: {2}, WDist: {3}'.format(iteration, d_loss_total.item(), g_loss.item(), w_dist.item()))
            img_path = path + "images/gan/"
            utils.save_imgs(G, BS, fixed_noise, iteration, img_path, one_dim=one_dim)

    model_path = path + "saved_models/gan"
    torch.save(G.state_dict(), model_path + "-g.pth")
    torch.save(D.state_dict(), model_path + "-d.pth")
    lplot_path = path + "loss_plots/gan.png"
    utils.plot_loss(train_hist, lplot_path)


def run(path, loss="hinge", one_dim=False, data_type="strong"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128, help='input batch size')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for G, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for D, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam. default=0.99')
    parser.add_argument('--ngf', type=int, default=320, help='number of generator filters')
    parser.add_argument('--lgf', type=int, default=11, help='length of generator filters')
    parser.add_argument('--g_iters', type=int, default=1, help='number of generator iters per global iter')
    parser.add_argument('--d_iters', type=int, default=5, help='number of discriminator iters per global iter')

    opt = parser.parse_args()
    train(path, loss=loss, one_dim=one_dim, data_type=data_type, **vars(opt))


if __name__ == "__main__":
    path = "/home/pbromley/generative_dhs/"
    run(path, loss="hinge", one_dim=False, data_type="mean_signal")
