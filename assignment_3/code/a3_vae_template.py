import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

import time

import numpy as np

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.input_dim = 784

        self.feature_learning = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        feature_representation = self.feature_learning(input)

        mean = self.linear_mean(feature_representation)
        std = self.linear_std(feature_representation)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.out_dim = 784

        self.total = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        mean = self.total(input)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder.forward(input)

        epsilons = torch.randn(std.size())

        z = epsilons * std + mean

        reconstruction = self.decoder.forward(z)

        rec_loss = self.calc_rec_loss(reconstruction, input)

        reg_loss = self.calc_reg_loss(mean, std)

        average_negative_elbo = reg_loss + rec_loss

        return average_negative_elbo

    def calc_reg_loss(self, mean, std):
        eps = 1e-8
        L_reg_per_pixel = torch.log(1/(std+eps)) + (std.pow(2) + mean.pow(2))/(2) - 0.5
        mean_L_reg_per_pixel = torch.mean(L_reg_per_pixel, dim=0)
        L_reg = torch.sum(mean_L_reg_per_pixel)
        return L_reg

    def calc_rec_loss(self, reconstruction, input):
        bernoulli_stats = input * torch.log(reconstruction) + (1-input) * torch.log(1-reconstruction)
        average_bernoulli_stats = torch.mean(bernoulli_stats, dim=0)
        L_rec = - torch.sum(average_bernoulli_stats)
        return L_rec

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        zs = torch.randn((n_samples, self.z_dim))
        constructions = self.decoder.forward(zs)
        im_means = constructions.reshape((constructions.size()[0], 1, int(np.sqrt(constructions.size()[1])), int(np.sqrt(constructions.size()[1]))))
        sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means

    def sample_manifold(self, n_samples, start, stop):
        one_d_grid = torch.linspace(start, stop, n_samples)

        grid = [torch.tensor([x, y]) for x in one_d_grid for y in one_d_grid]

        zs = torch.stack(grid)

        constructions = self.decoder.forward(zs)
        im_means = constructions.reshape(
            (constructions.size()[0], 1, int(np.sqrt(constructions.size()[1])), int(np.sqrt(constructions.size()[1]))))
        return im_means




def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    training_elbos = []

    for batch in data:
        batch = batch.reshape(batch.size()[0], batch.size(2) * batch.size(3))
        mean_elbo = model.forward(batch)
        training_elbos.append(mean_elbo.item())
        if model.training:
            model.zero_grad()
            mean_elbo.backward()
            optimizer.step()

    average_epoch_elbo = torch.tensor(training_elbos).mean()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save_samples(samples, train_elbo, val_elbo, epoch, binaryormean):
    grid = make_grid(samples, int(np.sqrt(samples.size()[0])))
    timestamp = int(time.time())
    if binaryormean == "manifold":
        saveas = "vae-samples/{}/{}_zdim={}.png".format(
            binaryormean,
            timestamp,
            ARGS.zdim,

        )
    else:
        saveas = "vae-samples/{}/{}_z={}_epoch={}_train={}_val={}.png".format(
            binaryormean,
            timestamp,
            ARGS.zdim,
            epoch,
            train_elbo,
            val_elbo
        )

    plt.imsave(saveas, grid.detach().numpy().transpose(1,2,0))


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    train_curve, val_curve = [], []

    # Plot before training -------------------------------
    binary_samples, mean_samples = model.sample(25)
    save_samples(binary_samples, train_elbo=0, val_elbo=0, epoch=0, binaryormean="binary")
    save_samples(mean_samples, train_elbo=0, val_elbo=0, epoch=0, binaryormean="binary")
    # ----------------------------------------------------

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.

        binary_samples, mean_samples = model.sample(25)

        save_samples(binary_samples, train_elbo, val_elbo, epoch, "binary")
        save_samples(mean_samples, train_elbo, val_elbo, epoch, "mean")


        # --------------------------------------------------------------------

    # TODO
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.

    if ARGS.zdim == 2:
        manifold_samples = model.sample_manifold(25,-3,3)
        save_samples(manifold_samples, train_elbo=0, val_elbo=0, epoch=0, binaryormean="manifold")

    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
