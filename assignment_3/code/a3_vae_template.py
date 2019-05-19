# Run with python 3.6.3

import argparse

import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')  # was required for my local python setup
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

import time  # used for time tracking of epochs
import numpy as np
import math


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.input_dim = 784

        # shared feature learning network
        self.feature_learning = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        )

        # separate layer for stdev
        self.linear_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU()  # added to ensure that the standard deviation is
            # never negative
        )
        self.linear_mean = nn.Linear(hidden_dim, z_dim)  # separate layer for stdev

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

        self.out_dim = 784  # output an image

        self.total = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),  # map from latent to hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),  # map to image dimension
            nn.Sigmoid()  # ensures that the output is a probability used as a mean for
            # bernoulli distribution
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
        mean, std = self.encoder.forward(input)  # get mean and stdev from encoder

        epsilons = torch.randn(std.size())  # parametrization trick, sample epsilon

        z = epsilons * std + mean  # parametriation trick

        reconstruction = self.decoder.forward(z)  # pass latent representation to decoder

        rec_loss = self.calc_rec_loss(reconstruction, input)  # calculate reconstruction loss

        reg_loss = self.calc_reg_loss(mean, std)  # calculate regularization loss

        average_negative_elbo = reg_loss + rec_loss  # combine for total loss

        return average_negative_elbo

    def calc_reg_loss(self, mean, std):
        eps = 1e-8  # added for numerical stability, might stdev be 0
        L_reg_per_pixel = torch.log(1 / (std + eps)) + (std.pow(2) + mean.pow(2)) / (2) - 0.5  # calc KL divergence
        mean_L_reg_per_pixel = torch.mean(L_reg_per_pixel, dim=0)  # take mean of batch
        L_reg = torch.sum(mean_L_reg_per_pixel)  # sum the loss per pixel into one loss
        return L_reg

    def calc_rec_loss(self, reconstruction, input):
        bernoulli_stats = input * torch.log(reconstruction) + (1 - input) * torch.log(
            1 - reconstruction)  # bernoulli loss
        average_bernoulli_stats = torch.mean(bernoulli_stats, dim=0)  # batch mean
        L_rec = - torch.sum(average_bernoulli_stats)  # sum
        return L_rec

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        zs = torch.randn((n_samples, self.z_dim))  # get n samples * laten feature dimension random vairables
        constructions = self.decoder.forward(zs)  # pass it to the decoder

        # get output, which are bernoulli mean
        im_means = constructions.reshape(
            (constructions.size()[0], 1, int(np.sqrt(constructions.size()[1])), int(np.sqrt(constructions.size()[1]))))
        sampled_ims = torch.bernoulli(im_means)  # sample using bernoulli distribtion the binary images
        return sampled_ims, im_means

    def sample_manifold(self, n_samples, start, stop):
        xy = torch.linspace(start, stop, n_samples)

        # we can't use torch.distributions. Below is a workaround to get a similar function
        # https://pytorch.org/docs/stable/torch.html#torch.erf
        # as the inverse error has a different normalizing parameter than the normal distribution,
        # we recalculate is a bit
        grid = [torch.erfinv(2 * torch.tensor([x, y], device='cpu') - 1) * math.sqrt(2) for x in xy for y in xy]
        zs = torch.stack(grid)

        # create all the image means
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
        mean_elbo = model.forward(batch)  # get the elbo by going through a forward
        training_elbos.append(mean_elbo.item())
        if model.training:  # only train if in training mode
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
    """
    puts all given samplse in a grid and saves it in a directory
    :param samples:  has to be a number of samples that has a square root
    :param train_elbo: used to save with the image
    :param val_elbo:  used to save with the image
    :param epoch: used to save with the image
    :param binaryormean: "manifold", "binary", or "mean". used to create image name and decide which directory to save
    :return: nothing, just saves
    """
    grid = make_grid(samples, int(np.sqrt(samples.size()[0])))
    timestamp = int(time.time())
    if binaryormean == "manifold":
        saveas = "vae-samples/{}/{}_zdim={}_epoch={}.png".format(
            binaryormean,
            timestamp,
            ARGS.zdim,
            epoch

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

    plt.imsave(saveas, grid.detach().numpy().transpose(1, 2, 0))


def main():
    """
    Runs all code
    :return:
    """
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

        # save samples during training ---------------------------------------
        binary_samples, mean_samples = model.sample(25)
        save_samples(binary_samples, train_elbo, val_elbo, epoch, "binary")
        save_samples(mean_samples, train_elbo, val_elbo, epoch, "mean")
        # --------------------------------------------------------------------

        # Plot manifold if number of laten dimensions is 2 -------------------
        if ARGS.zdim == 2:
            manifold_samples = model.sample_manifold(25, 0.05, 0.95)
            save_samples(manifold_samples, train_elbo=0, val_elbo=0, epoch=epoch, binaryormean="manifold")
        # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


def interpolate():
    """
    manifold interpolation function used to create the image outside of training
    instead, put the model you want to load below, and it will generate an an interpolation image
    """
    model = VAE(z_dim=ARGS.zdim)
    model.load_state_dict(torch.load('1558205580vae_z2.pt', map_location='cpu'))
    manifold_samples = model.sample_manifold(25, 0.05, 0.95)
    save_samples(manifold_samples, train_elbo=0, val_elbo=0, epoch=0, binaryormean="manifold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--interpolate', default=0, type=int,
                        help='1 if you want to interpolate with a given model')

    ARGS = parser.parse_args()

    if ARGS.interpolate == 1:
        interpolate()

    else:
        main()
