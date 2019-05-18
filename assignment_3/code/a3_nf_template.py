import argparse

import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid

import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """

    # TODO check implementation

    # take the formula for the normal distribution
    # set mu = 0, stdev = 1
    # take log
    # get the following!

    logp = (-x.pow(2) / 2) - torch.log(torch.sqrt(torch.tensor([2*np.pi]).to(DEVICE)))

    # to get the probability of the entire datapoint, we need the product of all probabilities
    # or, the sum of all log probabilities, yay computational efficiency

    logp = torch.sum(logp, dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    sample = torch.randn(size, device=DEVICE)
    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28 * 28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.

        # TODO add dimensions -- is this correct?
        self.shared_arch = torch.nn.Sequential(
            # Suggestion: Linear ReLU Linear ReLU Linear.

            nn.Linear(c_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.t = nn.Linear(n_hidden, c_in)

        self.s = nn.Sequential(
            nn.Linear(n_hidden, c_in),
            nn.Tanh()   # As suggested, we add a tanh for the scale
        )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.t.weight.data.zero_()
        self.t.bias.data.zero_()
        self.s[0].weight.data.zero_()
        self.s[0].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        masked_z = self.mask * z

        features = self.shared_arch(masked_z)

        t_out = self.t(features)
        s_out = self.s(features)


        if not reverse:
            # following equation (9) from the paper:
            # masked_z = b*x, z = x,
            z = masked_z + (1-self.mask)*(z * torch.exp(s_out) + t_out)

            # calculate the log determinant of the jacobian

            # TODO check calculation of ldj

            # determinant of jacobian is exp(sum(mask(s)))
            # so log determinant jacobian is sum(mask(s))
            # we keep track of the ldj throughout training
            #TODO check calculation, shouldn't be 1- right? But it only works this way...
            ldj = ldj + torch.sum((1-self.mask) * s_out, dim=1)

        else:
            # plugging the reverse equation from (8) into (9)
            z = masked_z + (1-self.mask) * ((z - t_out) * torch.exp(-s_out))

            # no need for LDJ tracking during inference time
            ldj = torch.zeros_like(ldj)


        return z, ldj  # ldj = log det Jacobian


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1 - mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z * (1 - alpha) + alpha * 0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1 - z), dim=1)
            z = torch.log(z) - torch.log(1 - z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1 - z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        z, ldj = self.flow(z,ldj, reverse=True)

        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    losses = []

    for i, (imgs, _) in enumerate(data):
        imgs = imgs.to(DEVICE)
        out = model.forward(imgs)

        loss = - torch.mean(out)

        losses.append(loss.data.item())

        if model.training:
            optimizer.zero_grad()
            loss.backward()

            # TODO clip norm required?

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

    avg_bpd = sum(losses) / len(losses) / 28**2 / math.log(2)

    return avg_bpd  # bpd = bits per dimension


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def save_images(model, epoch, train_bpd, val_bpd):

    current_time = str(int(time.time()))

    samples = model.sample(25).reshape(25, 1, 28, 28)

    grid = make_grid(samples, nrow=5, normalize=True)

    plt.imsave("./images_nfs/{}_epoch={}_train={}_val={}.jpg".format(
        current_time,
        epoch,
        int(train_bpd),
        int(val_bpd)
    ), grid.cpu().detach().numpy().transpose(1,2,0))




def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []

    epoch_times = []
    for epoch in range(ARGS.epochs):
        start_time = time.time()
        print("start epoch")
        bpds = run_epoch(model, data, optimizer)
        print("end epoch")
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        timeofepoch = time.time() - start_time
        epoch_times.append(timeofepoch)
        average_epoch_time = np.mean(epoch_times)

        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}. Epoch took {timeofepoch} seconds, approx {minutestogo} minutes to go".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd,
            timeofepoch=timeofepoch,
            minutestogo= ((ARGS.epochs - epoch)*average_epoch_time)/60


            ))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------

        save_images(model, epoch, train_bpd, val_bpd)



    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()

    main()
