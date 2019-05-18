import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import numpy as np

import time


#TODO add dropout during training


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.forward_pass = nn.Sequential(
            # add 3x dropout
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        out = self.forward_pass(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        out = self.forward_pass(img)
        return out

# custom weights initialization called on netG and netD
# as taken from pytorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def sample(generator, device):
    n = 25
    z = torch.randn((n, generator.latent_dim), device=device)
    Gz = generator(z)
    save_image(Gz,
               'images/{}-2-6.png'.format(str(int(time.time()))),
               nrow=5, normalize=True)





def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):

    stats = {}

    stats['G'] = {}
    stats['D'] = {}

    stats['G']['loss'] = []
    stats['G']['acc'] = []

    stats['D']['loss'] = []
    stats['D']['acc'] = []

    track_time = time.time()

    for epoch in range(args.n_epochs):
        # time tracking of how long en epoch takes
        print("--------------------")
        print("\t1 epoch takes {} seconds".format((time.time() - track_time)))
        print("\tTime to go: {} minutes".format((args.n_epochs-epoch) * (time.time() - track_time)/60))
        print("--------------------")
        print("Generator Accuracy = {}".format(stats['G']['acc']))
        print("Generator Loss = {}".format(stats['G']['loss']))
        print("Discriminator Accuracy = {}".format(stats['D']['acc']))
        print("Discriminator Loss = {}".format(stats['D']['loss']))
        print("--------------------")
        track_time = time.time()

        for i, (imgs, _) in enumerate(dataloader):

            if torch.cuda.is_available():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')

            x = imgs.to(device).reshape(-1, 784)

            # if we're at the end of training and batch size is not nice...
            # if x.shape[0] != args.batch_size:
            #     break

            # Train Discriminator
            # -------------------

            # TODO how long to train?
            optimizer_D.zero_grad()

            z = torch.randn((x.shape[0], args.latent_dim), device=device)
            # z = torch.randn((args.batch_size, args.latent_dim), device=device)

            Dx = discriminator(x)
            Gz = generator(z)
            DGz2 = discriminator(Gz)

            D_loss = -1 * (torch.log(Dx) + torch.log(1 - DGz2)).mean(dim=0)

            D_loss.backward()

            optimizer_D.step()

            # Train Generator
            # ---------------

            # TODO how long / often to train?

            z = torch.randn((x.shape[0], args.latent_dim), device=device)
            # z = torch.randn((args.batch_size, args.latent_dim), device=device)

            DGz1 = discriminator(generator(z))

            G_loss = (-1 * torch.log(DGz1)).mean(dim=0)

            optimizer_G.zero_grad()

            G_loss.backward()

            optimizer_G.step()



            # Save and print some stats
            # ----

            # we expect the generator accuracy to go from 0 to 0.5
            # we expect the discriminator accuracy to go from 1 to 0.5
            # we would like to plot the losses over time
            # we would like to

            generator_accuracy = np.mean((DGz1.mean(dim=0).item(), DGz2.mean(dim=0).item()))
            discriminator_accuracy = Dx.mean(dim=0).item()

            GL = G_loss.item()
            DL = D_loss.item()



            print_per_steps = 100
            if i % print_per_steps == 0:
                to_print = "Epoch: {}\tStep: {}\tLoss_D: {}\tLoss_G: {}\tD(x): {}\tD(G(z)): {}".format(
                    epoch,
                    i,
                    DL,
                    GL,
                    discriminator_accuracy,
                    generator_accuracy
                )

                print(to_print)

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                stats['G']['acc'].append(generator_accuracy)
                stats['G']['loss'].append(GL)

                stats['D']['acc'].append(discriminator_accuracy)
                stats['D']['loss'].append(DL)
                # TODO Anything to change here?
                Gz = Gz.reshape(-1, 1, 28, 28)
                save_image(Gz[:25],
                           'images/{}-{}.png'.format(str(int(time.time())) , batches_done),
                           nrow=5, normalize=True)
            torch.set_default_tensor_type('torch.FloatTensor')


def main(args):
    # check if GPU is available. If not, use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data -- fixed the Normalize dimensions
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    # reinitialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)



    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), str(int(time.time())) + "_mnist_generator.pt")



def interpolate():
    generator = Generator()
    generator.load_state_dict(torch.load('../gan_images/mnist_generator.pt', map_location='cpu'))
    for i in range(10):
        a = torch.randn(1,100)
        print(a)
        generator.eval()
        Gz = generator(a)
        Gz = Gz.reshape(-1, 1, 28, 28)
        save_image(Gz[:100],
                   'images/{}-{}.png'.format("check_figs_for_interpolation",i),
                   nrow=1, normalize=True)
    one = torch.tensor([1.2564,  0.8414,  0.1255, -0.5333, -0.6871, -0.1747, -0.4613, -0.2510,
          0.0960,  1.4093, -1.8332, -0.3168, -1.3448,  0.3383,  0.6217,  1.1507,
         -0.4281, -1.7358, -2.1052, -2.5599,  0.7375, -0.3222, -1.3106,  0.8149,
          0.5920,  1.2080, -0.1601,  0.0254, -0.1127, -1.2389,  1.0707, -0.6668,
          0.7448, -1.7727,  0.8835, -0.1735,  0.0457, -0.6953, -0.2177, -0.5276,
          1.3043, -1.3637,  1.3787, -0.5469, -0.3293, -0.0042,  1.5517,  0.1470,
         -0.8158,  1.1559, -0.6233, -0.8261,  0.8103,  0.9109,  0.0640,  1.1176,
         -0.6638,  0.5052,  0.0482,  0.1589, -0.1846, -0.9857,  0.5391, -1.1274,
          0.1466,  1.1127, -0.0119,  1.9409, -0.8361,  1.7848, -0.2336,  1.5701,
         -1.4049,  0.6410, -0.3872, -1.1225, -0.3526, -0.4586,  0.8969, -0.6610,
          0.8864,  1.8571,  2.8588,  0.1514,  0.7104,  1.6074, -0.1879,  0.0432,
         -1.0296, -0.4899, -0.2852, -0.7578, -0.4325,  0.8677, -0.1438, -0.1566,
         -0.5723,  0.5766, -0.9390, -0.1232])

    four = torch.tensor([-0.2105, -0.0654, -0.2802,  0.1469, -1.3909, -0.4472,  0.4429, -0.9437,
          0.6072, -1.8271, -0.4202, -0.3476, -0.8078, -0.5621, -1.6063, -0.2707,
          0.8154, -1.2030,  0.7389,  0.6329, -0.6455,  1.1768, -0.4836, -1.4237,
         -0.0642, -0.0297,  0.1303,  0.3108,  0.0700,  0.9920, -0.5220, -0.9031,
          0.6444, -0.2111, -0.0803, -0.8090,  1.3974,  0.7275, -0.2941,  0.5881,
          1.3565,  2.6986, -0.4989,  0.7162,  0.4580, -0.2082, -0.2922, -0.0848,
          0.4112, -1.2816,  1.6787,  0.1645,  0.8400,  0.3280,  0.2643, -1.9202,
          0.4570, -0.3416,  1.8842, -0.3578,  0.6817, -0.3741,  0.8443, -1.4836,
         -0.5232, -1.0516, -0.7320,  1.5984,  0.2249,  0.8976, -1.1413,  0.5177,
          0.3222, -0.6003, -0.1775,  0.9282, -0.1270, -1.3756, -0.6135,  0.0309,
          1.7447, -2.9688,  1.1458, -0.3751,  0.4037, -0.1922, -0.9678, -0.2509,
         -0.0566, -0.2371,  0.7322, -0.5133,  0.3741,  0.7375,  0.4940, -3.4541,
         -0.8005,  0.7474,  1.1390, -0.4337])

    diff = four - one
    steps = 8
    step = diff/steps
    z_interpolation = torch.stack([one + i*step for i in range(steps+1)])
    gen_interpolation = generator(z_interpolation)
    gen_interpolation = gen_interpolation.reshape(-1, 1, 28, 28)
    save_image(gen_interpolation,
               'images/{}-{}.png'.format("interpolation", "1"),
               nrow=9, normalize=True
               )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--interpolate', type=int, default=0,
                        help='set if you want to run interpolation')
    args = parser.parse_args()

    if args.interpolate == 1:
        interpolate()
    else:
        main(args)
