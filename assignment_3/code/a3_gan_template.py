import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import time


#TODO add dropout during training


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_dim = args.latent_dim

        self.forward_pass = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
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
        print("\t1 epoch takes {} seconds".format((time.time() - track_time)/1000))
        print("\tTime to go: {} minutes".format(args.n_epochs-epoch * (time.time() - track_time)/1000/60))
        print("--------------------")
        track_time = time.time()

        for i, (imgs, _) in enumerate(dataloader):

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

            generator_accuracy = (DGz1.mean(dim=0).item(), DGz2.mean(dim=0).item())
            discriminator_accuracy = Dx.mean(dim=0).item()

            GL = G_loss.item()
            DL = D_loss.item()

            stats['G']['acc'].append(generator_accuracy)
            stats['G']['loss'].append(GL)

            stats['D']['acc'].append(discriminator_accuracy)
            stats['D']['loss'].append(DL)

            print_per_steps = 100
            if i % print_per_steps == 0:
                to_print = "Epoch: {}\tStep: {}\tLoss_D: {}\tLoss_G: {}\tD(x): {}\tD(G(z)): {} / {}".format(
                    epoch,
                    i,
                    DL,
                    GL,
                    discriminator_accuracy,
                    generator_accuracy[0],
                    generator_accuracy[1]
                )

                print(to_print)

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # TODO Anything to change here?
                Gz = Gz.reshape(-1, 1, 28, 28)
                save_image(Gz[:25],
                           'images/{}-{}.png'.format(str(int(time.time())) , batches_done),
                           nrow=5, normalize=True)


def main(args):
    # check if GPU is available. If not, use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
    args = parser.parse_args()

    main(args)
