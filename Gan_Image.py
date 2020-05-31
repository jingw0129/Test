import torch
import torchvision
from torch.autograd import Variable
import cv2
from torchvision import datasets, transforms
import numpy as np

epoch_n =10
batch_size = 128

torch.manual_seed(1)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#  randomly assign a torch tensor wrapped in pytorch variable, will be passed to generator net as the initial input
def rand_img(batch_size,output_size):
    noise = np.random.uniform(-1.,1., size=(batchs_ize, output_size))
    #Images which passed to the interface requires a single piece of image with 3 channels.
    noise = np.float32(noise)
    noise = torch.from_numpy(noise)
    noise = Variable(noise)
    return noise

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64 * 4 * 4, 1)
        )

    def forward(self, input):
        output = self.conv(input)
        output = output.view(-1, 64 * 4 * 4)
        output = self.dense(output)
        return output


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv_dense = torch.nn.Sequential(
            torch.nn.Linear(100, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.Linear(1024, 7 * 7 * 128),
            torch.nn.BatchNorm1d(num_features=7 * 7 * 128)
        )
        self.transpose_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, input):
        output = self.conv_dense(input)
        output = output.view(-1, 128, 7, 7)
        output = self.transpose_conv(output)
        return output

discriminator = Discriminator()

generator = Generator()

loss_f = torch.nn.BCEWithLogitsLoss()

gen_images = []

train_loss = []

optimizer_dis = torch.optim.Adam(discriminator.parameters(),lr=0.001)
optimizer_gen = torch.optim.Adam(generator.parameters(),lr=0.001)

for epoch in range(epoch_n):

    for batch in trainloader:
        #  data prepare
        X_train, y_train = batch
        X_train, y_train = Variable(X_train), Variable(y_train)

        noise = rand_img(batchsize=batchsize, output_size=100)
        # ----------------------------------------------------------------

        optimizer_dis.zero_grad()
        X_gen = generator(noise)
        X_gen = X_gen.view(-1, 1, 28, 28)
        X_train = X_train.view(-1, 1, 28, 28)

        d_x_train = discriminator(X_train)

        d_gen = discriminator(X_gen)
        # loss computation, sum up the loss from d_x_train and d_gen
        d_loss = loss_f(d_x_train, torch.ones_like(d_x_train) * (1 - 0.1)) + loss_f(d_gen, torch.zeros_like(d_gen))

        d_loss.backward(retain_graph=True)
        optimizer_dis.step()
        
        
        # -------------------------------------------------------------------------------------------
        
        optimizer_gen.zero_grad()

        noise = rand_img(batchsize=batchsize, output_size=100)
        X_gen = generator(noise)
        X_gen = X_gen.view(-1, 1, 28, 28)
        d_gen = discriminator(X_gen)
        g_loss = loss_f(d_gen, torch.ones_like(d_gen))
        
        g_loss.backward()
        optimizer_gen.step()

    train_loss.append((d_loss, g_loss))

    gen_img = generator(noise)
    gen_images.append(gen_img)


for i in range(len(gen_images)):
    img = cv2.imread(gen_images[i])
    cv2.imshow('gan_image', img)
    cv2.waitKey(0)
