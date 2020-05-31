import torch
import torchvision
from mnist_net import Net
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# -------------------------------------------------------------------------------------------

epochs = 15
net = Net()
torch.manual_seed(1)

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)


def train(epochs):
    time0 = time()

    for e in range(epochs):
        train_loss = 0
        net.train()
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Training pass
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)

            # model learns by backpropagati
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            train_loss += loss.item()

            print("Epoch {} - Training loss: {}".format(e, train_loss / len(trainloader)))
            print("\nTraining Time (in minutes) =", (time() - time0) / 60)
        # save model every 3 times iteration
        if e % 3 == 0:
            save_dict = {

                "net": net.state_dict()

            }
            torch.save(save_dict, os.path.join('mnist_model', str(e) + 'model.pth'))

def test():
  correct = 0
  total =  0

  for images, labels in valloader:
      for i in range(len(labels)):
          img = images[i].view(1, 784)
          with torch.no_grad():
              out = net(img)

          ps = torch.exp(out)
          probab = list(ps.numpy()[0])
          pred_label = probab.index(max(probab))
          true_label = labels.numpy()[i]
          if (true_label == pred_label):
              correct += 1
          total += 1

  print("Number Of Images Tested =", total)
  print("\nModel Accuracy =", (correct / total))

test()

train(30)
test()
